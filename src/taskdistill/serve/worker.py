"""The model worker: one dedicated thread owns the student model.

MLX (0.31.2 and later) keeps a default GPU stream per thread, and unevaluated graphs and explicitly created
streams are bound to the thread that made them. So the backend is loaded, the label trie is built and every
generation runs on one thread, fed by a queue. Jobs run one at a time in submission order, so generations
never interleave, and only plain Python values leave the thread.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal, ParamSpec, TypeVar

from taskdistill.backends.base import Backend
from taskdistill.config import TaskSpec
from taskdistill.predict import Prediction, predict

log = logging.getLogger("taskdistill.serve")

T = TypeVar("T")
P = ParamSpec("P")
WorkerStatus = Literal["idle", "loading", "ready", "error", "stopped"]


class ModelNotReady(RuntimeError):
    """The student model failed to load, or the worker has been stopped."""


@dataclass
class _Job:
    future: Future[Any]
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class ModelWorker:
    """Runs ``loader()``, ``backend.load()``, the label trie build and every job on one thread.

    ``loader`` returns the (not yet loaded) backend; it is called on the worker thread by :meth:`start`.
    A load failure is kept in :attr:`load_error` and every job then fails with :class:`ModelNotReady`.
    """

    def __init__(self, loader: Callable[[], Backend], *, spec: TaskSpec, name: str = "taskdistill-model") -> None:
        self._loader = loader
        self.spec = spec
        self.name = name
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._loaded = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False
        self.backend: Backend | None = None
        self.trie: Any = None
        self.load_error: BaseException | None = None
        self.load_ms: float | None = None
        self.thread_id: int | None = None

    # lifecycle ----------------------------------------------------------------------------------
    def start(self) -> None:
        """Start the worker thread (idempotent); loading begins on it at once."""
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._stopped:
            raise ModelNotReady("the model worker has been stopped")
        if self._thread is None:
            self._thread = threading.Thread(target=self._main, name=self.name, daemon=True)
            self._thread.start()

    def stop(self, timeout: float | None = 30.0) -> None:
        """Finish the jobs already queued, then end the thread. Later submissions raise :class:`ModelNotReady`."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
            if thread is not None:
                self._jobs.put(None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until loading has finished (or ``timeout`` passed); True when the model is ready."""
        self._loaded.wait(timeout)
        return self.ready

    @property
    def ready(self) -> bool:
        return self._loaded.is_set() and self.load_error is None and self.backend is not None

    @property
    def status(self) -> WorkerStatus:
        if self._stopped:
            return "stopped"
        if self._thread is None:
            return "idle"
        if not self._loaded.is_set():
            return "loading"
        return "ready" if self.load_error is None else "error"

    @property
    def backend_name(self) -> str | None:
        return None if self.backend is None else str(self.backend.name)

    # jobs ---------------------------------------------------------------------------------------
    def submit(self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        """Queue ``fn(*args, **kwargs)`` for the worker thread (starting it if needed)."""
        future: Future[T] = Future()
        with self._lock:
            self._start_locked()
            self._jobs.put(_Job(future, fn, args, kwargs))
        return future

    def predict_now(self, input_text: str) -> Prediction:
        """The student's prediction for ``input_text``; call only on the worker thread (see :meth:`run`)."""
        if self.backend is None:
            raise ModelNotReady(f"the student model is not loaded: {self.load_error or 'loading did not finish'}")
        return predict(self.spec, self.backend, input_text, trie=self.trie)

    async def run(self, input_text: str) -> Prediction:
        """Predict on the worker thread without blocking the event loop."""
        return await asyncio.wrap_future(self.submit(self.predict_now, input_text))

    # the thread ---------------------------------------------------------------------------------
    def _load(self) -> None:
        started = time.perf_counter()
        try:
            backend = self._loader()
            backend.load()
            if self.spec.type == "classification":
                self.trie = backend.label_trie(list(self.spec.labels))
            self.backend = backend
        except Exception as exc:
            self.load_error = exc
            log.error("the student model failed to load: %s: %s", type(exc).__name__, exc)
        finally:
            self.load_ms = (time.perf_counter() - started) * 1000.0
            self._loaded.set()

    def _main(self) -> None:
        self.thread_id = threading.get_ident()
        self._load()
        while True:
            job = self._jobs.get()
            if job is None:
                return
            if not job.future.set_running_or_notify_cancel():
                continue  # cancelled while queued (the client went away)
            try:
                result = job.fn(*job.args, **job.kwargs)
            except BaseException as exc:
                job.future.set_exception(exc)
            else:
                job.future.set_result(result)
