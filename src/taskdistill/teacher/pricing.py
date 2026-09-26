"""Dated pricing snapshot for the teacher: USD per prompt token, per completion token and per request.

``taskdistill pricing refresh`` builds it from OpenRouter's model list (``GET /models``: ``pricing.prompt``,
``pricing.completion`` and the optional ``pricing.request``, all USD per token or per request as decimal
strings) and, for the models it is asked about, the per-provider prices from ``GET /models/{slug}/endpoints``.
Providers are keyed by the endpoint ``tag`` (the value ``provider.order`` accepts, e.g. ``deepinfra/fp8``). A
model's dated ``canonical_slug`` is kept as an alias of its ``id``, since OpenRouter accepts either.

The snapshot also keeps each model's maximum completion tokens per choice (``top_provider.max_completion_tokens``,
else ``context_length``, and per endpoint ``max_completion_tokens`` else ``context_length``): the limit
:func:`~taskdistill.ledger.reservation_cost` reserves against for a request that sets no ``max_tokens``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import httpx

from taskdistill import paths

log = logging.getLogger("taskdistill.pricing")
_warned: set[tuple[str, str]] = set()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class PricingError(RuntimeError):
    """The pricing snapshot could not be built or read."""


@dataclass(frozen=True)
class ModelPrice:
    """USD per prompt token, per completion token and per request."""

    prompt: float
    completion: float
    request: float = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.prompt + completion_tokens * self.completion + self.request

    def to_json(self) -> dict[str, float]:
        return {"prompt": self.prompt, "completion": self.completion, "request": self.request}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ModelPrice:
        return cls(float(data["prompt"]), float(data["completion"]), float(data.get("request") or 0.0))

    @classmethod
    def from_openrouter(cls, pricing: dict[str, Any] | None) -> ModelPrice | None:
        """Parse an OpenRouter ``pricing`` object; None when it is missing or dynamic (negative sentinel)."""
        if not isinstance(pricing, dict):
            return None
        try:
            values = [float(pricing.get(key) or 0) for key in ("prompt", "completion", "request")]
        except (TypeError, ValueError):
            return None
        if "prompt" not in pricing or "completion" not in pricing or any(v < 0 for v in values):
            return None
        return cls(*values)

    @staticmethod
    def max_of(prices: list[ModelPrice]) -> ModelPrice:
        """Field-wise maximum: the most any of these endpoints could charge."""
        return ModelPrice(
            max(p.prompt for p in prices), max(p.completion for p in prices), max(p.request for p in prices)
        )


class ModelEntry(TypedDict):
    default: ModelPrice
    providers: dict[str, ModelPrice]


class LimitEntry(TypedDict):
    default: int | None
    providers: dict[str, int]


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _model_limit(model: Mapping[str, Any]) -> int | None:
    """A model's maximum completion tokens: ``top_provider.max_completion_tokens``, else ``context_length``."""
    top_provider = model.get("top_provider")
    if isinstance(top_provider, Mapping):
        limit = _positive_int(top_provider.get("max_completion_tokens"))
        if limit is not None:
            return limit
    return _positive_int(model.get("context_length"))


def _endpoint_limit(endpoint: Mapping[str, Any]) -> int | None:
    """One endpoint's maximum completion tokens: its own ``max_completion_tokens``, else ``context_length``."""
    return _positive_int(endpoint.get("max_completion_tokens")) or _positive_int(endpoint.get("context_length"))


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _models_list(models_json: Any) -> list[dict[str, Any]]:
    data = models_json.get("data") if isinstance(models_json, dict) else models_json
    if not isinstance(data, list):
        raise PricingError("models JSON must be a list or an object with a 'data' list")
    return [m for m in data if isinstance(m, dict)]


def _endpoints_list(endpoints_json: Any) -> list[dict[str, Any]]:
    data = endpoints_json.get("data", endpoints_json) if isinstance(endpoints_json, dict) else {}
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    return [e for e in endpoints or [] if isinstance(e, dict)]


@dataclass
class PricingSnapshot:
    date: str
    source: str
    models: dict[str, ModelEntry]
    #: ``canonical_slug`` -> ``id`` for models whose dated slug differs from the id.
    aliases: dict[str, str] = field(default_factory=dict)
    #: Maximum completion tokens per choice, by model slug; a model absent here has no known limit. Named
    #: ``completion_limits``, not ``limits``, so a caller's own ``limits`` attribute (a subclass overriding
    #: :meth:`max_completion_tokens` with its own lookup table) is never shadowed by this dataclass field.
    completion_limits: dict[str, LimitEntry] = field(default_factory=dict)

    # construction ------------------------------------------------------------------------------
    @classmethod
    def from_openrouter(
        cls,
        models_json: Any,
        endpoints_json_by_slug: dict[str, Any] | None = None,
        date: str | None = None,
        *,
        source: str = f"{OPENROUTER_BASE_URL}/models",
    ) -> PricingSnapshot:
        """Build a snapshot from ``GET /models`` and optional ``GET /models/{slug}/endpoints`` payloads.

        Models with a dynamic price (OpenRouter's ``"-1"`` sentinel) are skipped: they cannot be budgeted.
        When one endpoint tag appears twice, its field-wise maximum is kept.
        """
        models: dict[str, ModelEntry] = {}
        limits: dict[str, LimitEntry] = {}
        aliases: dict[str, str] = {}
        for model in _models_list(models_json):
            slug = model.get("id")
            price = ModelPrice.from_openrouter(model.get("pricing"))
            if isinstance(slug, str) and price is not None:
                models[slug] = {"default": price, "providers": {}}
                model_limit = _model_limit(model)
                if model_limit is not None:
                    limits[slug] = {"default": model_limit, "providers": {}}
                canonical = model.get("canonical_slug")
                if isinstance(canonical, str) and canonical and canonical != slug:
                    aliases[canonical] = slug
        aliases = {alias: target for alias, target in aliases.items() if alias not in models}
        for requested, endpoints_json in sorted((endpoints_json_by_slug or {}).items()):
            slug = requested if requested in models else aliases.get(requested, requested)
            providers: dict[str, ModelPrice] = {}
            provider_limits: dict[str, int] = {}
            for endpoint in _endpoints_list(endpoints_json):
                name = endpoint.get("tag") or endpoint.get("provider_name")
                price = ModelPrice.from_openrouter(endpoint.get("pricing"))
                if not isinstance(name, str) or price is None:
                    continue
                providers[name] = ModelPrice.max_of([providers[name], price]) if name in providers else price
                limit = _endpoint_limit(endpoint)
                if limit is not None:
                    provider_limits[name] = max(limit, provider_limits.get(name, limit))
            if not providers:
                continue
            if slug in models:
                models[slug]["providers"] = dict(sorted(providers.items()))
            else:
                models[slug] = {
                    "default": ModelPrice.max_of(list(providers.values())),
                    "providers": dict(sorted(providers.items())),
                }
            if provider_limits or slug in limits:  # never create an empty (no default, no providers) entry
                entry = limits.setdefault(slug, {"default": None, "providers": {}})
                entry["providers"] = dict(sorted(provider_limits.items()))
                if entry["default"] is None and provider_limits:
                    entry["default"] = max(provider_limits.values())
        return cls(
            date=date or _today(),
            source=source,
            models=dict(sorted(models.items())),
            aliases=dict(sorted(aliases.items())),
            completion_limits=dict(sorted(limits.items())),
        )

    # persistence -------------------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "source": self.source,
            "models": {
                slug: {
                    "default": entry["default"].to_json(),
                    "providers": {name: p.to_json() for name, p in sorted(entry["providers"].items())},
                }
                for slug, entry in sorted(self.models.items())
            },
            "aliases": dict(sorted(self.aliases.items())),
            "limits": {
                slug: {"default": entry.get("default"), "providers": dict(sorted(entry["providers"].items()))}
                for slug, entry in sorted(self.completion_limits.items())
                if entry.get("default") is not None or entry.get("providers")
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PricingSnapshot:
        models: dict[str, ModelEntry] = {}
        for slug, entry in (data.get("models") or {}).items():
            models[slug] = {
                "default": ModelPrice.from_json(entry["default"]),
                "providers": {name: ModelPrice.from_json(p) for name, p in (entry.get("providers") or {}).items()},
            }
        aliases = {str(k): str(v) for k, v in (data.get("aliases") or {}).items()}
        limits: dict[str, LimitEntry] = {}
        for slug, entry in (data.get("limits") or {}).items():
            limits[slug] = {
                "default": _positive_int(entry.get("default")),
                "providers": {
                    name: value for name, value in (entry.get("providers") or {}).items() if _positive_int(value)
                },
            }
        return cls(
            date=str(data["date"]),
            source=str(data.get("source", "")),
            models=models,
            aliases=aliases,
            completion_limits=limits,
        )

    def save(self, path: Path | str | None = None) -> Path:
        out = Path(path) if path is not None else paths.pricing_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path | str | None = None) -> PricingSnapshot:
        src = Path(path) if path is not None else paths.pricing_path()
        if not src.is_file():
            raise PricingError(f"no pricing snapshot at {src.name}; run `taskdistill pricing refresh` first")
        try:
            return cls.from_json(json.loads(src.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError) as exc:
            raise PricingError(f"pricing snapshot {src.name} is not valid: {exc}") from exc

    # lookup ------------------------------------------------------------------------------------
    def price_for(self, model: str, provider: str | None = None) -> ModelPrice:
        """The price to budget ``model`` at.

        The pinned provider's price when the snapshot knows it (an exact endpoint tag such as
        ``deepinfra/fp8``, a base slug such as ``deepinfra`` or a display name such as ``DeepInfra``; a base slug
        matching several endpoints gives their maximum). Otherwise the field-wise maximum over the model's known
        providers, since routing may pick any of them; otherwise the model's list price. ``model`` may be the
        models-list ``id`` or its dated ``canonical_slug``.
        """
        entry = self.models.get(model)
        if entry is None and model in self.aliases:
            entry = self.models.get(self.aliases[model])
        if entry is None:
            raise KeyError(
                f"no price for teacher model '{model}' in the pricing snapshot of {self.date} ({self.source}); "
                "run `taskdistill pricing refresh` or check teacher.model"
            )
        providers = entry["providers"]
        if provider and providers:
            wanted = provider.strip().lower()
            if provider in providers:
                return providers[provider]
            exact = [p for name, p in providers.items() if name.lower() == wanted]
            if exact:
                return ModelPrice.max_of(exact)
            if "/" not in wanted:
                base = [p for name, p in providers.items() if name.lower().split("/", 1)[0] == wanted]
                if base:
                    return ModelPrice.max_of(base)
        if providers:
            return ModelPrice.max_of(list(providers.values()))
        if provider and (model, provider) not in _warned:
            _warned.add((model, provider))
            log.warning(
                "the pricing snapshot has no per-provider prices for %s, so the pinned provider %s is budgeted at "
                "the list price; refresh the snapshot with this model's endpoints",
                model,
                provider,
            )
        return entry["default"]

    def max_completion_tokens(self, model: str, provider: str | None = None) -> int | None:
        """The most completion tokens per choice ``model`` can generate, resolving aliases like :meth:`price_for`.

        The pinned endpoint's limit when known, else the largest known limit for the model, else None (a model
        or endpoint the snapshot never saw a limit for, or one that ``pricing refresh`` was not asked about).
        Unlike :meth:`price_for`, an unknown model is not an error: the caller (a reservation) treats None as
        "no worst case can be bounded" rather than "no price is known".
        """
        entry = self.completion_limits.get(model)
        if entry is None and model in self.aliases:
            entry = self.completion_limits.get(self.aliases[model])
        if entry is None:
            return None
        providers = entry["providers"]
        if provider and providers:
            wanted = provider.strip().lower()
            if provider in providers:
                return providers[provider]
            exact = [v for name, v in providers.items() if name.lower() == wanted]
            if exact:
                return max(exact)
            if "/" not in wanted:
                base = [v for name, v in providers.items() if name.lower().split("/", 1)[0] == wanted]
                if base:
                    return max(base)
        candidates = list(providers.values())
        default = entry.get("default")
        if default is not None:
            candidates.append(default)
        return max(candidates) if candidates else None


def _get_json(http: httpx.Client, url: str, headers: dict[str, str]) -> Any:
    try:
        resp = http.get(url, headers=headers)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise PricingError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
    if resp.status_code != 200:
        raise PricingError(f"GET {url} returned HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError as exc:
        raise PricingError(f"GET {url} did not return JSON") from exc


def refresh(
    base_url: str = OPENROUTER_BASE_URL,
    api_key: str | None = None,
    models: list[str] | None = None,
    client: httpx.Client | None = None,
    *,
    date: str | None = None,
) -> PricingSnapshot:
    """Fetch ``GET {base_url}/models`` and, for each slug in ``models``, ``GET {base_url}/models/{slug}/endpoints``.

    Any failure (transport error, HTTP status other than 200, a body that is not JSON) raises :class:`PricingError`.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    own = client is None
    http = client if client is not None else httpx.Client(timeout=30.0)
    try:
        models_json = _get_json(http, f"{base}/models", headers)
        endpoints: dict[str, Any] = {}
        for slug in models or []:
            endpoints[slug] = _get_json(http, f"{base}/models/{slug}/endpoints", headers)
    finally:
        if own:
            http.close()
    snapshot = PricingSnapshot.from_openrouter(models_json, endpoints, date, source=f"{base}/models")
    missing = [slug for slug in models or [] if slug not in snapshot.models and slug not in snapshot.aliases]
    if missing:
        raise PricingError(f"no usable price for {', '.join(missing)} at {base}/models")
    return snapshot
