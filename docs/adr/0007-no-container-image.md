# ADR 0007: No container image

- Status: accepted
- Date: 2026-09-26

## Context

The portfolio's common definition of done asks services to ship a Docker image on GHCR. taskdistill has a
server (`taskdistill serve`), but its primary runtime is MLX on Apple Silicon, which needs Metal. Docker on
macOS runs containers in a Linux VM that cannot reach the Metal GPU, so an image would only be able to run the
CPU torch path, slowly, and would misrepresent how the tool is meant to be used.

## Decision

taskdistill ships as a Python package (installed with `uvx` or `pip`) and does not publish a container image.
The CUDA path (`--backend torch`) is documented and CPU-tested, but no CUDA image is built.

## Consequences

- The "service" clause of the definition of done is answered by this ADR rather than by an image.
- Teams that want to run the cascade server on Linux can build their own image from the package with the
  `torch` extra; that path is untested on GPUs, as the README says.
