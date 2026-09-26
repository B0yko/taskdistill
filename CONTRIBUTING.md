# Contributing

Thanks for looking at taskdistill. Bug reports, reproductions on other Apple Silicon machines and small, focused
pull requests are all welcome.

## Development setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it for you).

```bash
git clone https://github.com/B0yko/taskdistill
cd taskdistill
uv sync --extra torch --extra hub      # dev tools come from the "dev" dependency group
uv run taskdistill --help
```

On Linux, `uv sync --extra torch` installs the CPU-only PyTorch wheel from the PyTorch index (configured in
`pyproject.toml`), so the torch path and its tests run without CUDA.

## Checks

Run these before opening a pull request; CI runs the same commands.

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q                        # everything except the Metal tests
uv run python scripts/sync_readme.py --check
```

Tests that need Apple Silicon with Metal are marked `mlx` and are excluded by default, because GitHub-hosted
macOS runners are not a reliable Metal target. Run them locally on an M-series Mac:

```bash
uv run pytest -m mlx -p no:cacheprovider --no-header -q
```

They download `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (about 0.3 GB) on first use.

Tests never call a paid API. HTTP is faked with respx, the teacher is replayed from the packaged recordings,
and the torch tests build a tiny random-weight model inside the test. Tests marked `network` download public
data (the pinned Banking77 CSVs from GitHub, tokenizer files from the Hugging Face Hub).

## Numbers in the README

Every number in the README comes from a committed file under `reports/`. Do not edit README tables by hand:

- `scripts/reproduce.sh` regenerates every replay-based report (it trains the students, so it takes hours on
  a laptop);
- `scripts/sync_readme.py` rewrites the README tables from `reports/*.json`, and `--check` fails when they
  differ.

Live measurements (teacher latency, the live bench, spend) are dated and are not expected to reproduce exactly.

## Style

- Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- Type hints everywhere; `mypy` is strict for `evaluate/`, `ledger.py` and `confidence.py`.
- Every selection step (model, checkpoint, run, threshold, calibration, baseline tuning) takes a
  `ValidationSplit`. Code that reads the test split may only report.
- Synthetic contact data in code, tests and fixtures uses reserved values only: `example.com`/`.test`
  domains, `192.0.2.0/24` addresses, `555-01xx` phone numbers, published test card numbers and example IBANs.

## Releasing

Releases are tagged on GitHub. The PyPI workflow (`.github/workflows/publish.yml`) only runs when started by
hand from the Actions tab and uses PyPI trusted publishing:

1. On pypi.org, open *Account settings → Publishing* and add a pending GitHub publisher: project `taskdistill`,
   owner `B0yko`, repository `taskdistill`, workflow `publish.yml`, environment `pypi`.
2. In the GitHub repository, create an environment named `pypi` (*Settings → Environments*).
3. Run the `publish` workflow from the Actions tab.
