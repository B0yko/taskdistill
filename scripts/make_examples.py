"""Write the 20-document invoice sample under examples/invoices/ (or check that it is up to date).

    uv run python scripts/make_examples.py [--check]

Each document goes to ``<id>.txt`` as the model sees it; ``sample.jsonl`` holds the same documents as ``inputs``
import rows (text, gold fields, template, kind and traits), so the sample can also be fed to ``taskdistill import``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskdistill.demos.invoices import doc_to_record, sample_docs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "invoices"


def expected_files() -> dict[str, str]:
    docs = sample_docs(20)
    files = {f"{doc.id}.txt": doc.text.rstrip("\n") + "\n" for doc in docs}
    files["sample.jsonl"] = "".join(json.dumps(doc_to_record(doc), ensure_ascii=False) + "\n" for doc in docs)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed sample differs")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    files = expected_files()
    if args.check:
        present = {p.name for p in args.out.glob("*") if p.suffix in (".txt", ".jsonl")}
        stale = sorted(
            name
            for name, text in files.items()
            if not (args.out / name).is_file() or (args.out / name).read_text(encoding="utf-8") != text
        )
        extra = sorted(present - set(files))
        for name in stale:
            print(f"out of date: {name}")
        for name in extra:
            print(f"not generated: {name}")
        return 1 if stale or extra else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} files to {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
