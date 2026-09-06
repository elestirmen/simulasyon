"""Synchronize the shared template with another simulator checkout.

python -m mission_ui.sync "C:/path/to/other/repo" --check
python -m mission_ui.sync "C:/path/to/other/repo"
"""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--check", action="store_true", help="Compare without writing")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repository = args.repository.resolve()
    if not any((repository / name).is_file() for name in ("pyproject.toml", "requirements.txt")):
        parser.error("Target must be a simulator repository containing pyproject.toml or requirements.txt")
    destination = repository / "mission_ui"
    changed = []
    for path in sorted(source.glob("*.py")):
        target = destination / path.name
        if not target.exists() or path.read_bytes() != target.read_bytes():
            changed.append(path.name)
            if not args.check:
                destination.mkdir(exist_ok=True)
                target.write_bytes(path.read_bytes())
    if args.check and changed:
        print("Template differs: " + ", ".join(changed))
        return 1
    print("Template identical" if not changed else "Updated: " + ", ".join(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
