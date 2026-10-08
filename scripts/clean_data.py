"""Show how much disk each dataset in data/ uses, and delete the ones you name.

Datasets are never in git, so deleting them is safe once the models are trained:
scripts/download_data.py downloads them again with one command.

Usage:
  python scripts/clean_data.py                 # list sizes, delete nothing
  python scripts/clean_data.py mrl cew         # delete data/mrl and data/cew (asks first)
  python scripts/clean_data.py --all --yes     # delete everything in data/ without asking
"""
import argparse
import shutil
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="folders in data/ to delete")
    parser.add_argument("--all", action="store_true", help="delete every folder in data/")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    entries = sorted(DATA.iterdir()) if DATA.exists() else []
    sizes = {p.name: size(p) for p in entries}
    for name, s in sizes.items():
        print(f"  {name:<14} {s / 1e6:>10,.1f} MB")
    print(f"  {'total':<14} {sum(sizes.values()) / 1e6:>10,.1f} MB")

    targets = list(sizes) if args.all else args.names
    unknown = [n for n in targets if n not in sizes]
    if unknown:
        raise SystemExit(f"not in data/: {', '.join(unknown)}")
    if not targets:
        return
    freed = sum(sizes[n] for n in targets) / 1e6
    if not args.yes and input(f"Delete {', '.join(targets)} ({freed:,.0f} MB)? [y/N] ").strip().lower() != "y":
        print("Nothing deleted.")
        return
    for name in targets:
        path = DATA / name
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    print(f"Deleted {', '.join(targets)}: {freed:,.0f} MB freed.")


if __name__ == "__main__":
    main()
