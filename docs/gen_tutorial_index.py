#!/usr/bin/env python3
"""Generate a Doxygen parent page that lists all tutorial .cpp files as subpages.

Usage:
    python3 gen_tutorial_index.py <tutorial_dir> <output.dox> [options]

Page names match tutorial_filter.py: "tut_" + file basename (non-word
characters replaced by "_"). Files are listed in natural filename order,
so prefix them with numbers (1_basics.cpp, 2_advanced.cpp, 10_extras.cpp)
to control the order in the navigation.

The output file is only rewritten when its content changes, so build
systems don't trigger unnecessary Doxygen runs.
"""
import argparse
import re
import sys
from pathlib import Path


def page_name(path: Path) -> str:
    # Must match the naming used in tutorial_filter.py
    return "tut_" + re.sub(r"\W", "_", path.stem)


def natural_key(path: Path):
    # Sort "2_x" before "10_x"
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(path))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tutorial_dir", type=Path, help="directory with tutorial .cpp files")
    ap.add_argument("output", type=Path, help="path of the .dox file to generate")
    ap.add_argument("--pattern", default="*.cpp", help='glob pattern (default: "*.cpp")')
    ap.add_argument("--recursive", action="store_true", help="search subdirectories too")
    ap.add_argument("--page", default="tutorials", help='parent page name (default: "tutorials")')
    ap.add_argument("--title", default="Tutorials", help='parent page title (default: "Tutorials")')
    ap.add_argument("--intro", default="", help="optional intro text above the list")
    args = ap.parse_args()

    if not args.tutorial_dir.is_dir():
        print(f"error: {args.tutorial_dir} is not a directory", file=sys.stderr)
        return 1

    finder = args.tutorial_dir.rglob if args.recursive else args.tutorial_dir.glob
    files = sorted((p for p in finder(args.pattern) if p.is_file()), key=natural_key)

    if not files:
        print(f"warning: no files matching {args.pattern!r} in {args.tutorial_dir}",
              file=sys.stderr)

    # Duplicate page names (e.g. same basename in different subfolders) would
    # make Doxygen silently merge or drop pages, so fail loudly instead.
    seen = {}
    for f in files:
        name = page_name(f)
        if name in seen:
            print(f"error: {f} and {seen[name]} both map to page '{name}'",
                  file=sys.stderr)
            return 1
        seen[name] = f

    lines = ["/**", f"\\page {args.page} {args.title}", ""]
    if args.intro:
        lines += [args.intro, ""]
    lines += [f"- \\subpage {page_name(f)}" for f in files]
    lines += ["*/", ""]
    content = "\n".join(lines)

    if args.output.exists() and args.output.read_text(encoding="utf-8") == content:
        print(f"{args.output} is up to date ({len(files)} tutorials)")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"wrote {args.output} ({len(files)} tutorials)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
