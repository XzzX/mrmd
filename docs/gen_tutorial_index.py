#!/usr/bin/env python3
"""Generate a Doxygen parent page that lists all tutorial .cpp files as subpages.

Usage:
    python3 gen_tutorial_index.py <tutorial_dir> <output.dox> [options]

Page names match tutorial_filter.py: "tut_" + file basename (non-word
characters replaced by "_"). Files are listed in natural filename order,
so prefix them with numbers (1_basics.cpp, 2_advanced.cpp, 10_extras.cpp)
to control the order in the navigation.

If a tutorial starts with a block of prose lines (marker "//>", see
tutorial_filter.py), the first line is the title and the following prose
lines are the description, which is shown below the link in the overview.

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


def read_description(path: Path, marker: str) -> list:
    """Return the prose lines directly after the title line at the top of the file.

    The block ends at the first line that is not a prose line. Empty marker
    lines ("//>") are kept as blank lines, i.e. paragraph breaks.
    """
    prose = re.compile(r"^\s*" + re.escape(marker) + r">* ?(.*)$")  # tolerate "//>>"
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines or not prose.match(lines[0]):
        return []  # no title line, so no description block either
    desc = []
    for line in lines[1:]:
        m = prose.match(line)
        if not m:
            break
        desc.append(m.group(1).rstrip())
    while desc and not desc[0]:
        desc.pop(0)
    while desc and not desc[-1]:
        desc.pop()
    return desc


MAX_TITLE_LEN = 60  # a title is a label, not a sentence


def read_title(path: Path, marker: str) -> str:
    """Return the tutorial's title line, or "" if it does not have a usable one.

    A title is the first non-blank line of the file, it has to be a prose line,
    and it has to look like a label rather than the opening of a sentence.
    """
    prose = re.compile(r"^\s*" + re.escape(marker) + r">* ?(.*)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = prose.match(line)
        if not m:
            return ""
        title = m.group(1).strip()
        return title if 0 < len(title) <= MAX_TITLE_LEN else ""
    return ""


def natural_key(path: Path):
    # Sort "2_x" before "10_x". Each token becomes a (kind, number, text) tuple so
    # that a digit token is never compared against a text token, which would raise
    # TypeError inside sorted().
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower())
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
    ap.add_argument("--intro-file", type=Path, default=None,
                    help="read the intro text from this file instead of the command line, "
                         "which avoids quoting a multi-line string through the build system")
    ap.add_argument("--marker", default="//>", help='prose marker used in the tutorials (default: "//>")')
    ap.add_argument("--strict", action="store_true",
                    help="fail if a tutorial has no title line, instead of publishing a page "
                         "titled with a sentence fragment")
    args = ap.parse_args()

    if args.intro_file:
        if not args.intro_file.is_file():
            print(f"error: {args.intro_file} is not a file", file=sys.stderr)
            return 1
        args.intro = args.intro_file.read_text(encoding="utf-8").strip()

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

    if args.strict:
        bad = [f for f in files if not read_title(f, args.marker)]
        if bad:
            for f in bad:
                print(f"error: {f} has no '{args.marker} <Title>' line as its first "
                      f"non-blank line", file=sys.stderr)
            print("error: every tutorial must start with a short title line, otherwise its "
                  "first sentence is used as the page title", file=sys.stderr)
            return 1

    lines = [f"\\page {args.page} {args.title}", ""]
    if args.intro:
        # Split: every output line gets its own "///" prefix below, so an intro
        # passed as one string with embedded newlines would leave all but its
        # first line outside the comment block.
        lines += [l.rstrip() for l in args.intro.splitlines()] + [""]
    for f in files:
        lines.append(f"- \\subpage {page_name(f)}")
        desc = read_description(f, args.marker)
        if desc:
            # Indented continuation lines keep the description inside the list item
            lines.append("")
            lines += ["  " + d if d else "" for d in desc]
            lines.append("")
    # Emit as /// line comments, so description text can't end a /** */ block early
    content = "\n".join("/// " + l if l else "///" for l in lines) + "\n"

    if args.output.exists() and args.output.read_text(encoding="utf-8") == content:
        print(f"{args.output} is up to date ({len(files)} tutorials)")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"wrote {args.output} ({len(files)} tutorials)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
