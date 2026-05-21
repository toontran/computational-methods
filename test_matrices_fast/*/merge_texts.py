#!/usr/bin/env python3
from pathlib import Path
import argparse

def merge_text_files(input_dir: str, output_file: str, recursive: bool = False) -> None:
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    pattern = "**/*.txt" if recursive else "*.txt"
    files = sorted(
        f for f in root.glob(pattern)
        if f.is_file() and f.resolve() != Path(output_file).resolve()
    )

    with open(output_file, "w", encoding="utf-8") as out:
        for i, file_path in enumerate(files):
            rel_name = file_path.relative_to(root)
            out.write(f"===== {rel_name} =====\n")
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            if i != len(files) - 1:
                out.write("\n")

    print(f"Merged {len(files)} text file(s) into {output_file}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all .txt files into one file, writing the filename before each file's content."
    )
    parser.add_argument("input_dir", help="Folder containing .txt files")
    parser.add_argument("output_file", help="Path to output merged file")
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Include .txt files in subfolders recursively"
    )
    args = parser.parse_args()

    merge_text_files(args.input_dir, args.output_file, args.recursive)

if __name__ == "__main__":
    main()
    