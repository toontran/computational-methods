import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all generated matrix benchmarks and write them into benchmarks.txt."
    )
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--preset", choices=("small", "fast", "cex-replicate"), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="benchmarks.txt")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of appending a new benchmark section.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed through to cex_restricted_space_probe.py. Repeat as needed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    probe = here / "cex_restricted_space_probe.py"
    output = Path(args.output)
    if not output.is_absolute():
        output = here / output

    cmd = [
        sys.executable,
        str(probe),
        "--benchmark-all",
        "--n",
        str(args.n),
        "--win",
        str(args.win),
        "--rank",
        str(args.rank),
        "--preset",
        args.preset,
        "--seed",
        str(args.seed),
        "--benchmark-output",
        str(output),
    ]
    if not args.overwrite:
        cmd.append("--benchmark-append")
    cmd.extend(args.extra_arg)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=here, check=True)
    print(f"Wrote benchmark results to {output}")


if __name__ == "__main__":
    main()
