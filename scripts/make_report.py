#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Results\n"]
    lines.append(df.to_markdown(index=False))
    out.write_text("\n\n".join(lines))
    print("Wrote", out)


if __name__ == "__main__":
    main()

