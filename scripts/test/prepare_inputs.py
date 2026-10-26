"""Freeze uniformly sampled PNG inputs and shared prompts without loading a model."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from welding_bench.sampling import prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', type=Path, nargs='+', required=True)
    p.add_argument('--frames', type=int, nargs='+', default=[16, 64])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = prepare(args.videos, args.frames, args.output)
    print(f"Prepared {len(manifest['cases'])} cases in {args.output}")
