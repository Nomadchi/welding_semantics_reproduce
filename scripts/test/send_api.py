"""Send frozen API requests, including on a different host; no Torch dependency."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from welding_bench.api import execute_frozen

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--retry-transport-errors', action='store_true')
    a = p.parse_args()
    execute_frozen(a.output, a.timeout, a.retry_transport_errors)
