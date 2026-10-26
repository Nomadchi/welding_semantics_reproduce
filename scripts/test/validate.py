import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from welding_bench.validation import validate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Validate provenance and report parse/schema failures without repairing answers.')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(validate(a.inputs, a.output), ensure_ascii=False, indent=2))
