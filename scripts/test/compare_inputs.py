import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from welding_bench.common import read

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Compare common inputs; spatial and generation settings may intentionally differ.')
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    a = p.parse_args()
    left, right = read(a.left/'protocol.json'), read(a.right/'protocol.json')
    if set(left['files']) != set(right['files']):
        raise ValueError('Case sets differ; compare runs with the same frame-count selection')
    checks = []
    for cid in left['files']:
        x, y = read(a.left/cid/'input_audit.json'), read(a.right/cid/'input_audit.json')
        keys = ['system','prompt','source_sha256','actual_indices','sampled_times_s','raw_rgb_sha256']
        differences = [k for k in keys if x[k] != y[k]]
        checks.append({'case_id': cid, 'common_input_differences': differences,
                       'processed_pixels_equal': x['processed_pixels_sha256'] == y['processed_pixels_sha256']})
    print(json.dumps({'cases': checks, 'left_settings': left['settings'], 'right_settings': right['settings'],
                     'note': 'Processed hash compares reference tensors. API PNG quantization and server processing remain different.'}, indent=2))
    if any(row['common_input_differences'] for row in checks):
        raise SystemExit(1)
