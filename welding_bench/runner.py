import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from .adapters import Adapter
from .api import execute_frozen
from .common import file_sha, parse_answer, read, save, source_hashes, utc
from .sampling import load_cases


def parser():
    p = argparse.ArgumentParser(description='Run one frozen semantic protocol with a selected model adapter.')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['codeasworlds', 'cosmos3', 'robobrain', 'responses'], required=True)
    p.add_argument('--weights', type=Path, required=True, help='Local model directory; API uses only its reference processor')
    p.add_argument('--frames', type=int, nargs='+', default=[16, 64])
    p.add_argument('--max-tokens', type=int, default=3072, help='Explicitly pass 32768 for the larger API budget')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='API only: reuse frozen requests, skip every completed answer')
    p.add_argument('--retry-transport-errors', action='store_true', help='API only: explicitly retry errors/uncertain attempts')
    p.add_argument('--model', default='gpt-6-astra', help='Responses API model identifier')
    p.add_argument('--base-url', default='http://127.0.0.1:8317/v1')
    p.add_argument('--reasoning-effort', choices=['none','minimal','low','medium','high','xhigh','max'], default='max')
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--chat-template', type=Path, help='Official Code-as-World qwen3_5_no_think.jinja')
    p.add_argument('--caw-size', choices=['4b','9b'], default='9b')
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--context', type=int, default=16384)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.88)
    p.add_argument('--gpu-memory-gib', type=int, default=20)
    return p


def settings(args):
    keys = ['backend','max_tokens','frames','model','base_url','reasoning_effort','caw_size','tp',
            'context','gpu_memory_utilization','gpu_memory_gib']
    value = {k: getattr(args, k) for k in keys}
    value['weights'] = str(args.weights.resolve())
    value['chat_template'] = str(args.chat_template.resolve()) if args.chat_template else None
    value['seed'] = None if args.backend == 'responses' else 1
    value['temperature'] = None if args.backend == 'responses' else 0
    return value


def run(args):
    if args.max_tokens <= 0 or args.tp <= 0 or args.context <= 0 or args.timeout <= 0:
        raise ValueError('Budgets, timeout and device counts must be positive')
    if len(set(args.frames)) != len(args.frames):
        raise ValueError('Frame counts must be unique')
    if args.resume and args.backend != 'responses':
        raise ValueError('Resume is supported only for frozen API requests; use a new local-model output directory')
    if args.retry_transport_errors and not args.resume:
        raise ValueError('--retry-transport-errors requires --resume')
    url = urlsplit(args.base_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('API base URL must not contain credentials, query or fragment')
    key = os.environ.get('WELDING_API_KEY')
    if args.backend == 'responses' and not args.prepare_only and not key:
        raise ValueError('Set WELDING_API_KEY in the environment')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    cases = load_cases(args.inputs, args.frames)
    if set(c['nframes'] for c, _ in cases) != set(args.frames):
        raise ValueError('Some requested frame counts are missing from the input manifest')
    out = args.output.resolve()
    protocol = {'settings': settings(args), 'source_hashes': source_hashes(),
                'input_manifest_sha256': file_sha(args.inputs / 'manifest.json'), 'protocol': 'schema_fields_only'}
    prepared = []
    adapter = None
    if args.resume:
        old = read(out / 'protocol.json')
        for k, v in protocol.items():
            if old[k] != v:
                raise ValueError(f'Frozen protocol changed: {k}')
        protocol = old
        run_data = read(out / 'run.json')
        for case, _ in cases:
            directory = out / case['case_id']
            for name, digest in protocol['files'][case['case_id']].items():
                if file_sha(directory / name) != digest:
                    raise ValueError(f'Frozen input changed: {name}')
            prepared.append((case, (directory / 'request.json').read_bytes()))
    else:
        out.mkdir(parents=True, exist_ok=False)
        adapter = Adapter(args)
        protocol['files'] = {}
        for case, frames in cases:
            directory = out / case['case_id']
            directory.mkdir()
            inputs, audit = adapter.prepare(case, frames, directory)
            protocol['files'][case['case_id']] = {p.name: file_sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
            prepared.append((case, inputs))
            print('INPUT_READY', case['case_id'], flush=True)
        protocol['created_at_utc'] = utc()
        protocol['runtime'] = {'python': sys.version}
        for name in ['numpy','opencv-python-headless','pillow','torch','torchvision','transformers','vllm','qwen-vl-utils']:
            try:
                protocol['runtime'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        protocol['model_config_hashes'] = {p.name: file_sha(p) for p in args.weights.glob('*.json')}
        protocol['weight_identity_limit'] = 'Local path and config hashes; full weight bytes are NOT verified by this runner.'
        save(out / 'protocol.json', protocol)
        run_data = {'status': 'prepared', 'protocol_sha256': file_sha(out / 'protocol.json'), 'results': [], 'transport_failures': []}
        save(out / 'run.json', run_data)
    if args.prepare_only:
        return
    if args.backend == 'responses':
        execute_frozen(out, args.timeout, args.retry_transport_errors)
        return
    run_data['status'] = 'running'
    save(out / 'run.json', run_data)
    try:
        if args.backend != 'responses':
            adapter.load_model(out)
        for case, inputs in prepared:
            cid = case['case_id']
            directory = out / cid
            previous = next((r for r in run_data['results'] if r['case_id'] == cid), None)
            if previous and not previous.get('transport_error'):
                continue  # Includes invalid JSON and incomplete responses: never cherry-pick.
            attempts = list(directory.glob('attempt_*.json'))
            if (previous or attempts) and not args.retry_transport_errors:
                raise RuntimeError('Inspect retained attempts, then explicitly use --retry-transport-errors')
            save(directory / 'attempt_001.json', {'status': 'started', 'started_at_utc': utc()})
            tick = time.perf_counter()
            row = adapter.generate(inputs)
            row['elapsed_s'] = time.perf_counter() - tick
            save(directory / 'raw_response.json', row)
            save(directory / 'attempt_001.json', {'status': 'response_saved', 'finished_at_utc': utc()})
            row['case_id'] = cid
            if row.get('transport_error'):
                row.update(parsed_json=None, parse_error='Transport failure; partial text is not a semantic answer')
            else:
                row['final_text'], row['parsed_json'], row['parse_error'] = parse_answer(row['raw_text'])
            if previous:
                run_data['transport_failures'].append(previous)
                run_data['results'].remove(previous)
            run_data['results'].append(row)
            save(out / 'run.json', run_data)
            print('RESULT', cid, row['finish_reason'], 'parsed', row['parsed_json'] is not None, flush=True)
            if row.get('transport_error'):
                raise RuntimeError('Transport failure saved; no automatic retry')
        run_data['status'] = 'complete'
        run_data['completed_at_utc'] = utc()
    except Exception as exc:
        run_data['status'] = 'failed'
        run_data['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save(out / 'run.json', run_data)


def main():
    run(parser().parse_args())
