"""Responses-compatible SSE transport. Preserve every attempt; never retry silently."""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from .common import file_sha, parse_answer, read, save, sha, source_hashes, utc


def send(endpoint, key, body, directory, timeout=1800):
    attempt = len(list(directory.glob('attempt_*.json'))) + 1
    stem = f'attempt_{attempt:03d}'
    attempt_path = directory / f'{stem}.json'
    events = directory / f'{stem}.sse'
    record = {'started_at_utc': utc(), 'request_sha256': sha(body), 'status': 'sent', 'events_file': events.name}
    save(attempt_path, record)
    request = urllib.request.Request(endpoint, data=body, headers={
        'Authorization': 'Bearer '+key, 'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    partial, response, status = [], None, None
    started = time.perf_counter()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as stream, events.open('wb') as log:
            status = stream.status
            pending = []
            for line in stream:
                log.write(line)
                log.flush()
                if line.startswith(b'data:'):
                    pending.append(line[5:].strip())
                elif not line.strip() and pending:
                    data = b'\n'.join(pending)
                    pending = []
                    if data == b'[DONE]':
                        break
                    event = json.loads(data)
                    if event.get('type') == 'response.output_text.delta':
                        partial.append(event.get('delta', ''))
                    elif event.get('type') in ('response.completed', 'response.incomplete'):
                        response = event['response']
                        break
                    elif event.get('type') in ('error', 'response.failed'):
                        raise RuntimeError(json.dumps(event, ensure_ascii=False))
            if response is None:
                raise RuntimeError('Stream ended before completion; partial text retained')
        save(directory / f'{stem}.response.json', response)
        text = ''.join(c['text'] for item in response.get('output', []) if item.get('type') == 'message'
                       for c in item.get('content', []) if c.get('type') == 'output_text')
        result = {'raw_text': text, 'finish_reason': response.get('status'), 'usage': response.get('usage'),
                  'response_id': response.get('id'), 'returned_model': response.get('model'),
                  'returned_reasoning': response.get('reasoning'), 'returned_max_output_tokens': response.get('max_output_tokens'),
                  'response_file': f'{stem}.response.json',
                  'observed_output_types': [item.get('type') for item in response.get('output', [])]}
        record['status'] = 'response_saved'
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            status = exc.code
            events.write_bytes(exc.read())
        result = {'raw_text': ''.join(partial), 'finish_reason': 'transport_error', 'transport_error': str(exc)}
        record['status'] = 'transport_error'
    result.update(http_status=status, elapsed_s=time.perf_counter()-started, attempt_file=attempt_path.name)
    record.update(finished_at_utc=utc(), **result)
    save(attempt_path, record)
    return result


def execute_frozen(output, timeout=1800, retry=False):
    """Can execute a copied prepared run on an API host without Torch/weights."""
    output = Path(output)
    protocol, run = read(output/'protocol.json'), read(output/'run.json')
    if protocol['settings']['backend'] != 'responses':
        raise ValueError('Expected a prepared Responses API run')
    if run['protocol_sha256'] != file_sha(output/'protocol.json') or protocol['source_hashes'] != source_hashes():
        raise ValueError('Frozen protocol or framework source changed')
    key = os.environ.get('WELDING_API_KEY')
    if not key:
        raise ValueError('Set WELDING_API_KEY')
    for cid, files in protocol['files'].items():
        for name, digest in files.items():
            if file_sha(output/cid/name) != digest:
                raise ValueError(f'Frozen input changed: {cid}/{name}')
    run['status'] = 'running'
    run.pop('error', None)
    save(output/'run.json', run)
    try:
        for cid in protocol['files']:
            directory = output/cid
            previous = next((row for row in run['results'] if row['case_id'] == cid), None)
            if previous and not previous.get('transport_error'):
                continue
            if (previous or list(directory.glob('attempt_*.json'))) and not retry:
                raise RuntimeError('Inspect retained attempts; use --retry-transport-errors explicitly')
            row = send(protocol['settings']['base_url'].rstrip('/')+'/responses', key,
                       (directory/'request.json').read_bytes(), directory, timeout)
            row['case_id'] = cid
            if row.get('transport_error'):
                row.update(parsed_json=None, parse_error='Transport failure; partial text is not a semantic answer')
            else:
                row['final_text'], row['parsed_json'], row['parse_error'] = parse_answer(row['raw_text'])
                requested = protocol['settings']['model']
                if row.get('returned_model') != requested:
                    row['model_identity_note'] = f'Requested {requested}; returned {row.get("returned_model")}'
            if previous:
                run['transport_failures'].append(previous)
                run['results'].remove(previous)
            run['results'].append(row)
            save(output/'run.json', run)
            print('RESULT', cid, row['finish_reason'], 'parsed', row['parsed_json'] is not None, flush=True)
            if row.get('transport_error'):
                raise RuntimeError('Transport failure retained; no automatic retry')
        run.update(status='complete', completed_at_utc=utc())
    except Exception as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(output/'run.json', run)
