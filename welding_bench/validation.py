from pathlib import Path
from .common import file_sha, parse_answer, read
from .sampling import load_cases
from .semantic_contract import TEMPLATE


def validate(inputs, output):
    inputs, output = Path(inputs), Path(output)
    protocol, run = read(output/'protocol.json'), read(output/'run.json')
    if file_sha(output/'protocol.json') != run['protocol_sha256']:
        raise ValueError('Protocol changed after preparation')
    if file_sha(inputs/'manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    cases = {c['case_id']: c for c, _ in load_cases(inputs, protocol['settings']['frames'])}
    if set(cases) != set(protocol['files']):
        raise ValueError('Prepared cases differ from input manifest')
    for cid, files in protocol['files'].items():
        for name, digest in files.items():
            if file_sha(output/cid/name) != digest:
                raise ValueError(f'Frozen file changed: {cid}/{name}')
        audit, case = read(output/cid/'input_audit.json'), cases[cid]
        for k in ['system','prompt','actual_indices','sampled_times_s','nframes']:
            if audit[k] != case[k]:
                raise ValueError(f'Adapter changed common input: {cid}/{k}')
        if audit['raw_rgb_sha256'] != [i['rgb_sha256'] for i in case['images']]:
            raise ValueError('Adapter raw pixels differ from frozen input')
    rows, seen = [], set()
    for row in run['results']:
        cid = row['case_id']
        if cid not in cases or cid in seen:
            raise ValueError('Unexpected or repeated result')
        seen.add(cid)
        if row.get('transport_error'):
            rows.append({'case_id': cid, 'transport_error': True})
            continue
        if protocol['settings']['backend'] == 'responses':
            response = read(output/cid/row['response_file'])
            text = ''.join(c['text'] for item in response.get('output',[]) if item.get('type') == 'message'
                           for c in item.get('content',[]) if c.get('type') == 'output_text')
            if response.get('previous_response_id') is not None:
                raise ValueError('Response unexpectedly links to another conversation')
        else:
            text = read(output/cid/'raw_response.json')['raw_text']
        final, parsed, error = parse_answer(text)
        if text != row['raw_text'] or parsed != row['parsed_json'] or error != row['parse_error']:
            raise ValueError('Parsed result differs from saved raw response')
        missing, nulls, types = [], [], []
        if parsed is not None:
            for group, fields in TEMPLATE.items():
                if isinstance(fields, dict):
                    layer = parsed.get(group)
                    if not isinstance(layer, dict):
                        types.append(group)
                        missing.extend(f'{group}.{key}' for key in fields)
                        continue
                    for key in fields:
                        if key not in layer:
                            missing.append(f'{group}.{key}')
                        elif layer[key] is None:
                            nulls.append(f'{group}.{key}')
                elif not isinstance(parsed.get(group), list):
                    types.append(group)
        rows.append({'case_id': cid, 'parsed': parsed is not None, 'parse_error': error,
                     'missing_fields': missing, 'null_fields': nulls, 'wrong_layer_types': types})
    if run['status'] == 'complete' and (seen != set(cases) or any(r.get('transport_error') for r in rows)):
        raise ValueError('Run incorrectly marked complete')
    return {'provenance': 'passed', 'run_status': run['status'], 'expected_cases': len(cases),
            'results': rows, 'note': 'JSON shape and provenance checks are not semantic accuracy scores.'}
