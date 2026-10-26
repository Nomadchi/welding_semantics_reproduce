import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def parse_answer(text):
    """Remove only complete think blocks / an outer code fence; never repair JSON."""
    final = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if final.startswith('```') and final.endswith('```'):
        final = re.sub(r'^```(?:json)?\s*', '', final).removesuffix('```').strip()
    try:
        def reject_nonfinite(value):
            raise ValueError(f'Non-finite constant is not valid JSON: {value}')
        value = json.loads(final, parse_constant=reject_nonfinite)
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        return final, value, None
    except (ValueError, json.JSONDecodeError) as exc:
        return final, None, str(exc)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'welding_bench').glob('*.py')) + list((root / 'scripts/test').glob('*.py'))
    return {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(paths)}
