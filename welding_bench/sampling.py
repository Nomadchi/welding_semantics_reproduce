"""One temporal sampler shared by every model adapter. No random seeks."""
import math
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .common import file_sha, read, save, sha
from .prompt import SYSTEM, make_prompt


def uniform_indices(frame_count, nframes):
    if nframes < 2 or nframes % 2 or nframes > frame_count:
        raise ValueError('nframes must be even, >= 2 and <= the declared frame count')
    return np.linspace(0, frame_count - 1, nframes).round().astype(int).tolist()


def resolve_indices(requested, selected, declared, decoded):
    """Only allow the historically used one-frame-at-EOF fallback; log both lists."""
    if decoded not in (declared, declared - 1):
        raise ValueError(f'Decoded {decoded} frames, container declares {declared}')
    actual = []
    for index in requested:
        if index in selected:
            actual.append(index)
        elif index == declared - 1 and decoded == declared - 1 and index - 1 in selected:
            actual.append(index - 1)
        else:
            raise ValueError(f'Missing requested frame {index}')
    return actual


def decode(path, counts):
    path = Path(path).expanduser().resolve()
    digest = file_sha(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if min(width, height, declared) <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid video metadata')
        requested = {n: uniform_indices(declared, n) for n in counts}
        wanted = set(i for group in requested.values() for i in group) | {declared - 2}
        selected, decoded = {}, 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr.shape[:2] != (height, width):
                raise ValueError('Frame dimensions changed')
            if decoded in wanted:
                selected[decoded] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decoded += 1
    finally:
        cap.release()
    if file_sha(path) != digest:
        raise ValueError('Source changed while decoding')
    meta = {'source': str(path), 'source_sha256': digest, 'width': width, 'height': height,
            'frame_count': declared, 'decoded_frame_count': decoded, 'fps': fps,
            'duration_s': declared / fps, 'decoder': 'opencv_sequential'}
    result = {}
    for n, indices in requested.items():
        actual = resolve_indices(indices, selected, declared, decoded)
        result[n] = (indices, actual, np.stack([selected[i] for i in actual]))
    return meta, result


def montage(frames, times, path):
    cols = 4 if len(frames) <= 16 else 8
    sheet = Image.new('RGB', (240 * cols, 202 * math.ceil(len(frames) / cols)), (20, 25, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (frame, time_s) in enumerate(zip(frames, times)):
        image = Image.fromarray(frame)
        image.thumbnail((240, 180))
        x, y = i % cols * 240, i // cols * 202
        sheet.paste(image, (x, y + 22))
        draw.text((x + 4, y + 4), f'{time_s:.6f}s', fill='white')
    sheet.save(path)


def prepare(paths, counts, output):
    output = Path(output)
    if not paths or len(set(counts)) != len(counts):
        raise ValueError('Need videos and unique frame counts')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'version': 1, 'sampling': 'linspace(0, N-1, n).round(); OpenCV sequential RGB', 'cases': []}
    for order, path in enumerate(paths, 1):
        meta, groups = decode(path, counts)
        for n, (requested, actual, frames) in groups.items():
            case_id = f'video_{order:04d}_{n}'
            directory = output / case_id
            directory.mkdir()
            times = [i / meta['fps'] for i in actual]
            case = {'case_id': case_id, 'video_id': f'video_{order:04d}', 'nframes': n, 'video': meta,
                    'requested_indices': requested, 'actual_indices': actual, 'sampled_times_s': times,
                    'system': SYSTEM, 'prompt': make_prompt(meta['duration_s']), 'images': []}
            for i, frame in enumerate(frames):
                target = directory / f'{i:03d}.png'
                Image.fromarray(frame).save(target)
                case['images'].append({'path': target.relative_to(output).as_posix(),
                    'sha256': file_sha(target), 'rgb_sha256': sha(frame.tobytes())})
            montage(frames, times, directory / 'montage.jpg')
            audit = directory / 'case.json'
            save(audit, case)
            manifest['cases'].append({'case_id': case_id, 'path': audit.relative_to(output).as_posix(), 'sha256': file_sha(audit)})
    save(output / 'manifest.json', manifest)
    return manifest


def load_cases(directory, counts=None):
    directory = Path(directory).resolve()
    manifest = read(directory / 'manifest.json')
    result, seen = [], set()
    for item in manifest['cases']:
        path = (directory / item['path']).resolve()
        if directory not in path.parents or file_sha(path) != item['sha256']:
            raise ValueError('Case manifest hash/path mismatch')
        case = read(path)
        if not re.fullmatch(r'video_\d{4,}_\d+', case['case_id']):
            raise ValueError('Invalid case ID')
        if case['case_id'] != item['case_id'] or case['case_id'] in seen:
            raise ValueError('Duplicate or inconsistent case ID')
        seen.add(case['case_id'])
        if counts and case['nframes'] not in counts:
            continue
        if case['system'] != SYSTEM or case['prompt'] != make_prompt(case['video']['duration_s']):
            raise ValueError('Case prompt differs from shared protocol')
        if case['requested_indices'] != uniform_indices(case['video']['frame_count'], case['nframes']):
            raise ValueError('Case sampling differs from shared protocol')
        v = case['video']
        if not math.isfinite(v['fps']) or v['fps'] <= 0 or v['duration_s'] != v['frame_count'] / v['fps']:
            raise ValueError('Invalid timing metadata')
        available = {i: True for i in case['actual_indices']}
        expected = resolve_indices(case['requested_indices'], available, v['frame_count'], v['decoded_frame_count'])
        if expected != case['actual_indices'] or case['sampled_times_s'] != [i / v['fps'] for i in expected]:
            raise ValueError('Actual sampling/timestamps differ from shared protocol')
        if len(case['images']) != case['nframes']:
            raise ValueError('Wrong frame count')
        frames = []
        for image in case['images']:
            path = (directory / image['path']).resolve()
            if directory not in path.parents or file_sha(path) != image['sha256']:
                raise ValueError('Input image hash/path mismatch')
            rgb = np.asarray(Image.open(path).convert('RGB'))
            if rgb.shape != (v['height'], v['width'], 3) or sha(rgb.tobytes()) != image['rgb_sha256']:
                raise ValueError('RGB hash mismatch')
            frames.append(rgb)
        result.append((case, np.stack(frames)))
    if not result:
        raise ValueError('No cases selected')
    return result
