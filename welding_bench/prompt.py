"""Shared welding prompt, without inference imports or environment changes."""
import json
from .semantic_contract import TEMPLATE

PROMPT = '''Analyze only this welding video. Return a JSON object with exactly the structure below, filling values from the visible frames. Use English, concise strings, and null for unknown values; never replace unknown with zero, false or a plausible setting. All *_candidate fields are hypotheses. Use evidence_notes as an array of objects with fields (array of dotted field names), basis (observed/inferred/unknown), confidence (high/medium/low/none), and note. Explain the main hypotheses and unknowns there.
No metric calibration, electrical records, shielding gas, material certificate, WPS, drawing or inspection report is provided. Do not infer physical welding position from image orientation. Bright reflections are not proof of an arc. Previously deposited beads can be visible during a dry run. Check that arc state, phases, pool claims and purpose agree. Proposed requirements are suggestions, not actual acceptance criteria. Do not claim confirmed internal defects or acceptance from appearance. Segmentation must be null unless you actually generate image-coordinate polygons with frame index and class definitions. Do not invent segmentation coordinates. Stages are approximate seconds and should cover the duration. Do not infer a pass role unless visible evidence supports it.
JSON structure:
''' + json.dumps(TEMPLATE, separators=(',', ':'))

SYSTEM = 'You are a careful visual observer. Separate evidence from hypotheses. Respond with JSON only.'
FIELDS_ONLY = PROMPT.partition('JSON structure:')[0] + '\nRequired JSON keys (fill each field; this is a key list, not an answer):\n' + '\n'.join(
    f'{layer}: ' + (', '.join(value) if isinstance(value, dict) else 'array of evidence note objects')
    for layer, value in TEMPLATE.items())

def make_prompt(duration_s):
    return FIELDS_ONLY + f"\nDuration: {duration_s:.6f} seconds."
