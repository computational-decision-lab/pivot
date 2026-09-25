"""Replay the frozen vector-PDF label fixes for legacy Figures 2 and 8."""
from pathlib import Path
import argparse
import collections
import hashlib
import json

import fitz
from matplotlib.font_manager import findfont

ROOT = Path(__file__).resolve().parent.parent
FONT = Path(findfont('DejaVu Sans'))
font = fitz.Font(fontfile=str(FONT))

RULES = {
    'fig2_operator_shift.pdf': [
        ('log(1 + χ2)', 'Design intensity s²', 90, 'center', 3),
        ('global Spearman', 'Fixed global rank', 83, 'center', 1),
    ],
    'fig5_closed_loop.pdf': [
        ('regret (CISR)', 'regret (CISR_C)', 90, 'center', 1),
        ('PIVOT-KG', 'PIVOT batch', 65, 'center', 1),
    ],
}


def replay(name: str, output: Path) -> dict:
    doc = fitz.open(ROOT / 'base' / name)
    page = doc[0]
    old_drawings = page.get_drawings()
    lines = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            lines.append((''.join(span['text'] for span in line['spans']), line))
    replacements = []
    for target, new, max_width, align, expected in RULES[name]:
        found = [line for text, line in lines if text == target]
        assert len(found) == expected, (name, target, len(found), expected)
        for line in found:
            old = fitz.Rect(line['bbox'])
            size = max(span['size'] for span in line['spans'])
            angle = 90 if line['dir'] == (0., -1.) else 0
            page.add_redact_annot(old + (-.05, -.05, .05, .05),
                                  fill=False, cross_out=False)
            replacements.append((old, new, size, max_width, align, angle))
    page.apply_redactions(images=0, graphics=0, text=0)
    page.insert_font(fontname='PIVOTDejaVu', fontfile=str(FONT))
    for old, new, size, max_width, align, angle in replacements:
        if angle:
            size = min(size, (max_width or old.height) / max(font.text_length(new, fontsize=1), 1))
            length = font.text_length(new, fontsize=size)
            x = (old.x0 + old.x1)/2 + (font.ascender + font.descender)*size/2
            y = (old.y0 + old.y1)/2 + length/2
        else:
            size = min(size, (max_width or old.width) / max(font.text_length(new, fontsize=1), 1))
            length = font.text_length(new, fontsize=size)
            x = old.x1-length if align == 'right' else (old.x0 + old.x1-length)/2
            y = (old.y0 + old.y1)/2 + (font.ascender + font.descender)*size/2
        page.insert_text((x, y), new, fontsize=size, fontname='PIVOTDejaVu',
                         color=(0, 0, 0), rotate=angle)
    dest = output / 'figures' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.set_metadata({**doc.metadata, 'modDate': 'D:20260925000000Z'})
    doc.save(dest, garbage=4, deflate=True)
    doc.close()
    rendered = fitz.open(dest)
    def signature(d):
        return str((d['items'], d.get('color'), d.get('fill'), d.get('width'), d.get('dashes')))
    old_paths = collections.Counter(signature(d) for d in old_drawings)
    new_paths = collections.Counter(signature(d) for d in rendered[0].get_drawings())
    assert old_paths == new_paths, f'chart paths changed in {name}'
    return {'figure': name, 'labels_replaced': len(replacements),
            'unchanged_vector_paths': sum(old_paths.values()),
            'base_sha256': hashlib.sha256((ROOT / 'base' / name).read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [replay(name, args.output.resolve()) for name in RULES]
    (args.output / 'legacy-label-audit.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))
