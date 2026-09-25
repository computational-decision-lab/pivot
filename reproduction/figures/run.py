"""Reconstruct the ten frozen manuscript figures from packaged evidence.

Usage: python reproduction/figures/run.py --output /path/to/figures
The output mirrors the manuscript's figures/ directory. No experiment is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INPUT_HASHES = {
    'data/metadrive/summary.json': 'afa33f6e334966a9bd3c034d20b778e1a692d3ef37d068923226239a49594a0d',
    'data/metadrive/seed_results.csv': 'f67607ee6e4135f7cedd92d565b84c3a2c4480874e3ce4788d82f343e5242e13',
    'data/leduc/summary.json': '8b6e68e71497fe9de396c2c342a762c95830b4bd3717c5a1c118949965494529',
    'base/fig2_operator_shift.pdf': '28717c6b6173dce32f58e096ea985c9862138cc9b7adc2a8c64ab3765acfb3a1',
    'base/fig5_closed_loop.pdf': 'ec4e1469132e0cd4e1934530a26bab89c7123de2a8b1257ced9af61c93a179da',
    'base/figA_response_footprint.pdf': '22581c90601c810077fc2a0c1cdb1edeb39e0db5d366695a39a1d253e4be359b',
    'base/01_proxy_deployment_reversal.pdf': '588c7a8acc3dc78a64e0eee140406212950defb9c0362071937e133bd4bd584d',
    'base/05_v4_component_contrasts.pdf': 'b8b270af819dec522390760ca1210cb87b940168f7fb0bf8a3a2381fe3477ff8',
}

# Relative to the manuscript's figures/ directory. Hashes are from the final
# 2026-09-26 Overleaf readback; they pin the visual reference for every plot.
FIGURES = [
    ('fig1_improvement_reversal.pdf', '267e2172abaa97434d5431c3e95006052abb98bcc533f01e3253e5f02e699c6e', 'supplied'),
    ('fig2_operator_shift.pdf', '4dfbaafab23fe92f6b26c1183a83231a71960b7aa0bfb6635b97e816b0069c52', 'replayed'),
    ('fig1_decision_relevance.pdf', 'd30b5d8e4a8d4f9360926c61a0d2a1e6d5718f223baa32e0c2063bfce2107d27', 'supplied'),
    ('fig4_evidence_efficiency.pdf', '66a796b6deb61dd20f953dd14bebe41179ae9199019d559dbcc23de5007849ea', 'supplied'),
    ('figA_response_footprint.pdf', '9345213234ff20a0d9a523ff17cfe9a722d0d824d9179bee6b74e4a9d573c9f4', 'replayed'),
    ('additional_leduc/01_proxy_deployment_reversal.pdf', '6b5f0f7554559495f011b294906a96ec3d2ac5132165f2fce0fe1c38543aeb36', 'replayed'),
    ('additional_leduc/05_v4_component_contrasts.pdf', '4a5172b63ebac3d32b5c7864ad1b4f3103b702a9eb528d14c6b65c0d40738689', 'generated'),
    ('fig5_closed_loop.pdf', '69189f0643fcb5a4672ba30197c11153c99d621f0425daf3d96d8c4108723a7e', 'replayed'),
    ('figC_posterior_robustness.pdf', 'b394018c2f55d7de29097357ca156b14763880d2dd39814969efb618d4a5cda0', 'supplied'),
    ('additional_metadrive/matched_posterior_selected_audit_gain.pdf', 'b473f411fb3c9d2e9f80c3017fd84082ac303bc39896041f777ab17bc123a03d', 'generated'),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'{path}: SHA-256 {actual}, expected {expected}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='Destination directory corresponding to manuscript figures/')
    args = parser.parse_args()
    output = args.output.resolve()
    for rel, digest in INPUT_HASHES.items():
        check(ROOT / rel, digest)
    for rel, digest, _ in FIGURES:
        check(ROOT / 'assets' / Path(rel).name, digest)

    output.mkdir(parents=True, exist_ok=True)
    (output / 'figures/additional_leduc').mkdir(parents=True, exist_ok=True)
    (output / 'figures/additional_metadrive').mkdir(parents=True, exist_ok=True)
    for rel, _, kind in FIGURES:
        if kind == 'supplied':
            source = ROOT / 'assets' / Path(rel).name
            target = output / 'figures' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    subprocess.run([sys.executable, str(ROOT / 'source/replay_legacy_labels.py'),
                    '--output', str(output)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'source/refine_leduc_and_response.py'),
                    '--output', str(output)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'source/replot_metadrive.py'),
                    '--data-dir', str(ROOT / 'data/metadrive'),
                    '--output-dir', str(output / 'figures/additional_metadrive')], check=True)

    records = []
    for number, (rel, final_hash, kind) in enumerate(FIGURES, 1):
        target = output / 'figures' / rel
        source = ROOT / 'assets' / Path(rel).name
        actual = sha256(target)
        if kind == 'supplied':
            check(target, final_hash)
            comparison = 'byte-identical'
        else:
            import fitz
            import numpy as np
            reference = fitz.open(source)
            produced = fitz.open(target)
            if len(reference) != len(produced) or reference[0].rect != produced[0].rect:
                raise AssertionError(f'page geometry changed: {rel}')
            a = reference[0].get_pixmap(dpi=144, alpha=False)
            b = produced[0].get_pixmap(dpi=144, alpha=False)
            if (a.width, a.height) != (b.width, b.height):
                raise AssertionError(f'render geometry changed: {rel}')
            pixels_a = np.frombuffer(a.samples, dtype=np.uint8)
            pixels_b = np.frombuffer(b.samples, dtype=np.uint8)
            changed = int(np.count_nonzero(pixels_a != pixels_b))
            changed_fraction = changed / pixels_a.size
            if changed_fraction > 0.001:
                raise AssertionError(f'visual drift {changed_fraction:.3%}: {rel}')
            comparison = f'144-dpi pixel difference {changed_fraction:.5%}'
        records.append(dict(figure=number, path='figures/' + rel,
                            provenance=kind, sha256=actual,
                            frozen_reference_sha256=final_hash,
                            reference_comparison=comparison))
    manifest = dict(version='2026-09-26 final Overleaf readback f7afee89',
                    scope='saved-evidence plotting and PDF replay; no simulation rerun',
                    input_sha256=INPUT_HASHES, figures=records)
    (output / 'figure-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'figures': len(records),
                      'classes': {kind: sum(r['provenance'] == kind for r in records)
                                  for kind in ['generated', 'replayed', 'supplied']},
                      'manifest': str(output / 'figure-manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
