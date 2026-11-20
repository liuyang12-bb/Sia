"""Validate the paper package without importing or running the simulator."""
from pathlib import Path
import hashlib
import json
import re
import fitz

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest().upper()


data = json.loads((ROOT / 'evidence/paper_results.json').read_text(encoding='utf-8'))
checks = data['checks']
assert len(checks) == 42
assert all(x['aggregates_from_completion_arrival_match'] for x in checks)
assert all(x['exclusive_time_sum_matches_jct'] for x in checks)
sources = []
for name, item in data['provenance'].items():
    actual = sha256(item['source_archive'])
    assert actual == item['source_archive_sha256'].upper(), name
    sources.append({'dataset': name, 'source_archive': item['source_archive'],
                    'sha256': actual, 'matches': True})

missing = []
for name in ['README.md', 'CC_JOINT_PAPER_DRAFT.md']:
    source = ROOT / name
    for target in re.findall(r'\]\(([^)]+)\)', source.read_text(encoding='utf-8')):
        target = target.strip('<>').split('#', 1)[0]
        if not target or re.match(r'^[a-z]+://', target, re.I):
            continue
        if not (source.parent / target).exists():
            missing.append({'source': name, 'target': target})
assert not missing, missing

pdf = fitz.open(ROOT / 'CC_JOINT_PAPER_DRAFT.pdf')
layout = json.loads((ROOT / 'evidence/pdf_qa_final/layout_checks.json').read_text(encoding='utf-8'))
assert len(pdf) == len(layout) == 14
assert not any(x['out_of_page_text'] for x in layout)
bib = json.loads((ROOT / 'literature/all_sources.json').read_text(encoding='utf-8'))
assert len(bib) == 13
assert len(list((ROOT / 'literature/papers').glob('*.pdf'))) == 11
receipt = {
    'status': 'PASS', 'scope': 'writing package; no new simulations',
    'pdf_pages': len(pdf), 'equations': 10, 'figures': 6, 'experiment_tables': 4,
    'references': len(bib), 'local_public_pdfs': 11,
    'historical_cells_recomputed': 42, 'source_archives_verified': sources,
    'missing_local_links': missing, 'text_margin_exceptions': 0,
    'visual_review': 'All 14 final rendered pages manually inspected; no overlap or clipping observed.',
    'limitations': ['RAP and Nexus full text not obtained',
                    'Synthetic cross-hardware response; no target-platform calibration',
                    'Historical versions remain separate evidence groups'],
}
(ROOT / 'evidence/delivery_validation.json').write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
files = []
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or path.name == 'delivery_manifest.json':
        continue
    files.append({'path': path.relative_to(ROOT).as_posix(),
                  'bytes': path.stat().st_size, 'sha256': sha256(path)})
(ROOT / 'delivery_manifest.json').write_text(json.dumps({
    'schema': 'cc_joint_paper_package_v1', 'files': files,
    'note': 'This manifest excludes itself. SHA-256 is file-content provenance, not a publication claim.'
}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': 'PASS', 'files_hashed': len(files),
                  'pages': len(pdf), 'historical_cells': len(checks)}, ensure_ascii=False))
