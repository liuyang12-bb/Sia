"""Package immutable completed evidence; no scheduler or simulation imports."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
DEST = OUT/'completed_analysis_20261028_run1'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest().upper()


def main():
    validation = json.loads((DEST/'final_validation.json').read_text(encoding='utf-8'))
    assert validation['status'] == 'THIRTY_COMPLETE_VALID'
    prereg = json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
    archive = DEST/'complete_evidence_30cells.zip'
    assert not archive.exists()
    files = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p != archive
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    with zipfile.ZipFile(archive, 'x') as z:
        for p in files:
            z.write(p, p.relative_to(OUT).as_posix(),
                    compress_type=zipfile.ZIP_STORED if p.suffix == '.gz' else zipfile.ZIP_DEFLATED,
                    compresslevel=None if p.suffix == '.gz' else 6)
        z.write(Path(__file__), 'maintenance/package_cc_resource_factorial_20261028.py',
                compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in prereg['source_hashes'].items():
            assert hashlib.sha256(z.read('source/'+name)).hexdigest().upper() == expected, name
        for case in prereg['cells']:
            for name in ['start.json', 'rounds.jsonl.gz', 'summary.json', 'output_hashes.json']:
                assert 'cells/'+'_'.join(case)+'/'+name in z.namelist()
    result = dict(status='ARCHIVE_CRC_AND_SOURCE_HASHES_VALID', archive=archive.name,
                  sha256=sha(archive), bytes=archive.stat().st_size, source_files=len(prereg['source_hashes']),
                  complete_raw_cells=30, package_script_sha256=sha(Path(__file__)))
    with (DEST/'archive_validation.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    manifest = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
                and p.name != 'final_delivery_manifest.json'}
    with (DEST/'final_delivery_manifest.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
