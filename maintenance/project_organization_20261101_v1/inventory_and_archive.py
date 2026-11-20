"""Create a verified, lossless archive. This script never deletes source files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import zipfile

ROOT = Path(r'D:\科研启航\Sia').resolve()
OUT = Path(__file__).resolve().parent
KEEP = ['cc-joint', 'cc-joint-core80-full-lex-repair-v8', 'cc-joint-clean-ab-r3',
        'cc-joint-resource-study', 'cc-joint-breakpoint-precision',
        'cc-joint-recovery-equivalent-v1']
ARCHIVE = ROOT / '_archive/cc_intermediate_sources_20261101_v1.zip'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest().upper()


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def linked(path):
    # Python 3.10 has no Path.is_junction; reject all Windows reparse points.
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)


def entries(directory, content=True):
    out = []
    for base, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            p = Path(base) / name
            if linked(p):
                raise RuntimeError(f'Unreviewed link: {p}')
        for name in sorted(files):
            p = Path(base) / name
            st = p.stat()
            item = {'path': p.relative_to(ROOT).as_posix(), 'bytes': st.st_size,
                    'mtime_ns': st.st_mtime_ns}
            if content:
                item['sha256'] = sha(p)
            out.append(item)
    return sorted(out, key=lambda x: x['path'])


if ARCHIVE.exists() or (OUT / 'archive_manifest.json').exists():
    raise RuntimeError('Refusing to overwrite an existing archive or manifest')
protected = []
for name in KEEP + ['official-artifacts', 'research_sources', 'research_notes']:
    protected.extend(entries(ROOT / name))
dump('protected_content_before.json', protected)
# Historical result bytes are not rewritten; size + nanosecond mtime protect against accidents.
dump('historical_results_before.json', entries(ROOT / 'research_runs', content=False))

trees = sorted(p for p in ROOT.glob('cc-joint-*') if p.is_dir() and p.name not in KEEP)
files, empty_dirs = [], []
for tree in trees:
    assert tree.parent == ROOT and tree.resolve().parent == ROOT
    if linked(tree) or (tree / '.git').exists():
        raise RuntimeError(f'Cannot archive linked/Git tree: {tree}')
    if any(p.name == '.git' for p in tree.rglob('.git')):
        raise RuntimeError(f'Embedded Git: {tree}')
    files.extend(entries(tree))
    for base, dirs, names in os.walk(tree):
        if not dirs and not names:
            empty_dirs.append(Path(base).relative_to(ROOT).as_posix() + '/')
plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT),
        'archive': str(ARCHIVE), 'preserved_trees': KEEP,
        'trees': [p.name for p in trees], 'files': files, 'empty_dirs': empty_dirs,
        'source_bytes': sum(x['bytes'] for x in files)}
dump('archive_plan.json', plan)
print('Creating archive:', len(trees), 'trees;', len(files), 'files', flush=True)
with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for item in files:
        z.write(ROOT / item['path'], item['path'])
    for name in empty_dirs:
        z.writestr(name, '')
    z.writestr('_ARCHIVE_SOURCE_MANIFEST.json', json.dumps(plan, ensure_ascii=False, indent=2))
print('Verifying every archived and original file...', flush=True)
with zipfile.ZipFile(ARCHIVE) as z:
    for item in files:
        assert hashlib.sha256(z.read(item['path'])).hexdigest().upper() == item['sha256'], item['path']
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
plan.update({'status': 'VERIFIED_BEFORE_REMOVAL', 'archive_sha256': sha(ARCHIVE),
             'archive_bytes': ARCHIVE.stat().st_size, 'all_files_verified': True})
dump('archive_manifest.json', plan)
print(json.dumps({k: plan[k] for k in ['status', 'source_bytes', 'archive_bytes', 'archive_sha256']}, ensure_ascii=False), flush=True)
