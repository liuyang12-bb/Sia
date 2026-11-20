"""Archive actual source, frozen inputs, history fixtures and new raw evidence."""
import hashlib, json, platform, sys, time, zipfile
from experiment import OUT, ROOT, LIVE, PREVIOUS, sha, save, load


def main():
    p = load()
    for name, digest in p['source_hashes'].items():
        assert sha(LIVE / name) == digest, name
    inherited = PREVIOUS / 'complete_rebuildable_evidence.zip'
    previous_manifest = json.loads((PREVIOUS / 'delivery_manifest.json').read_text())
    assert sha(inherited) == previous_manifest['archive_sha256']
    save(OUT / 'environment.json', dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), source_fingerprint=p['source_fingerprint']))
    hashes = {}
    archive = OUT / 'complete_rebuildable_evidence.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        def add_bytes(name, data):
            digest = hashlib.sha256(data).hexdigest().upper()
            if name in hashes:
                assert hashes[name] == digest, ('DUPLICATE_WITH_DIFFERENT_CONTENT', name)
                return
            z.writestr(name, data)
            hashes[name] = digest
        # Parent archive supplies unchanged real-source snapshots, official
        # read-only inputs and historical regression fixtures. Avoid zip nesting.
        with zipfile.ZipFile(inherited) as old:
            for name in old.namelist():
                if not name.endswith('.zip'):
                    data = old.read(name)
                    assert hashlib.sha256(data).hexdigest().upper() == previous_manifest['archive_files'][name]
                    add_bytes(name, data)
        for directory in [LIVE, OUT]:
            for f in sorted(directory.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.pyc', '.zip']:
                    add_bytes(f.relative_to(ROOT).as_posix(), f.read_bytes())
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(hashes)
        for name, digest in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest().upper() == digest, name
    save(OUT / 'delivery_manifest.json', dict(archive=archive.name,
        archive_sha256=sha(archive), archive_files=hashes, archive_readback_valid=True,
        inherited_archive_sha256=sha(inherited), created_unix=time.time(),
        current_source_hashes=p['source_hashes'], input_hashes=p['input_hashes'],
        warning=p['warning']))
    print('PACKAGE_VERIFIED', len(hashes), sha(archive), flush=True)


if __name__ == '__main__':
    main()
