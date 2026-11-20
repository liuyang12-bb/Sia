"""Resume only not-yet-started cells after a memory-gate stop."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

OUT = Path(__file__).resolve().parent
MANIFEST = json.loads((OUT / 'preregistration.json').read_text(encoding='utf-8'))
RETRY = OUT / 'attempt2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    records = [json.loads(line) for line in (OUT / 'execution_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == 4
    assert [row['cell']['id'] for row in records] == [c['id'] for c in MANIFEST['run_order'][:4]]
    assert [row['exit_code'] for row in records] == [0, 0, 0, 1]
    for row in records[:3]:
        folder = OUT / row['cell']['id']
        for name, expected in row['output_hashes'].items():
            assert sha(folder / name) == expected, (row['cell']['id'], name)
        assert json.loads((folder / 'independent_validation.json').read_text(encoding='utf-8'))['valid']
    assert not (OUT / MANIFEST['run_order'][3]['id']).exists()
    assert all(not (OUT / c['id']).exists() for c in MANIFEST['run_order'][4:])
    active = []
    for process in psutil.process_iter(['name', 'cmdline']):
        argv = process.info['cmdline'] or []
        if (process.pid != psutil.Process().pid and 'python' in (process.info['name'] or '').lower()
                and any(Path(part).name in ('run_cell.py', 'run_all.py') for part in argv)):
            active.append(process.pid)
    if active:
        raise RuntimeError(f'EXISTING_EXPERIMENT_PROCESS:{active}')
    if list(RETRY.iterdir()):
        raise RuntimeError('RETRY_DIR_NOT_EMPTY')
    for cell in MANIFEST['run_order'][3:]:
        # Margin prevents the parent process passing 2 GiB while the child
        # falls below it during imports. Scientific capacity is unchanged.
        memory = psutil.virtual_memory().available / 2**30
        if memory < 2.25:
            raise RuntimeError(f'MEMORY_GATE_WITH_CHILD_IMPORT_MARGIN:{memory:.3f}')
        if psutil.disk_usage(str(RETRY)).free < 5 * 2**30:
            raise RuntimeError('DISK_GATE_5GiB')
        folder = RETRY / cell['id']
        if folder.exists():
            raise RuntimeError('REFUSE_EXISTING_OUTPUT')
        command = [sys.executable, '-B', str(OUT / 'run_cell.py'),
                   '--manifest', str(OUT / 'preregistration.json'),
                   '--cell', cell['id'], '--output', str(folder)]
        started = time.perf_counter()
        print('START', cell['id'], flush=True)
        with (RETRY / (cell['id'] + '.log')).open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=Path(MANIFEST['tree']) / 'sia-simulator',
                                    stdout=log, stderr=subprocess.STDOUT)
        event = dict(cell=cell, command=command, exit_code=result.returncode,
                     elapsed_seconds=time.perf_counter() - started,
                     finished_utc=datetime.now(timezone.utc).isoformat(),
                     output_hashes={entry.name: sha(entry) for entry in folder.iterdir() if entry.is_file()}
                     if folder.exists() else {})
        with (RETRY / 'execution_manifest.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(event, ensure_ascii=False) + '\n')
        print('END', cell['id'], result.returncode, round(event['elapsed_seconds'], 3), flush=True)
        if result.returncode:
            raise RuntimeError(f'RETRY_CELL_FAILED_STOP:{cell["id"]}')


if __name__ == '__main__':
    main()
