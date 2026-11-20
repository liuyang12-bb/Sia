"""Check completed prefix and start only pending cells in a fresh attempt."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

OUT = Path(__file__).resolve().parent
PLAN = json.loads((OUT / 'preregistration.json').read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def completed():
    found = {}
    for source in [OUT] + sorted(p for p in OUT.glob('attempt*') if p.is_dir()):
        log = source / 'execution_manifest.jsonl'
        if not log.exists():
            continue
        for line in log.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            cell = event['cell']['id']
            if event['exit_code'] != 0:
                continue
            if cell in found:
                raise RuntimeError(f'DUPLICATE_SUCCESS:{cell}')
            folder = source / cell
            for name, expected in event['output_hashes'].items():
                if sha(folder / name) != expected:
                    raise RuntimeError(f'COMPLETED_PREFIX_HASH_CHANGED:{cell}:{name}')
            found[cell] = folder
    expected_ids = [cell['id'] for cell in PLAN['run_order']]
    if list(found) != expected_ids[:len(found)]:
        raise RuntimeError('NONPREFIX_OR_WRONG_ORDER')
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    destination = OUT / args.attempt
    if destination.exists():
        raise RuntimeError('ATTEMPT_DIR_EXISTS')
    existing = completed()
    if len(existing) == len(PLAN['run_order']):
        raise RuntimeError('MATRIX_ALREADY_COMPLETE')
    for process in psutil.process_iter(['name', 'cmdline']):
        argv = process.info.get('cmdline') or []
        if (process.pid != psutil.Process().pid and 'python' in (process.info.get('name') or '').lower()
                and any(Path(part).name in ('run_cell.py', 'run_all.py', 'resume.py') for part in argv)):
            raise RuntimeError(f'DUPLICATE_RUNNING_PROCESS:{process.pid}')
    first_available = psutil.virtual_memory().available / 2**30
    if first_available < 2.25:
        raise RuntimeError(f'MEMORY_GATE_BEFORE_ATTEMPT_DIR:{first_available:.3f}')
    # The directory itself is created only after the completed prefix is checked.
    destination.mkdir(exist_ok=False)
    with (destination / 'resume_start.json').open('x', encoding='utf-8') as handle:
        json.dump(dict(prefix={cell: str(path) for cell, path in existing.items()},
                       manifest_sha256=sha(OUT / 'preregistration.json'),
                       safety_margin_gib=2.25), handle, indent=2)
    for cell in PLAN['run_order'][len(existing):]:
        available = psutil.virtual_memory().available / 2**30
        if available < 2.25:
            raise RuntimeError(f'MEMORY_GATE_WITH_CHILD_IMPORT_MARGIN:{available:.3f}')
        if psutil.disk_usage(str(destination)).free < 5 * 2**30:
            raise RuntimeError('DISK_GATE_5GiB')
        folder = destination / cell['id']
        command = [sys.executable, '-B', str(OUT / 'run_cell.py'), '--manifest',
                   str(OUT / 'preregistration.json'), '--cell', cell['id'], '--output', str(folder)]
        started = time.perf_counter()
        print('START', cell['id'], flush=True)
        with (destination / (cell['id'] + '.log')).open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=Path(PLAN['tree']) / 'sia-simulator',
                                    stdout=log, stderr=subprocess.STDOUT)
        event = dict(cell=cell, command=command, exit_code=result.returncode,
                     elapsed_seconds=time.perf_counter() - started,
                     finished_utc=datetime.now(timezone.utc).isoformat(),
                     output_hashes={entry.name: sha(entry) for entry in folder.iterdir() if entry.is_file()}
                     if folder.exists() else {})
        with (destination / 'execution_manifest.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(event, ensure_ascii=False) + '\n')
        print('END', cell['id'], result.returncode, round(event['elapsed_seconds'], 3), flush=True)
        if result.returncode:
            raise RuntimeError(f'CELL_FAILED_STOP:{cell["id"]}')


if __name__ == '__main__':
    main()
