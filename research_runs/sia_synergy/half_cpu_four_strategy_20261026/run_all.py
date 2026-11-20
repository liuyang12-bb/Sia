"""Serial controller. Existing partial outputs are never overwritten."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    frozen = OUT / 'preregistration.json'
    plan = json.loads(frozen.read_text(encoding='utf-8'))
    if sha(__file__) != plan['controller_sha256']:
        raise RuntimeError('CONTROLLER_CHANGED')
    work = Path(plan['tree']) / 'sia-simulator'
    for cell in plan['run_order']:
        folder = OUT / cell['id']
        if folder.exists():
            raise RuntimeError(f'REFUSE_EXISTING_OUTPUT:{folder}')
        available = psutil.virtual_memory().available
        if available < plan['memory_gate_gib'] * 2**30:
            raise RuntimeError(f'MEMORY_GATE_2GiB:{available / 2**30:.3f}')
        if psutil.disk_usage(str(OUT)).free < plan['disk_gate_gib'] * 2**30:
            raise RuntimeError('DISK_GATE_5GiB')
        command = [sys.executable, '-B', str(OUT / 'run_cell.py'), '--manifest', str(frozen),
                   '--cell', cell['id'], '--output', str(folder)]
        started = time.perf_counter()
        print('START', cell['id'], flush=True)
        with (OUT / (cell['id'] + '.log')).open('x', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
        event = dict(cell=cell, command=command, exit_code=completed.returncode,
                     elapsed_seconds=time.perf_counter() - started,
                     finished_utc=datetime.now(timezone.utc).isoformat(),
                     output_hashes={entry.name: sha(entry) for entry in folder.iterdir() if entry.is_file()}
                     if folder.exists() else {})
        with (OUT / 'execution_manifest.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')
        print('END', cell['id'], completed.returncode, round(event['elapsed_seconds'], 3), flush=True)
        if completed.returncode:
            raise RuntimeError(f'CELL_FAILED_STOP:{cell["id"]}')


if __name__ == '__main__':
    main()
