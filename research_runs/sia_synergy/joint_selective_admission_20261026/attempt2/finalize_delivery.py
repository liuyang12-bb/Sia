"""Correct only a manifest self-reference, preserving every provisional file."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def main():
    prior=json.loads((OUT/'delivery_manifest.json').read_text())
    mismatches=[name for name,h in prior['artifacts'].items() if sha(OUT/name)!=h]
    assert mismatches==['delivery_manifest.json'],mismatches
    assert prior['artifacts']['delivery_manifest.json']==hashlib.sha256(b'').hexdigest().upper()
    text=('Final delivery seal: DELIVERY_MANIFEST_FINAL.json.\n'
          'The provisional delivery_manifest.json hashed its own still-empty open file. '
          'Independent checking found exactly that one mismatch; every source, input, raw result, '
          'test, report and other provisional artifact matched. The provisional file and tool are preserved.\n'
          'FINAL_REVIEW.md references the provisional filename; use the final manifest named above. '
          'The final artifacts dictionary is computed before its output is opened, intentionally excluding itself. '
          'No experiment rerun, algorithm, configuration or metric was changed.\n')
    with (OUT/'DELIVERY_SEAL_README.txt').open('x',encoding='utf-8') as f:f.write(text)
    receipt=dict(provisional_mismatches=mismatches,provisional_nonself_entries_verified=len(prior['artifacts'])-1,
        final_seal='DELIVERY_MANIFEST_FINAL.json',results_and_source_unchanged=True,
        provisional_manifest_sha256=sha(OUT/'delivery_manifest.json'))
    with (OUT/'seal_correction.json').open('x') as f:json.dump(receipt,f,indent=2)
    # Critical: construct BEFORE opening the output; no self-hash cycle.
    final=dict(source_version=prior['source_version'],seal_correction=receipt,
        artifacts={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file()})
    with (OUT/'DELIVERY_MANIFEST_FINAL.json').open('x') as f:json.dump(final,f,indent=2)
    reloaded=json.loads((OUT/'DELIVERY_MANIFEST_FINAL.json').read_text())
    assert all(sha(OUT/name)==h for name,h in reloaded['artifacts'].items())
    print('FINAL_SEAL_VERIFIED',len(reloaded['artifacts']),'files; only provisional self-reference corrected',flush=True)
if __name__=='__main__':main()
