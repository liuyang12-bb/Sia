"""Same frozen audit, independent output location for the six available cells."""
import hashlib
from experiment import OUT,save,sha

target=OUT/'audit_at_memory_stop';target.mkdir(exist_ok=False)
code=(OUT/'audit.py').read_text(encoding='utf-8')
for name in ['independent_audit.json','comparison_summary.json','per_job.csv']:
    old=f"OUT/'{name}'";new=f"RESULT_OUT/'{name}'"
    assert code.count(old)==1,(name,code.count(old))
    code=code.replace(old,new)
save(target/'auditor_provenance.json',dict(original_auditor_sha256=sha(OUT/'audit.py'),
    transformed_auditor_sha256=hashlib.sha256(code.encode()).hexdigest().upper(),
    transformation='Only three output paths moved; all frozen validation and read paths unchanged.',
    no_training_run=True))
scope=dict(__name__='audit_available_scope',__file__=str(OUT/'audit.py'),RESULT_OUT=target)
exec(compile(code,str(OUT/'audit.py'),'exec'),scope)
scope['main']()
