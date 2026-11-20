"""Read-only final checks; writes only this maintenance receipt/report."""
from pathlib import Path
import hashlib
import json
import os
import re
from datetime import datetime, timezone

ROOT = Path(r'D:\科研启航\Sia').resolve()
OUT = Path(__file__).resolve().parent


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest().upper()


plan, done = read('archive_manifest.json'), read('cleanup_receipt.json')
assert sha(Path(plan['archive'])) == plan['archive_sha256']
assert done['status'] == 'COMPLETED'
assert all(not (ROOT / x).exists() for x in plan['trees'])
before = read('protected_content_before.json')
for item in before:
    assert sha(ROOT / item['path']) == item['sha256'], item['path']
history = read('historical_results_before.json')
current = {}
for base, dirs, files in os.walk(ROOT / 'research_runs'):
    for name in files:
        p = Path(base) / name
        st = p.stat()
        current[p.relative_to(ROOT).as_posix()] = (st.st_size, st.st_mtime_ns)
assert len(current) == len(history)
for item in history:
    assert current[item['path']] == (item['bytes'], item['mtime_ns']), item['path']
for name in ['README.md', 'PROJECT_INDEX.md']:
    for target in re.findall(r'\]\(([^)]+)\)', (ROOT / name).read_text(encoding='utf-8')):
        if '://' not in target:
            assert (ROOT / target).exists(), (name, target)
workspace = json.loads((ROOT / 'CC_Joint_Research.code-workspace').read_text(encoding='utf-8'))
assert all((ROOT / x['path']).is_dir() for x in workspace['folders'])
json.loads((ROOT / '.vscode/settings.json').read_text(encoding='utf-8-sig'))
# Removing source-backed caches does not remove an installed package or sourceless module.
caches = read('removed_regenerable_bytecode.json')
assert all(Path(x['source']).is_file() and not Path(x['path']).exists() for x in caches)
paper = ROOT / 'research_notes/paper/cc_joint_draft_20261101_v1'
paper_manifest = json.loads((paper / 'delivery_manifest.json').read_text(encoding='utf-8'))
for item in paper_manifest['files']:
    assert sha(paper / item['path']) == item['sha256'], item['path']
result = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
          'protected_content_files_unchanged': len(before),
          'historical_result_files_size_mtime_unchanged': len(history),
          'paper_delivery_files_unchanged': len(paper_manifest['files']),
          'archived_trees': len(plan['trees']), 'archived_files': len(plan['files']),
          'workspace_folders_valid': len(workspace['folders']),
          'source_backed_cache_files_removed': len(caches),
          'net_content_MiB_freed': round(done['net_content_bytes_freed'] / 2**20, 2),
          'archive_sha256': plan['archive_sha256'],
          'algorithm_changes': False, 'new_simulations': 0,
          'git_note': 'Workspace root and CC snapshots are not Git repositories; no reset/clean/commit/push used.',
          'validation_limit': 'Historical logs checked by exact file list, size and nanosecond mtime; not all 16.4 GiB rehashed.'}
(OUT / 'final_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
report = f'''# 项目整理完成报告

## 结果

- 状态：PASS。
- 保留六套有明确用途的源码入口；28套中间版本共{len(plan['files'])}文件完成逐文件归档校验后，从根目录移除。
- 归档前原内容{plan['source_bytes']/2**20:.2f} MiB，压缩包{plan['archive_bytes']/2**20:.2f} MiB；归档内含逐文件清单与hash。
- 清除两个独立环境中有对应源码的{len(caches)}个字节码缓存，释放{done['removed_cache_bytes']/2**20:.2f} MiB。
- 合计净释放约{result['net_content_MiB_freed']:.2f} MiB（文件内容字节口径，不是文件系统块计量）。
- 没有永久删除历史源码；没有更改算法、依赖版本或实验输入；未启动仿真。

## 保留与校验

- 六套保留源码、官方源、research_sources与research_notes共{len(before)}文件SHA-256前后相同。
- research_runs共{len(history)}文件的完整路径、大小、纳秒mtime前后相同；不声称全量重算16.4 GiB日志hash。
- 论文交付的{len(paper_manifest['files'])}个文件全部与原delivery_manifest的hash相同。
- 项目根和CC快照目录不是Git工作树；官方仓库原先的未跟踪缓存/PDF保留，没有reset/clean/提交/推送。
- 六套保留源码的Python文件未发现对本次28个归档目录名的显式引用；历史maintenance脚本仍可能要求恢复旧目录，未改写它们。
- 无实验原生Python进程检查在整理前完成；本轮没有停止、启动或恢复任何实验。

## 新入口

- 根README.md及PROJECT_INDEX.md说明论文版本、修正版、消融版、数据和环境用途。
- CC_Joint_Research.code-workspace提供12个具名目录入口，隐藏缓存并避免监听/搜索大体积原始日志。
- 原根README与VSCode设置备份保存在本目录。
- cc-joint/README.md为2026-10-27历史版本，保持字节原样；根README已解释其“唯一入口”不代表后续版本。

## 恢复

归档：`_archive/cc_intermediate_sources_20261101_v1.zip`

SHA-256：`{plan['archive_sha256']}`

`restore_intermediate_sources.ps1`默认dry run，添加`-Execute`恢复所有中间目录到原路径，拒绝覆盖并逐文件验证hash。归档不是可随意删除的缓存。

## 有意未删除

research_runs原始结果、失败现场、旧报告、既有_archive、tmp早期试验数据/截图和maintenance历史脚本均可能承担溯源/复现作用，保留原路径。两个Python环境只清缓存，安装包未删。

## 整理工具兼容性

归档脚本首次在Python 3.10因Path.is_junction不可用停止，发生在任何归档/删除前；改为检查Windows reparse属性后成功。该修复只影响本轮整理脚本。恢复脚本另外以dry run核验，不实际恢复重复副本。

具体机器验收见final_validation.json、cleanup_receipt.json、archive_manifest.json与cleanup_journal.jsonl。
'''
(OUT / 'CLEANUP_REPORT.md').write_text(report, encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
