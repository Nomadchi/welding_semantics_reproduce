"""One workspace and one directory per experiment; no model-output repair."""
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = timezone(timedelta(hours=8))

def model_slug(args):
    value = {'codeasworlds': 'codeasworlds-'+args.caw_size,
             'cosmos3': 'cosmos3-nano', 'robobrain': 'robobrain25-8b',
             'rynnbrain': 'rynnbrain11-9b', 'qwen38': 'qwen3.8-27b'}.get(args.backend, args.model)
    value = re.sub(r'[^a-zA-Z0-9.-]+', '-', value).strip('.-').lower()
    if not value:
        raise ValueError('The model name must contain letters or numbers')
    return value

def new_run_path(args, root=None, now=None):
    now = now or datetime.now(SHANGHAI)
    if now.tzinfo is None:
        raise ValueError('A timezone-aware test time is required')
    target = Path(root or ROOT/'results') / (model_slug(args)+'_'+now.astimezone(SHANGHAI).strftime('%Y%m%d_%H%M%S'))
    if target.exists():
        target = target.with_name(target.name+now.strftime('%f'))
    if target.exists():
        raise FileExistsError(target)
    return target

def collect_runs(root=ROOT):
    root=Path(root); rows=[]
    for directory in sorted((root/'results').glob('*')):
        if not directory.is_dir():continue
        meta=directory/'experiment.json'
        record=json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {'model':directory.name,'time':'','kind':'unregistered'}
        state=directory/'run.json'
        if state.exists():
            run=json.loads(state.read_text(encoding='utf-8'))
            record={**record,'status':run.get('status',record.get('status','unknown')),'result_count':len(run.get('results',[]))}
        rows.append({**record,'path':directory.relative_to(root).as_posix()})
    return sorted(rows,key=lambda r:r.get('time',''),reverse=True)

def build_index(root=ROOT):
    root=Path(root);rows=collect_runs(root);esc=html.escape
    body=[]
    for r in rows:
        directory=root/r['path']; links=[]
        for name,label in [('report.html','本次报告'),('experiment.json','测试信息'),('run.json','完整结果'),('raw_predictions.json','原始结果'),('protocol.json','测试协议'),('exports','语义导出'),('segmentation_conclusions.json','分割结论')]:
            if (directory/name).exists():links.append('<a href="'+esc(r['path']+'/'+name,quote=True)+'">'+label+'</a>')
        body.append('<tr><td>'+esc(r.get('model',directory.name))+'</td><td>'+esc(directory.name)+'</td><td>'+esc(str(r.get('status','unknown')))+'</td><td>'+esc(str(r.get('result_count') or '—'))+'</td><td>'+' · '.join(links)+'</td></tr>')
    nav=[]
    for name,label in [('results.html','语义结果浏览'),('model_compare.html','模型对比'),('semantics.html','语义定义'),('sam3_segmentation.html','SAM3 分割'),('README.md','使用说明'),('reports/焊接语义复现结果汇总.xlsx','结果汇总表'),('reports/1026_comparison/1026_焊接语义总表.xlsx','1026 对照表')]:
        if (root/name).exists():nav.append('<a href="'+name+'">'+label+'</a>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>焊接语义复现</title><style>body{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#20334b;background:#f5f7fa;margin:0}main{max-width:1500px;margin:35px auto;padding:0 24px}nav{display:flex;gap:22px;flex-wrap:wrap}a{color:#215da0}table{border-collapse:collapse;width:100%;background:white;margin-top:25px}td,th{text-align:left;padding:12px;border-bottom:1px solid #dce3ed}th{background:#243d64;color:white}.scroll{overflow:auto}.muted{color:#607086}h1{font-size:30px}</style><main><h1>焊接语义复现</h1><nav>'''+''.join(nav)+'''</nav><p class="muted">每次测试独立保存原始回答、输入审计与协议；目录时间采用北京时间。历史记录中未保存准确时间的，测试信息会注明时间来源。</p><div class="scroll"><table><thead><tr><th>模型</th><th>测试目录</th><th>执行状态</th><th>结果数</th><th>查看</th></tr></thead><tbody>'''+''.join(body)+'</tbody></table></div></main></html>'
    (root/'index.html').write_text(page,encoding='utf-8')
    return rows
