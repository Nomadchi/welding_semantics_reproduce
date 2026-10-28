"""Unified command entry point. Heavy model dependencies load only for inference."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import runpy
import sys
from .workspace import ROOT, build_index, collect_runs

def main():
    commands={'run':'run.py','prepare':'prepare_inputs.py','send-api':'send_api.py',
              'validate':'validate.py','compare-inputs':'compare_inputs.py'}
    if len(sys.argv)>1 and sys.argv[1] in commands:
        command=sys.argv.pop(1)
        runpy.run_path(str(ROOT/'scripts/test'/commands[command]),run_name='__main__')
        return
    p=argparse.ArgumentParser(description='Welding workspace: one input protocol, one directory per test.')
    p.add_argument('command',choices=[*commands,'list','build-index','browse'])
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args()
    if args.command=='list':
        print(json.dumps(collect_runs(),ensure_ascii=False,indent=2));return
    rows=build_index()
    if args.command=='build-index':
        print(f'Built {ROOT / "index.html"}: {len(rows)} experiments');return
    handler=functools.partial(SimpleHTTPRequestHandler,directory=str(ROOT))
    print(f'http://127.0.0.1:{args.port}/index.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler).serve_forever()

if __name__=='__main__':main()
