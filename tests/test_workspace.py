from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from welding_bench.workspace import new_run_path, collect_runs, build_index

class WorkspaceTests(unittest.TestCase):
    def test_timezone_and_collision(self):
        args=SimpleNamespace(backend='responses',model='team/model',caw_size='9b')
        with tempfile.TemporaryDirectory() as tmp:
            now=datetime(2026,10,28,1,2,3,456789,tzinfo=timezone.utc)
            p=new_run_path(args,tmp,now)
            self.assertEqual(p.name,'team-model_20261028_090203')
            p.mkdir()
            q=new_run_path(args,tmp,now)
            self.assertNotEqual(p,q)
            q.mkdir()
            with self.assertRaises(FileExistsError):new_run_path(args,tmp,now)

    def test_index_reads_state_and_escapes_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'results/model_20261028_090000';run.mkdir(parents=True)
            (run/'experiment.json').write_text('{"model":"<model>","time":"2026-10-28T01:00:00Z","status":"prepared"}')
            (run/'run.json').write_text('{"status":"complete","results":[{}]}')
            rows=build_index(root)
            self.assertEqual(rows[0]['status'],'complete')
            self.assertEqual(rows[0]['result_count'],1)
            page=(root/'index.html').read_text(encoding='utf-8')
            self.assertIn('&lt;model&gt;',page)
            self.assertIn('results/model_20261028_090000/run.json',page)

    def test_existing_run_is_not_overwritten_and_failure_is_registered(self):
        from welding_bench.runner import run
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'test';out.mkdir()
            sentinel=out/'experiment.json';sentinel.write_text('{"keep":true}')
            args=SimpleNamespace(output=out,resume=False,backend='responses',model='astra',caw_size='9b',frames=[16])
            with self.assertRaises(FileExistsError):run(args)
            self.assertEqual(sentinel.read_text(),'{"keep":true}')
            args.output=root/'failed'
            def fail(value):
                value.output.mkdir()
                raise RuntimeError('processor error')
            with patch('welding_bench.runner._run',side_effect=fail), self.assertRaises(RuntimeError):run(args)
            meta=json.loads((args.output/'experiment.json').read_text())
            self.assertEqual(meta['status'],'failed')
            self.assertIn('processor error',meta['last_error'])

if __name__=='__main__':unittest.main()
