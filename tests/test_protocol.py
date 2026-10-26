import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from welding_bench.adapters import build_api_request
from welding_bench.common import file_sha, parse_answer, read, save, source_hashes
from welding_bench.api import execute_frozen, send
from welding_bench.prompt import SYSTEM, make_prompt
from welding_bench.sampling import decode, load_cases, prepare, resolve_indices, uniform_indices


class ProtocolTests(unittest.TestCase):
    def test_known_frame_positions(self):
        self.assertEqual(uniform_indices(10, 4), [0, 3, 6, 9])
        for n in (0, 1, 3, 12):
            with self.assertRaises(ValueError):
                uniform_indices(10, n)

    def test_only_last_frame_fallback(self):
        self.assertEqual(resolve_indices([0,3,6,9], {0:1,3:1,6:1,8:1},10,9), [0,3,6,8])
        with self.assertRaises(ValueError):
            resolve_indices([0,3,6,9], {0:1,6:1,8:1},10,9)
        with self.assertRaises(ValueError):
            resolve_indices([0,3,6,9], {0:1,3:1,6:1,9:1},10,8)

    def test_no_answer_repair_or_false_conversion(self):
        self.assertEqual(parse_answer('```json\n{"arc":false,"speed":null,"zero":0}\n```')[1],
                         {'arc':False,'speed':None,'zero':0})
        self.assertIsNone(parse_answer('{"a":1,}')[1])
        self.assertIsNone(parse_answer('explanation {"a":1}')[1])
        self.assertIsNone(parse_answer('[]')[1])
        self.assertIsNone(parse_answer('{"speed":NaN}')[1])

    def test_api_excludes_filenames_and_answers(self):
        case = {'nframes':2,'system':SYSTEM,'prompt':make_prompt(2), 'sampled_times_s':[0,1],
                'video':{'source':'PRIVATE_FILENAME.avi'},'old_answer':'NEVER_SEND_THIS'}
        value = build_api_request(case,[b'first',b'second'],'test-model','max',32768)
        encoded = json.dumps(value)
        self.assertNotIn('PRIVATE_FILENAME', encoded)
        self.assertNotIn('NEVER_SEND_THIS', encoded)
        self.assertEqual(set(value), {'model','input','reasoning','max_output_tokens','store','stream'})
        self.assertEqual(len(value['input']),2)
        self.assertEqual(value['input'][1]['content'][0]['text'],'Timestamp: 0.000000 seconds.')
        self.assertEqual(base64.b64decode(value['input'][1]['content'][3]['image_url'].split(',')[1]),b'second')

    def test_freeze_roundtrip_and_tamper_detection(self):
        # Generate synthetic frames in memory; no real video or inference outputs are tracked.
        frames=np.zeros((4,12,16,3),dtype=np.uint8)
        meta={'source':'private.avi','source_sha256':'source','width':16,'height':12,'frame_count':4,
              'decoded_frame_count':4,'fps':2.0,'duration_s':2.0,'decoder':'opencv_sequential'}
        with tempfile.TemporaryDirectory() as tmp, patch('welding_bench.sampling.decode',return_value=(meta,{4:([0,1,2,3],[0,1,2,3],frames)})):
            out=Path(tmp)/'inputs'
            prepare(['private.avi'],[4],out)
            self.assertEqual(len(load_cases(out)),1)
            with self.assertRaises(FileExistsError):
                prepare(['private.avi'],[4],out)
            (out/'video_0001_4/000.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                load_cases(out)

    def test_sequential_decoder(self):
        class Capture:
            def __init__(self): self.i=0
            def isOpened(self): return True
            def get(self,k):
                import cv2
                return {cv2.CAP_PROP_FRAME_WIDTH:16, cv2.CAP_PROP_FRAME_HEIGHT:12,
                        cv2.CAP_PROP_FRAME_COUNT:10, cv2.CAP_PROP_FPS:5}[k]
            def read(self):
                if self.i==9:return False,None
                self.i+=1
                return True,np.full((12,16,3),self.i,dtype=np.uint8)
            def release(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.write_bytes(b'mocked video')
            with patch('welding_bench.sampling.cv2.VideoCapture',return_value=Capture()):
                meta,groups=decode(source,[4])
            self.assertEqual(meta['decoded_frame_count'],9)
            self.assertEqual(groups[4][1],[0,3,6,8])
            self.assertEqual(groups[4][2][:,0,0,0].tolist(),[1,4,7,9])

    def test_completed_parse_failure_is_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'WELDING_API_KEY':'test-key'}):
            out=Path(tmp);directory=out/'video_0001_2';directory.mkdir()
            (directory/'request.json').write_text('{}')
            protocol={'settings':{'backend':'responses','base_url':'http://localhost/v1','model':'test'},
                      'source_hashes':source_hashes(),'files':{'video_0001_2':{'request.json':file_sha(directory/'request.json')}}}
            save(out/'protocol.json',protocol)
            save(out/'run.json',{'protocol_sha256':file_sha(out/'protocol.json'),'status':'complete',
                 'results':[{'case_id':'video_0001_2','raw_text':'invalid JSON','parsed_json':None}], 'transport_failures':[]})
            with patch('welding_bench.api.send') as mock:
                execute_frozen(out,retry=True)
                mock.assert_not_called()

    def test_sse_preserves_original_response(self):
        response={'id':'unit','model':'test','status':'completed','output':[{'type':'message',
                  'content':[{'type':'output_text','text':'{"arc": false}'}]}]}
        data=('data: '+json.dumps({'type':'response.completed','response':response})+'\n\n').encode()
        class Stream:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __iter__(self):return iter(data.splitlines(keepends=True))
        with tempfile.TemporaryDirectory() as tmp, patch('welding_bench.api.urllib.request.build_opener') as opener:
            opener.return_value.open.return_value=Stream()
            row=send('http://localhost/v1/responses','test-key',b'{}',Path(tmp))
            self.assertEqual(row['raw_text'],'{"arc": false}')
            self.assertEqual((Path(tmp)/'attempt_001.sse').read_bytes(),data)
            self.assertEqual(read(Path(tmp)/row['response_file']),response)


if __name__=='__main__':
    unittest.main()
