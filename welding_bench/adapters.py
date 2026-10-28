"""Model-specific spatial preprocessing; temporal sampling and task text are shared."""
import base64
import io
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
from .common import file_sha, save, sha
from .sampling import montage


def metadata(case):
    v = case['video']
    return {'fps': v['fps'], 'frames_indices': case['actual_indices'],
            'total_num_frames': v['decoded_frame_count'], 'height': v['height'],
            'width': v['width'], 'video_backend': 'opencv_sequential'}


def messages(case):
    return [{'role': 'system', 'content': case['system']},
            {'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': case['prompt']}]}]


def build_api_request(case, pngs, model, effort, budget):
    if len(pngs) != case['nframes']:
        raise ValueError('Wrong API image count')
    content = []
    for time_s, blob in zip(case['sampled_times_s'], pngs):
        content += [{'type': 'input_text', 'text': f'Timestamp: {time_s:.6f} seconds.'},
                    {'type': 'input_image', 'detail': 'original',
                     'image_url': 'data:image/png;base64,' + base64.b64encode(blob).decode('ascii')}]
    content.append({'type': 'input_text', 'text': case['prompt']})
    return {'model': model, 'input': [{'role': 'system', 'content': case['system']},
            {'role': 'user', 'content': content}], 'reasoning': {'effort': effort},
            'max_output_tokens': budget, 'store': False, 'stream': True}


class Adapter:
    def __init__(self, args):
        import torch
        import cv2
        from transformers import AutoProcessor
        cv2.setNumThreads(2)
        torch.set_num_threads(8)
        self.torch, self.args = torch, args
        self.processor = AutoProcessor.from_pretrained(args.weights, local_files_only=True)
        self.engine = None
        self.template_hash = None
        if args.backend == 'rynnbrain':
            self.template_hash = sha(self.processor.chat_template.encode('utf-8'))
        if args.backend == 'qwen38':
            self.template_hash = sha(self.processor.chat_template.encode('utf-8'))
            from vllm.config import ModelConfig
            from vllm.multimodal import MULTIMODAL_REGISTRY
            config = ModelConfig(model=str(args.weights), dtype='bfloat16', max_model_len=args.context)
            self.video_processor = MULTIMODAL_REGISTRY.create_processor(config)
        if args.backend == 'codeasworlds':
            if not args.chat_template:
                raise ValueError('Code-as-World requires its official --chat-template file')
            self.processor.chat_template = Path(args.chat_template).read_text(encoding='utf-8')
            self.template_hash = file_sha(args.chat_template)

    def prepare(self, case, frames, directory):
        torch, processor, args = self.torch, self.processor, self.args
        n = case['nframes']
        audit = {'case_id': case['case_id'], 'nframes': n, 'source_sha256': case['video']['source_sha256'],
                 'actual_indices': case['actual_indices'], 'sampled_times_s': case['sampled_times_s'],
                 'raw_rgb_sha256': [sha(f.tobytes()) for f in frames],
                 'system': case['system'], 'prompt': case['prompt'],
                 'filename_provided_to_model': False, 'montage_sent_to_model': False,
                 'processor_resampling': False, 'chat_template_sha256': self.template_hash}
        if args.backend == 'codeasworlds':
            from qwen_vl_utils.vision_process import smart_resize
            from torchvision.transforms.functional import resize
            from torchvision.transforms import InterpolationMode
            h, w = smart_resize(frames.shape[1], frames.shape[2], factor=28, min_pixels=0, max_pixels=262144)
            pixels = resize(torch.from_numpy(frames).permute(0, 3, 1, 2).float(), [h, w],
                            interpolation=InterpolationMode.BICUBIC, antialias=True)
            rendered = processor.apply_chat_template(messages(case), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            token_ids = processor.tokenizer.encode(rendered, add_special_tokens=False)
            inputs = {'prompt_token_ids': token_ids, 'multi_modal_data': {'video': [(pixels, metadata(case))]},
                      'mm_processor_kwargs': {'fps': case['video']['fps'], 'do_sample_frames': False}}
            preview = np.clip(pixels.permute(0, 2, 3, 1).numpy(), 0, 255).astype('uint8')
            audit.update(spatial_preprocessing='factor28 / bicubic antialias / max262144 pixels; native video tensor',
                         processed_pixels_sha256=sha(pixels.contiguous().numpy().tobytes()),
                         processed_shape=list(pixels.shape), processed_prompt_tokens=len(token_ids))
        else:
            template_kwargs = {}
            if args.backend == 'rynnbrain':
                template_kwargs = {'enable_thinking': False}
                audit.update(thinking_enabled=False, model='Alibaba-DAMO-Academy/RynnBrain1.1-9B')
            if args.backend == 'qwen38':
                template_kwargs = {'enable_thinking': args.qwen_thinking,
                                   'reasoning_effort': args.qwen_reasoning_effort}
                audit.update(thinking_enabled=args.qwen_thinking,
                             reasoning_effort=args.qwen_reasoning_effort if args.qwen_thinking else None)
            rendered = processor.apply_chat_template(messages(case), tokenize=False,
                                                      add_generation_prompt=True, **template_kwargs)
            video_kwargs = {'do_sample_frames': False,
                            'size': {'shortest_edge': 65536*n, 'longest_edge': 262144*n}}
            if args.backend == 'qwen38':
                from vllm.multimodal.processing.inputs import ProcessorInputs
                from vllm.multimodal.processing.context import TimingContext
                media = {'video': [(torch.from_numpy(frames).permute(0,3,1,2), metadata(case))]}
                items = self.video_processor.info.get_data_parser().parse_mm_data(media)
                processed = self.video_processor.apply(
                    ProcessorInputs(prompt=rendered, mm_data_items=items,
                                    hf_processor_mm_kwargs=video_kwargs),
                    timing_ctx=TimingContext(enabled=False))
                video = processed['mm_kwargs']['video'][0].get_data()
                grid = [video['video_grid_thw'].tolist()]
                pixels = video['pixel_values_videos']
                token_ids = np.asarray(processed['prompt_token_ids'], dtype=np.int64)
                inputs = {'prompt': rendered, 'multi_modal_data': media,
                          'mm_processor_kwargs': video_kwargs}
                audit.update(engine='vllm', processed_dtype=str(pixels.dtype),
                             temporal_patch_timestamps_s=video['timestamps'],
                             mm_processor_kwargs=video_kwargs,
                             processor_audit='vLLM native processor including dtype conversion; same raw video and kwargs passed to generation')
            else:
                inputs = processor(text=[rendered], videos=[frames], videos_kwargs={
                    **video_kwargs, 'video_metadata': [metadata(case)]}, return_tensors='pt')
                grid = inputs['video_grid_thw'].tolist()
                pixels = inputs['pixel_values_videos']
                token_ids = inputs.input_ids.numpy()
                if args.backend == 'rynnbrain':
                    timestamps = [float(v) for v in re.findall(r'<(\d+(?:\.\d+)?) seconds>',
                        processor.decode(inputs.input_ids[0], skip_special_tokens=False))]
                    temporal_size = processor.video_processor.temporal_patch_size
                    expected = [(case['sampled_times_s'][i] + case['sampled_times_s'][i+temporal_size-1])/2
                                for i in range(0, n, temporal_size)]
                    if len(timestamps) != len(expected) or any(abs(a-b)>0.051 for a,b in zip(timestamps, expected)):
                        raise ValueError('Video timestamp tokens do not match sampled frames')
                    audit.update(temporal_patch_timestamps_s=timestamps, processed_dtype=str(pixels.dtype),
                                 metadata=metadata(case), temporal_patch_size=temporal_size,
                                 patch_size=processor.video_processor.patch_size)
            vp = processor.video_processor
            patch, temporal, merge = vp.patch_size, vp.temporal_patch_size, vp.merge_size
            if len(grid) != 1 or grid[0][0] * temporal != n:
                raise ValueError(f'Processor changed frame count: {grid}')
            t, gh, gw = grid[0]
            packed = pixels.reshape(t, gh//merge, gw//merge, merge, merge, 3, temporal, patch, patch)
            rgb = packed.permute(0,6,5,1,3,7,2,4,8).reshape(n,3,gh*patch,gw*patch).permute(0,2,3,1).float().numpy()
            preview = np.clip((rgb*np.asarray(vp.image_std)+np.asarray(vp.image_mean))*255, 0, 255).round().astype('uint8')
            audit.update(spatial_preprocessing='official video processor / no resampling / max262144 pixels per frame',
                         video_grid_thw=grid, processed_pixels_sha256=sha(pixels.contiguous().view(torch.uint8).numpy().tobytes()),
                         processed_shape=list(pixels.shape), processed_prompt_tokens=int(token_ids.size),
                         input_ids_sha256=sha(token_ids.tobytes()))
            if args.backend == 'responses':
                blobs = []
                audit['api_images'] = []
                for i, frame in enumerate(preview):
                    buffer = io.BytesIO()
                    Image.fromarray(frame).save(buffer, format='PNG')
                    blob = buffer.getvalue()
                    blobs.append(blob)
                    (directory / f'{i:03d}.png').write_bytes(blob)
                    audit['api_images'].append({'file': f'{i:03d}.png', 'sha256': sha(blob),
                                                'rgb_sha256': sha(frame.tobytes())})
                audit['transport'] = 'independent PNG + timestamps; denormalize/clamp/round uint8; server preprocessing unknown'
                inputs = build_api_request(case, blobs, args.model, args.reasoning_effort, args.max_tokens)
                body = json.dumps(inputs, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                (directory / 'request.json').write_bytes(body)
                audit['request_sha256'] = sha(body)
                inputs = body
        montage(preview, case['sampled_times_s'], directory / 'input_montage.jpg')
        (directory / 'rendered_prompt.txt').write_text(rendered, encoding='utf-8')
        audit['rendered_prompt_sha256'] = sha(rendered.encode('utf-8'))
        audit['rendered_prompt_role'] = 'reference processor only; NOT API text' if args.backend == 'responses' else 'actual local model chat template'
        save(directory / 'input_audit.json', audit)
        return inputs, audit

    def load_model(self, output):
        torch, args = self.torch, self.args
        torch.manual_seed(1)
        if args.backend == 'responses':
            return
        if args.backend in ('codeasworlds', 'qwen38'):
            from vllm import LLM
            extra = {'generation_config': 'vllm', 'limit_mm_per_prompt': {'image': 0, 'video': 1}} if args.backend == 'qwen38' else {}
            self.engine = LLM(model=str(args.weights), trust_remote_code=args.backend == 'codeasworlds', dtype='bfloat16', seed=1,
                max_model_len=args.context, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.context, max_num_seqs=1, enforce_eager=True,
                disable_custom_all_reduce=True, enable_chunked_prefill=args.backend == 'qwen38' or args.caw_size == '9b',
                mm_processor_cache_gb=0, **extra)
        else:
            from transformers import AutoModelForImageTextToText, Cosmos3OmniForConditionalGeneration
            cls = Cosmos3OmniForConditionalGeneration if args.backend == 'cosmos3' else AutoModelForImageTextToText
            self.engine, loading = cls.from_pretrained(args.weights, local_files_only=True, dtype=torch.bfloat16,
                device_map='balanced', max_memory={i: f'{args.gpu_memory_gib}GiB' for i in range(torch.cuda.device_count())},
                attn_implementation='sdpa', output_loading_info=True)
            save(output / 'weight_loading.json', {k: sorted(v, key=str) if isinstance(v, set) else v for k, v in loading.items()})
            if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
                raise RuntimeError('Model loading mismatch; see weight_loading.json')
            if args.backend == 'rynnbrain':
                if loading.get('unexpected_keys'):
                    raise RuntimeError('Unexpected checkpoint keys; see weight_loading.json')
                parameter_devices = sorted({str(p.device) for p in self.engine.parameters()})
                if not parameter_devices or any(not v.startswith('cuda:') for v in parameter_devices):
                    raise RuntimeError('RynnBrain benchmark requires all model weights on GPU')
                save(output / 'model_runtime.json', {
                    'model_class': type(self.engine).__name__,
                    'parameters': sum(p.numel() for p in self.engine.parameters()),
                    'device_map': getattr(self.engine, 'hf_device_map', {'all_parameters': parameter_devices[0]}),
                    'actual_parameter_devices': parameter_devices,
                    'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    'torch_cuda_build': torch.version.cuda, 'dtype': str(self.engine.dtype), 'quantization': None})
            self.engine.eval()

    def generate(self, inputs):
        torch, args = self.torch, self.args
        if args.backend in ('codeasworlds', 'qwen38'):
            from vllm import SamplingParams
            value = self.engine.generate([inputs], SamplingParams(temperature=0, seed=1, max_tokens=args.max_tokens), use_tqdm=False)[0]
            answer = value.outputs[0]
            return {'raw_text': answer.text, 'finish_reason': str(answer.finish_reason), 'output_tokens': len(answer.token_ids)}
        torch.manual_seed(1)
        if args.backend == 'rynnbrain':
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        inputs = inputs.to(self.engine.device, torch.bfloat16)
        with torch.inference_mode():
            generated = self.engine.generate(**inputs, do_sample=False, max_new_tokens=args.max_tokens)
        tokens = generated[0, inputs.input_ids.shape[1]:].cpu().tolist()
        raw = self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = self.engine.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        inputs.to('cpu')
        result = {'raw_text': raw, 'finish_reason': 'stop' if tokens and tokens[-1] in eos else 'length', 'output_tokens': len(tokens)}
        if args.backend == 'rynnbrain':
            result['cuda_peak_allocated_mib'] = [torch.cuda.max_memory_allocated(i)/1024**2 for i in range(torch.cuda.device_count())]
        return result
