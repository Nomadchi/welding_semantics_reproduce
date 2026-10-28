# Welding Semantics Reproduce

焊接视频语义测试框架，统一 **Prompt、视频解码、抽帧索引、时间戳和结果字段**。支持 Code-as-World、Cosmos3-Nano Reasoner、RoBoBrain2.5、RynnBrain1.1-9B、Qwen3.8-27B 和 Responses 兼容 API。

## 入口与目录

在项目根目录使用 `python -m welding_bench <命令>`，各命令的参数通过 `--help` 查看。

```bash
python -m welding_bench list          # 列出实验
python -m welding_bench build-index   # 生成首页
python -m welding_bench browse        # 浏览 http://127.0.0.1:8765/index.html
```

```text
welding_bench/      Prompt、抽帧、模型适配器、推理与核验
scripts/test/       测试命令入口
scripts/download/   权重下载脚本
requirements/       环境依赖
tests/             框架测试
.venv*/             Python 环境
upstream/           上游项目及使用说明
weights/            模型权重
video_samples/      输入视频
inputs/             冻结输入
results/            按“模型名_测试时间”保存实验
web/、*.html        浏览页面及展示数据
docs/、reports/    实验文档与汇总表格
logs/               运行与校验日志
```

每次 `run` 自动创建 `results/<模型名>_<YYYYMMDD_HHMMSS>/`，使用北京时间，并打印 `EXPERIMENT_DIRECTORY`。一次调用中的视频和帧数条件保存在同一目录。

## 测试协议

- **抽帧**：OpenCV 顺序解码，默认每段视频取 16、64 帧；支持自定义偶数帧数。容器声明帧数为 `N` 时，索引为 `numpy.linspace(0, N-1, n).round().astype(int)`。最后一帧缺失且实际总帧数恰少一帧时，使用前一帧并记录替代。
- **时间**：时间戳为 `实际帧索引 / fps`，时长为 `声明帧数 / fps`。可变帧率视频先转换为恒定帧率。
- **Prompt**：system 为 `You are a careful visual observer. Separate evidence from hypotheses. Respond with JSON only.`；任务采用 `schema_fields_only`，包含五层字段列表和视频时长。模型消息由采样画面、时间信息和共享 Prompt 组成。
- **语义**：geometry、physical、process、temporal、intent_quality 五层，共 34 个字段，另含 evidence_notes。
- **生成**：本地模型默认 BF16、seed=1、贪心生成、3072 tokens；API 可配置推理强度和 token 预算。
- **记录**：先冻结输入与预处理审计，再执行推理。原始回答、缺失字段和 null 按原样保留；解析时移除完整 `<think>...</think>` 和外层代码围栏。

各模型复用抽帧结果，采用对应的媒体预处理：

| 后端 | 媒体输入 | 预处理与模板 |
|---|---|---|
| `codeasworlds` | 视频张量及 metadata | factor 28、bicubic antialias、每帧最多 262144 像素；官方 `qwen3_5_no_think.jinja` |
| `cosmos3` | 视频序列 | 官方处理器与模板，直接使用冻结帧；每帧预算 65536～262144 像素 |
| `robobrain` | 视频序列 | 模型自带处理器与模板，直接使用冻结帧 |
| `rynnbrain` | 视频序列 | 模型自带处理器与模板，`enable_thinking=False`，记录模型时间戳 |
| `qwen38` | 视频序列及帧索引、FPS | 官方处理器与模板，vLLM 推理，默认 `enable_thinking=False` |
| `responses` | 独立 PNG 及逐帧时间戳 | 参考处理器输出张量反归一化为 uint8 图片 |

抽帧拼图用于查看和核验输入。

## 环境

Python 3.11 或更新版本。按用途创建独立环境：

```bash
# 抽帧和框架测试
python -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements/base.txt

# Cosmos3、RoBoBrain、RynnBrain，以及 API 参考像素导出
python -m venv .venv-transformers
source .venv-transformers/bin/activate
python -m pip install -r requirements/transformers.txt

# Code-as-World、Qwen3.8
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -r requirements/codeasworlds.txt
```

Windows 使用对应环境的 `Scripts\Activate.ps1` 激活。Code-as-World 的 `--chat-template` 指向上游项目中的 `code_as_world/templates/qwen3_5_no_think.jinja`。

## 1. 冻结输入

按顺序传入待测视频，所有模型复用生成的输入目录：

```bash
python -m welding_bench prepare \
  --videos video_samples/video1.avi video_samples/video2.avi video_samples/video3.avi video_samples/video4.avi \
  --frames 16 64 --output inputs/benchmark-v1
```

使用新的输入目录，视频帧数应满足采样数量。输出包含采样 PNG、视频与像素哈希、请求及实际索引、时间戳、Prompt 和拼图。

## 2. 本地模型推理

```bash
# Code-as-World 9B；4B 使用 --caw-size 4b 和对应权重
python -m welding_bench run \
  --backend codeasworlds --caw-size 9b \
  --weights weights/code_as_worlds-9b \
  --chat-template upstream/codeasworlds/code_as_world/templates/qwen3_5_no_think.jinja \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --tp 2 --context 16384 --max-tokens 3072

# Cosmos3-Nano Reasoner
python -m welding_bench run \
  --backend cosmos3 --weights weights/cosmos3-nano \
  --inputs inputs/benchmark-v1 --frames 16 64 --max-tokens 3072

# RoBoBrain2.5
python -m welding_bench run \
  --backend robobrain --weights weights/robobrain25-8b-nv \
  --inputs inputs/benchmark-v1 --frames 16 64 --max-tokens 3072

# RynnBrain1.1-9B
python -m welding_bench run \
  --backend rynnbrain --weights weights/rynnbrain11-9b \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --gpu-memory-gib 60 --max-tokens 3072

# Qwen3.8-27B
python -m welding_bench run \
  --backend qwen38 --weights weights/qwen3.8-27b \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --tp 2 --context 16384 --gpu-memory-utilization 0.70 --max-tokens 3072
```

加 `--prepare-only` 可先检查处理器、模板和媒体输入。Transformers 使用 `balanced` 设备映射与 SDPA，`--gpu-memory-gib` 设置每张 GPU 的内存预算；vLLM 通过 `--tp`、`--context` 和 `--gpu-memory-utilization` 配置资源。

RynnBrain 使用 `enable_thinking=False`，记录权重加载、GPU 分配、时间戳和峰值显存。Qwen3.8 可通过 `--qwen-thinking --qwen-reasoning-effort xhigh` 开启思考模式，强度还支持 `medium`、`low`。

## 3. API 测试

先使用参考处理器生成完整请求：

```bash
python -m welding_bench run \
  --backend responses --weights weights/cosmos3-nano \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --base-url http://127.0.0.1:8317/v1 --model gpt-6-astra \
  --reasoning-effort max --max-tokens 32768 --prepare-only
```

将 `<测试目录>` 替换为命令打印的路径，在 API 所在机器上使用同一版本的框架发送请求：

```bash
export WELDING_API_KEY='<API 密钥>'
python -m welding_bench send-api --output "<测试目录>"
```

PowerShell 使用 `$env:WELDING_API_KEY = '<API 密钥>'`。`--base-url` 和 `--reasoning-effort` 按服务配置填写；API token 预算包含推理 token。跨机器发送时复制完整测试目录。

## 4. 核验与比较

```bash
python -m welding_bench validate --inputs inputs/benchmark-v1 --output "<测试目录>"
python -m welding_bench compare-inputs "<Cosmos测试目录>" "<RoBoBrain测试目录>"
python -m unittest discover -s tests -v
```

`validate` 核对请求、图像、Prompt 哈希及原始回答与 JSON 的一致性，统计解析失败、缺字段、null 和类型偏差。`compare-inputs` 比较相同视频与帧数条件下的像素、索引、时间戳、Prompt 和后端配置。

每个测试目录包含：

| 文件 | 内容 |
|---|---|
| `experiment.json` | 模型、时间、状态与结果数 |
| `protocol.json` | 配置、依赖版本、源码与输入哈希 |
| `run.json` | 原始回答、解析结果、耗时与运行状态 |
| `<case_id>/input_audit.json` | 时序信息、预处理、图像哈希和 Prompt |
| `<case_id>/raw_response.json` 或 API 响应文件 | 原始模型响应 |
| `<case_id>/input_montage.jpg` | 实际输入拼图 |

## 扩展模型

在 `welding_bench/adapters.py` 中实现预处理、模型加载和生成，复用 `sampling.py`、`prompt.py` 与输入 manifest。
