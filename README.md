# Welding Semantics Reproduce

统一焊接视频的语义测试方法：**共享 Prompt、视频解码、抽帧索引、时间戳与结果字段**，通过模型适配器分别接入 Code-as-World、Cosmos3-Nano Reasoner、RoBoBrain2.5 和 Responses 兼容的多图 API。

本仓库从既有 `repro` 服务器脚本抽取公共逻辑，去除了对历史回答、历史输入审计、固定四条视频及下载记录的依赖。只发布测试框架及本 README；实验文档、结论、权重、原始视频、结果网页与原始回答均不提交。PhysMind/SAM3 的专项分割实验不属于这里的五层语义推理入口。

## 目录

```text
welding_bench/
  semantic_contract.py   五层语义字段定义
  prompt.py              唯一的 system / task Prompt 来源
  sampling.py            顺序解码、均匀抽帧、PNG 输入冻结与哈希检查
  adapters.py            模型专用的空间预处理与本地推理
  api.py                 API 请求、SSE 原始记录与显式通信重试
  runner.py              统一运行入口
  validation.py          输入来源与 JSON 结构核验
scripts/
  test/                  提交：准备输入、推理、发送 API、核验与输入对比
  download/              不提交：下载、断点续传等内部运维脚本
requirements/            CPU、Transformers、vLLM 分开的依赖
tests/                   临时合成输入与模拟 API 的回归测试
inputs/                  不提交：按测试生成的输入与 manifest
outputs/                 不提交：请求、原始响应、审计及解析结果
weights/                 不提交：自行准备的模型与处理器文件
upstream/                不提交：所需的上游模板等外部文件
```

`scripts/download/` 被 Git 整目录忽略，克隆仓库后不存在是正常现象。框架不自动下载权重，也不要求下载脚本生成的专用 manifest。所有模型路径通过 `--weights` 指定，原实验目录不需要搬动。

## 统一协议

1. 使用 OpenCV 从头**顺序解码**，不使用随机 seek 或各模型自带的再次抽帧。默认每段视频分别取 16 和 64 帧，也可指定其他合法偶数帧数。
2. 容器声明帧数为 `N` 时，索引为 `numpy.linspace(0, N-1, n).round().astype(int)`。仅在最后一帧缺失、实际解码恰少一帧时，用前一帧替代，并同时记录请求与实际索引；其他缺帧/计数异常报错。
3. 时间戳为 `实际帧索引 / fps`，时长为 `声明帧数 / fps`。这是基于容器 FPS 的等时间间隔假设；可变帧率视频应先统一帧率，否则此协议不保证真实 PTS 时间。
4. system 固定为 `You are a careful visual observer. Separate evidence from hypotheses. Respond with JSON only.`。任务使用原 `schema_fields_only` 协议，即五层字段列表加 `Duration: ... seconds.`；不向模型提供文件名、视频 ID、参考答案或会话历史。
5. 五层为 geometry、physical、process、temporal、intent_quality，共 34 个字段，另有 evidence_notes。未测量量保持未知，候选工艺与观察分开；这些是任务要求，并非后处理强制填写的答案。
6. 本地模型默认 `max_tokens=3072`、seed=1、贪心生成；API 可显式选择推理强度及更大预算。不同预算要作为不同实验条件保留，不能宣称完全等预算。
7. 全部输入和模型适配后的输入审计先固定，再开始该次运行的推理。保留完整原文；解析只允许移除完整 `<think>...</think>` 和最外层代码围栏，不补 JSON、不补字段、不把 null 改成 false/0。

**统一时序输入不等于统一模型内部处理。**以下差异会进入 `input_audit.json` 和 `protocol.json`：

| 后端 | 模型实际接收的媒体 | 空间预处理与文本模板 |
|---|---|---|
| `codeasworlds` | 按时间排列的视频张量及 metadata | factor 28、bicubic antialias、每帧最多 262144 像素；官方 `qwen3_5_no_think.jinja` |
| `cosmos3` | 视频序列 | 本地官方处理器，禁止再次抽帧；视频总预算按帧数乘以每帧 65536～262144 像素；原 chat template |
| `robobrain` | 视频序列 | 同上，使用该模型自己的处理器和 chat template |
| `responses` | 独立 PNG 图片，逐张附时间戳 | 用指定参考处理器的实际视频张量反归一化，clip/round 为 uint8；API 内部图像处理不可见 |

拼图仅供核查，从不作为模型输入。API 的原始 PNG 无文字、框或掩膜。Responses 请求没有 tools、previous_response_id 或额外历史；服务端是否自行添加工具、实际路由到什么模型，只能以原响应提供的有限证据核查。

## 环境

Python **3.11 或更新版本**。以下为分开的依赖配置，不要把 vLLM 和 Transformers 两套 Torch 环境混装。版本取自整理时使用的服务器环境；GPU 驱动/CUDA 兼容性仍由部署环境负责。

```bash
# CPU：抽帧、协议测试、输入检查。API 发送本身只需标准库。
python -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements/base.txt

# Cosmos / RoBoBrain 推理，以及 API 参考像素导出：独立环境
python -m pip install -r requirements/transformers.txt

# Code-as-World：另建环境后安装
python -m pip install -r requirements/codeasworlds.txt
```

Windows 激活命令为 `.venv-cpu\Scripts\Activate.ps1`。API 参考像素导出需要 Torch 和本地处理器文件，但不会加载语言模型；`send_api.py` 不需要 Torch、CUDA 或模型文件。

Code-as-World 另需自行提供上游官方模板文件，通过 `--chat-template` 指向 `code_as_world/templates/qwen3_5_no_think.jinja`。不将上游仓库或模板源码复制进本仓库。

## 1. 冻结输入

在仓库根目录执行，按指定顺序传入待测视频；所有模型复用同一个输入目录：

```bash
python scripts/test/prepare_inputs.py \
  --videos /path/to/video1.avi /path/to/video2.avi /path/to/video3.avi /path/to/video4.avi \
  --frames 16 64 --output inputs/benchmark-v1
```

输出包含顺序编号、视频哈希、采样 PNG、每帧像素哈希、请求/实际索引、Prompt、时间戳和展示拼图。输入目录必须不存在，防止覆盖。任意数量的自有视频均可；短于要求帧数的视频会报错，不复制重复帧凑数。源路径仅留在审计，不进入模型消息。

## 2. 本地模型推理

先加 `--prepare-only` 可在 CPU 上检查处理器、模板和实际媒体输入，不加载语言模型。正式推理请使用新的输出目录，不覆盖准备实验或已有结果。

```bash
# Code-as-World 9B；4B 改 --caw-size 4b 和对应权重目录
python scripts/test/run.py \
  --backend codeasworlds --caw-size 9b \
  --weights weights/code_as_worlds-9b \
  --chat-template upstream/codeasworlds/code_as_world/templates/qwen3_5_no_think.jinja \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --tp 2 --context 16384 --max-tokens 3072 --output outputs/caw9b-v1

# Cosmos3：只调用 Reasoner 的文本生成
python scripts/test/run.py \
  --backend cosmos3 --weights weights/cosmos3-nano \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --max-tokens 3072 --output outputs/cosmos3-v1

# RoBoBrain2.5
python scripts/test/run.py \
  --backend robobrain --weights weights/robobrain25-8b-nv \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --max-tokens 3072 --output outputs/robobrain-v1
```

默认 BF16。Transformers 使用 `balanced` 设备映射、SDPA 和每张 GPU 20 GiB 内存预算，可用 `--gpu-memory-gib` 调整；vLLM 使用 `--tp`、`--context` 与 `--gpu-memory-utilization`。同一运行内逐条生成。显存不足会保留错误，框架不会自动缩小帧数、切换模型或改变预算。

## 3. API 测试

先在有参考处理器的机器上冻结完整 API 请求：

```bash
python scripts/test/run.py \
  --backend responses --weights weights/cosmos3-nano \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --base-url http://127.0.0.1:8317/v1 --model gpt-6-astra \
  --reasoning-effort max --max-tokens 32768 \
  --prepare-only --output outputs/astra-max-v1
```

可将生成的整个 `outputs/astra-max-v1` 和同一版本的框架复制到 API 所在电脑，再运行：

```bash
export WELDING_API_KEY='<在本机环境变量中配置>'
python scripts/test/send_api.py --output outputs/astra-max-v1
```

PowerShell 使用 `$env:WELDING_API_KEY = '<在本机配置>'`。密钥不写入请求文件、源码或 Git。`127.0.0.1` 始终是**运行发送脚本的电脑**；目标 URL 在准备阶段固定。跨主机发送不需要参考权重或原视频，仍会核对请求与框架源码哈希。

推理强度由目标服务支持范围决定；脚本不自动将不支持的强度降级。`32768` 是请求值，响应没有确认时不能认定代理实际执行了该上限。API 预算包含推理 token，与本地模型的生成预算不能简单等同。

每次尝试保存 `attempt_XXX.json`、原始 `attempt_XXX.sse` 以及完整响应。默认不自动重试。调用再次运行时跳过所有已有完整回答，包括 JSON 解析失败或被截断的回答；仅经人工检查通信失败后，显式执行：

```bash
python scripts/test/send_api.py --output outputs/astra-max-v1 --retry-transport-errors
```

失败尝试及部分文本保持原样。进程中断而没有完成记录也属于“是否完成未知”，必须显式确认重试；上游可能因此重复计算。不能将通信重试次数当成独立样本，也不能通过重跑低质量答案选优。

## 4. 核验与比较

```bash
python scripts/test/validate.py --inputs inputs/benchmark-v1 --output outputs/cosmos3-v1
python scripts/test/compare_inputs.py outputs/cosmos3-v1 outputs/robobrain-v1
python -m unittest discover -s tests -v
```

`validate.py` 检查固定请求/图像/提示词哈希、输入与输出对应关系、原始回答到 JSON 的一致性，并报告解析失败、缺字段、null 字段与错误层级类型。结构错误保留为模型结果，核验不会把它们修好；provenance 通过不等于语义正确。

`compare_inputs.py` 要求相同视频与帧数条件，核对原始像素、索引、时间戳及任务 Prompt，并分别列出预处理像素是否一致、预算与后端配置。不同帧数请作为独立条件比较，不应要求其输入哈希相同。

输出文件：

- `protocol.json`：调用前冻结的配置、源码哈希、依赖版本、模型配置文件哈希和全部输入文件哈希。
- `run.json`：运行状态、原始回答、严格解析结果、耗时及通信失败历史。
- `<case_id>/input_audit.json`：公共时序信息、媒体预处理、张量/图像哈希和提示词。
- `<case_id>/raw_response.json` 或 API 响应文件：可追溯的原始模型输出。
- `<case_id>/input_montage.jpg`：实际预处理输入的展示拼图。

模型身份记录的是本地路径和配置哈希，**本框架未逐字节校验完整权重**；有严格版本要求时，应在外部完成权重校验并保留相应记录。生成参数固定也不保证跨 GPU、库版本或 API 服务完全确定性。

## 扩展与提交边界

新增模型只在适配器中实现空间预处理、模型加载和生成，复用 `sampling.py`、`prompt.py` 与输入 manifest。修改 Prompt、抽帧、像素处理或预算必须创建新输出目录，保留旧协议。不要为了适配模型而读取其他模型答案或回填语义值。

新增内容提交前检查 `git diff --cached --stat` 和 `git ls-files`。`.gitignore` 排除下载脚本、文档目录、权重、媒体、实验数据、报告及凭据；请勿使用 `git add -f` 绕过这些规则。这里唯一的项目说明文件是本 README，测试不携带已有焊接视频或实验结论。
