# Welding Semantics Reproduce

统一焊接视频的语义测试方法：**共享 Prompt、视频解码、抽帧索引、时间戳与结果字段**，通过模型适配器分别接入 Code-as-World、Cosmos3-Nano Reasoner、RoBoBrain2.5、RynnBrain1.1-9B、Qwen3.8-27B 和 Responses 兼容的多图 API。

## 一个项目、一个入口

在项目根目录统一使用 `python -m welding_bench <命令>`：

```bash
python -m welding_bench --help
python -m welding_bench list          # 列出所有实验
python -m welding_bench build-index   # 生成项目首页
python -m welding_bench browse        # http://127.0.0.1:8765/index.html
```

`run`、`prepare`、`send-api`、`validate`、`compare-inputs` 分别负责推理、冻结输入、发送已冻结请求、核验和比较。每个命令支持 `--help`。`scripts/test/` 是同一实现的兼容入口。

```text
项目根目录/
  README.md                使用说明（提交 Git）
  welding_bench/           统一 Prompt、抽帧、适配器、推理和目录管理（提交 Git）
  scripts/test/            测试命令（提交 Git）
  requirements/           各环境依赖（提交 Git）
  tests/                  框架回归测试（提交 Git）
  .venv*/                 本机环境
  scripts/download/       权重下载辅助脚本
  scripts/                本机页面构建、历史实验审计脚本
  upstream/               各上游项目及其 README
  weights/                模型权重
  video_samples/          输入视频
  inputs/                 可复用的冻结输入
  results/                一个测试一个目录
    模型名_年月日_时分秒/
      experiment.json     测试登记、状态、时间来源
      protocol.json       原始测试配置、输入与源码哈希
      run.json            原始结果；早期实验名为 raw_predictions.json
      case_*/             输入审计、原始响应和抽帧拼图
      exports/            本次实验的语义 JSON 导出（如有）
  web/                    浏览页面使用的汇总数据
  index.html              统一浏览入口
  results.html            语义、中文转述和抽帧拼图
  model_compare.html      模型与帧数对比
  sam3_segmentation.html  SAM3 分割及结论
  docs/                   本机实验文档
  reports/                汇总表格与分析
  logs/                   校验、迁移及运行日志
  config/*.local.json     本机目录迁移映射
```

**Git 仅提交框架、依赖清单、测试和本 README。** 权重、视频、环境、上游项目、结果、页面、实验文档和下载脚本由 `.gitignore` 排除；克隆仓库不会附带这些资产。

新推理不指定 `--output` 时，自动创建 `results/<模型名>_<北京时间 YYYYMMDD_HHMMSS>/`，并打印 `EXPERIMENT_DIRECTORY`。一次调用中的所有视频及 16/64 帧条件属于同一次测试，保存在该目录内。重复运行创建新目录；已有目录拒绝覆盖。`--prepare-only` 同样登记为独立测试，状态为 prepared。API 续跑须显式指定原目录。

历史实验保留原始回答、请求和协议；未知值、解析失败均不修补。旧记录缺少准确时间时，`experiment.json` 明确注明文件时间估计，不能当成精确推理起始时间。原始审计内的旧绝对路径保留为证据，展示层通过本机映射解析。

161 服务器统一入口为 `/home/chenliangkai/welding_semantics_reproduce`。权重、视频和其他大文件实际存储在 `/mnt/chenliangkai/welding_semantics_reproduce/`，由项目根目录中的软链接访问。原 `/home/chenliangkai/welding_repro_20261022` 已合并移除。本地工作目录为 `repro/`，不再嵌套一个 `framework/` 仓库。

## 统一协议

1. 使用 OpenCV 从头**顺序解码**，不使用随机 seek 或各模型自带的再次抽帧。默认每段视频分别取 16 和 64 帧，也可指定其他合法偶数帧数。
2. 容器声明帧数为 `N` 时，索引为 `numpy.linspace(0, N-1, n).round().astype(int)`。仅在最后一帧缺失、实际解码恰少一帧时，用前一帧替代，并同时记录请求与实际索引；其他缺帧/计数异常报错。
3. 时间戳为 `实际帧索引 / fps`，时长为 `声明帧数 / fps`。这是基于容器 FPS 的等时间间隔假设；可变帧率视频应先统一帧率，否则此协议不保证真实 PTS 时间。
4. system 固定为 `You are a careful visual observer. Separate evidence from hypotheses. Respond with JSON only.`。任务使用原 `schema_fields_only` 协议，即五层字段列表加 `Duration: ... seconds.`；不向模型提供文件名、视频 ID、参考答案或会话历史。
5. 五层为 geometry、physical、process、temporal、intent_quality，共 34 个字段，另有 evidence_notes。未测量量保持未知，候选工艺与观察分开；这些是任务要求，并非后处理强制填写的答案。
6. 本地模型默认 `max_tokens=3072`、seed=1、贪心生成；API 可显式选择推理强度及更大预算。
7. 全部输入和模型适配后的输入审计先固定，再开始该次运行的推理。保留完整原文；解析只允许移除完整 `<think>...</think>` 和最外层代码围栏，不补 JSON、不补字段、不把 null 改成 false/0。

统一时序输入不等于统一模型内部处理：

| 后端 | 模型实际接收的媒体 | 空间预处理与文本模板 |
|---|---|---|
| `codeasworlds` | 按时间排列的视频张量及 metadata | factor 28、bicubic antialias、每帧最多 262144 像素；官方 `qwen3_5_no_think.jinja` |
| `cosmos3` | 视频序列 | 本地官方处理器，禁止再次抽帧；视频总预算按帧数乘以每帧 65536～262144 像素；原 chat template |
| `robobrain` | 视频序列 | 同上，使用该模型自己的处理器和 chat template |
| `rynnbrain` | 视频序列 | 官方 RynnBrain1.1-9B 处理器和模板；关闭思考与再次抽帧，核验模型时间戳；BF16、SDPA，全部权重驻留 GPU |
| `qwen38` | 带帧索引与 FPS 的视频序列，vLLM 推理 | 官方 Qwen3.8 模板及视频处理器；禁止再次抽帧，记录 vLLM 实际预处理张量及数据类型；默认关闭思考 |
| `responses` | 独立 PNG 图片，逐张附时间戳 | 用指定参考处理器的实际视频张量反归一化，clip/round 为 uint8；API 内部图像处理不可见 |

拼图仅供核查，从不作为模型输入。API 的原始 PNG 无文字、框或掩膜。Responses 请求没有 tools、previous_response_id 或额外历史；服务端是否自行添加工具、实际路由到什么模型，只能以原响应提供的有限证据核查。

## 环境

Python **3.11 或更新版本**。以下为分开的依赖配置，不要把 vLLM 和 Transformers 两套 Torch 环境混装。版本取自整理时使用的服务器环境；GPU 驱动/CUDA 兼容性仍由部署环境负责。

```bash
# CPU：抽帧、协议测试、输入检查。API 发送本身只需标准库。
python -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements/base.txt

# Cosmos / RoBoBrain / RynnBrain 推理，以及 API 参考像素导出：独立环境
python -m pip install -r requirements/transformers.txt

# Code-as-World：另建环境后安装
python -m pip install -r requirements/codeasworlds.txt
```

Windows 激活命令为 `.venv-cpu\Scripts\Activate.ps1`。API 参考像素导出需要 Torch 和本地处理器文件，但不会加载语言模型；`send_api.py` 不需要 Torch、CUDA 或模型文件。

Code-as-World 另需自行提供上游官方模板文件，通过 `--chat-template` 指向 `code_as_world/templates/qwen3_5_no_think.jinja`。不将上游仓库或模板源码复制进本仓库。

## 1. 冻结输入

在仓库根目录执行，按指定顺序传入待测视频；所有模型复用同一个输入目录：

```bash
python -m welding_bench prepare \
  --videos /path/to/video1.avi /path/to/video2.avi /path/to/video3.avi /path/to/video4.avi \
  --frames 16 64 --output inputs/benchmark-v1
```

输出包含顺序编号、视频哈希、采样 PNG、每帧像素哈希、请求/实际索引、Prompt、时间戳和展示拼图。输入目录必须不存在，防止覆盖。任意数量的自有视频均可；短于要求帧数的视频会报错，不复制重复帧凑数。源路径仅留在审计，不进入模型消息。

## 2. 本地模型推理

先加 `--prepare-only` 可在 CPU 上检查处理器、模板和实际媒体输入，不加载语言模型。正式推理请使用新的输出目录，不覆盖准备实验或已有结果。

```bash
# Code-as-World 9B；4B 改 --caw-size 4b 和对应权重目录
python -m welding_bench run \
  --backend codeasworlds --caw-size 9b \
  --weights weights/code_as_worlds-9b \
  --chat-template upstream/codeasworlds/code_as_world/templates/qwen3_5_no_think.jinja \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --tp 2 --context 16384 --max-tokens 3072

# Cosmos3：只调用 Reasoner 的文本生成
python -m welding_bench run \
  --backend cosmos3 --weights weights/cosmos3-nano \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --max-tokens 3072

# RoBoBrain2.5
python -m welding_bench run \
  --backend robobrain --weights weights/robobrain25-8b-nv \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --max-tokens 3072

# Qwen3.8-27B：使用上面的独立 vLLM 环境，官方 BF16 权重
python -m welding_bench run \
  --backend qwen38 --weights /path/to/qwen3.8-27b \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --tp 2 --context 16384 --gpu-memory-utilization 0.70 \
  --max-tokens 3072
```

Qwen3.8 默认关闭思考，沿用框架的 seed=1、贪心生成与 3072-token 预算；这是统一条件对照，不是官方推荐的随机采样配置。需要单独测试思考模式时，显式增加 `--qwen-thinking --qwen-reasoning-effort xhigh`（也支持 `medium`、`low`），同时自行设置合适的生成预算并使用新的输出目录。vLLM 使用自身生成默认值以避免模型 `generation_config.json` 隐式改变对照参数。

视频文件由 `prepare_inputs.py --videos ...` 直接读取后冻结；16/64 是实验条件，支持其他合法偶数帧数。Qwen 原生也支持视频文件/URL 接口，其解码与抽帧仍发生在预处理阶段。本框架固定抽帧结果后作为视频传入模型，保留帧顺序与时间信息，拼图只用于核验。

默认 BF16。Transformers 使用 `balanced` 设备映射、SDPA 和每张 GPU 20 GiB 内存预算，可用 `--gpu-memory-gib` 调整；vLLM 使用 `--tp`、`--context` 与 `--gpu-memory-utilization`。同一运行内逐条生成。显存不足会保留错误，框架不会自动缩小帧数、切换模型或改变预算。

## 3. API 测试

先在有参考处理器的机器上冻结完整 API 请求：

```bash
python -m welding_bench run \
  --backend responses --weights weights/cosmos3-nano \
  --inputs inputs/benchmark-v1 --frames 16 64 \
  --base-url http://127.0.0.1:8317/v1 --model gpt-6-astra \
  --reasoning-effort max --max-tokens 32768 \
  --prepare-only
```

记录命令打印的测试目录，可将该目录及同一版本的框架复制到 API 所在电脑。以下用 `<测试目录>` 代指实际的 `results/gpt-6-astra_年月日_时分秒`：

```bash
export WELDING_API_KEY='<在本机环境变量中配置>'
python -m welding_bench send-api --output "<测试目录>"
```

PowerShell 使用 `$env:WELDING_API_KEY = '<在本机配置>'`。密钥不写入请求文件、源码或 Git。`127.0.0.1` 始终是**运行发送脚本的电脑**；目标 URL 在准备阶段固定。跨主机发送不需要参考权重或原视频，仍会核对请求与框架源码哈希。

推理强度由目标服务支持范围决定；脚本不自动将不支持的强度降级。`32768` 是请求值，响应没有确认时不能认定代理实际执行了该上限。API 预算包含推理 token，与本地模型的生成预算不能简单等同。

## 4. 核验与比较

```bash
python -m welding_bench validate --inputs inputs/benchmark-v1 --output "<测试目录>"
python -m welding_bench compare-inputs "<Cosmos测试目录>" "<RoBoBrain测试目录>"
python -m unittest discover -s tests -v
```

`validate.py` 检查固定请求/图像/提示词哈希、输入与输出对应关系、原始回答到 JSON 的一致性，并报告解析失败、缺字段、null 字段与错误层级类型。结构错误保留为模型结果，核验不会把它们修好；provenance 通过不等于语义正确。

`compare_inputs.py` 要求相同视频与帧数条件，核对原始像素、索引、时间戳及任务 Prompt，并分别列出预处理像素是否一致、预算与后端配置。不同帧数请作为独立条件比较，不应要求其输入哈希相同。

每个测试目录中的输出文件：

- `experiment.json`：模型、创建时间、时间来源、实验状态与结果数。

- `protocol.json`：调用前冻结的配置、源码哈希、依赖版本、模型配置文件哈希和全部输入文件哈希。
- `run.json`：运行状态、原始回答、严格解析结果、耗时及通信失败历史。
- `<case_id>/input_audit.json`：公共时序信息、媒体预处理、张量/图像哈希和提示词。
- `<case_id>/raw_response.json` 或 API 响应文件：可追溯的原始模型输出。
- `<case_id>/input_montage.jpg`：实际预处理输入的展示拼图。
  

## 扩展与提交边界

新增模型只在适配器中实现空间预处理、模型加载和生成，复用 `sampling.py`、`prompt.py` 与输入 manifest。

## RynnBrain1.1-9B

使用官方本地权重，沿用冻结输入与统一 Prompt。`enable_thinking=False` 与官方 Transformers 快速示例一致；最大输出预算沿用焊接协议的 3072 tokens。示例使用一张 80 GB GPU，先确认卡空闲：

```bash
CUDA_VISIBLE_DEVICES=0 python -m welding_bench run \
  --backend rynnbrain --weights /path/to/RynnBrain1.1-9B \
  --inputs inputs/benchmark-v1 --frames 16 64 --gpu-memory-gib 60 \
  --max-tokens 3072
```

模型配置为 Qwen3.5，使用其自带 Qwen3-VL 视频处理器。本适配属于焊接语义任务迁移，并非官方机器人基准复现。运行记录含加载键检查、参数量、GPU 设备分配、实际视频时间戳和峰值显存；权重文件完整哈希仍由框架外的下载校验负责。
