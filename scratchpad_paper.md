# 论文复现与改进项目规划

## 背景和动机
用户选择复现论文 **2022.findings-naacl.58** ("A Generative Language Model for Few-shot Aspect-Based Sentiment Analysis")。
该论文提出了一种基于生成式语言模型（GPT-2）的方法来解决少样本（Few-shot）和全样本（Full-shot）的基于方面的情感分析（ABSA）任务。核心思想是将情感分类和方面提取任务转化为序列生成任务（如 `<|review|>...<|term|>...` -> `Positive`）。
本项目旨在：
1. 复现该论文的生成式模型方法。
2. 使用新的领域数据集 **Cryptocurrency Community Sentiment Analysis**（来自 `/root/autodl-tmp/NLP论文复现/dataset`）对模型进行微调和评估。
3. 验证该模型在加密货币社区评论数据上的有效性，并完成课程报告。

**重要约束**：
1. **环境**：模型微调必须在 `unsloth` 环境中进行（参考 `/root/.cursor/scratchpaad_unsloth.md`）。
2. **报告格式**：最终的 `课程报告.md` 内容必须是纯文本格式，不能包含任何 Markdown 标记（如 `##`, `**`, `-` 等），因为最终需要转换为 TXT。

## 关键挑战和分析
1. **环境配置**：
   - 需克隆 `https://github.com/salesforce/fewshot_absa.git`，注意配置 GitHub 镜像以确保在 autodl 环境下载成功。
   - 依赖库安装：需确认 `requirements.txt` 并解决潜在的版本冲突（特别是 PyTorch 和 Transformers 版本）。

2. **数据预处理与格式转换**：
   - **原始数据**：`reddit_sentiment_august2021.csv` 包含 `body` (文本) 和 `BERT-Sentiment` (情感标签) 等字段。
   - **目标格式**：论文代码通常需要特定的输入格式（如 `sentence` + `aspect` + `polarity`）。
   - **挑战**：该数据集主要为句子级/帖子级情感，可能缺乏显式的“方面（Aspect）”标注。
   - **解决方案**：
     - 方案A（默认）：将“Bitcoin”或“Crypto”作为统一的方面词（Target Aspect），将任务视为针对特定实体的细粒度情感分析。
     - 方案B：若文本中提及多种币种，使用简单的规则提取实体作为 Aspect。
     - 需将 CSV 转换为模型支持的文本序列格式（如 `<|review|> text <|endofreview|> <|term|> Bitcoin <|endofterm|>`）。

3. **模型微调**：
   - **环境依赖**：必须激活 `unsloth` 环境 (`conda activate /root/autodl-tmp/miniconda3/envs/unsloth`)。需参考历史项目 `/root/.cursor/scratchpaad_unsloth.md` 中的配置和脚本。
   - 需修改训练脚本以加载自定义数据集。
   - 调整超参数（如 Epochs, Batch Size, Learning Rate）以适应新数据集的大小和特性。

4. **评估指标**：
   - 论文主要使用 Accuracy 和 F1。需确保评估脚本能正确计算这些指标。

5. **报告撰写规范**：
   - **格式限制**：必须编写纯文本内容，通过缩进或空行分隔章节，严禁使用 Markdown 标题和列表符号。

## 高层任务拆分

### 阶段 1: 环境与数据准备 (Environment & Data)
1. **代码库准备**：
   - 克隆 `salesforce/fewshot_absa` 仓库。
   - 配置 Python 环境，安装依赖。
2. **数据分析与清洗**：
   - 分析 `reddit_sentiment_august2021.csv` 的标签分布（`BERT-Sentiment` 列）。
   - 清洗文本（去除 URL、特殊字符等）。
3. **数据格式转换**：
   - 编写脚本 `convert_data.py`，将 CSV 数据划分为 Train/Dev/Test 集（例如 8:1:1）。
   - 转换为模型所需的输入格式（参考 repo 中的 dataset 格式）。

### 阶段 2: 模型训练与微调 (Training & Fine-tuning)
1. **环境配置**：
   - 激活 Conda 环境：`/root/autodl-tmp/miniconda3/envs/unsloth`。
   - 参考 `/root/.cursor/scratchpaad_unsloth.md` 准备微调脚本（如需使用 Unsloth 加速或特定配置）。
2. **配置训练脚本**：
   - 创建新的运行脚本（如 `run_crypto.sh`），指向新的数据路径。
   - 设置微调参数（基于论文推荐配置）。
3. **执行微调**：
   - 运行训练，监控 Loss 和 Dev Set Accuracy。
   - 保存最佳模型检查点。

### 阶段 3: 评估与报告 (Evaluation & Reporting)
1. **模型测试**：
   - 在 Test 集上运行推理。
   - 计算 Accuracy, F1-score, Precision, Recall。
2. **结果分析**：
   - 分析模型在加密货币文本上的表现。
   - 检查典型错误案例。
3. **撰写报告**：
   - 填写 `课程报告.md`。
   - **注意**：确保内容为纯文本格式，无 Markdown 语法，以便转换为 TXT。

## 项目状态看板
- [x] **环境准备**：克隆代码，安装依赖 (Unsloth/Transformers 适配完成)。
- [x] **数据处理**：CSV -> 模型格式 (Train: 49.5k, Test: 5.5k)。
- [ ] **模型训练 (对照组)**：Unsloth Qwen3-8B (Text->Sentiment) 已暂停 (进度: ~43%)。
- [x] **基线测试**：Qwen3-8B Zero-shot (HF) 已完成 (Acc: 36.05%, Weighted F1: 27.67%)。
- [x] **数据重构 (核心)**：ABSA-fication (vLLM) 已完成 (生成 1016 条有效数据)。
- [x] **数据转换与清洗**：检测到 248 条 "no direct mention" 噪声并已清洗，最终训练集 914 条，测试集 102 条 (质量验证通过 ✅)。
- [x] **最终微调**：Generative ABSA Fine-tuning (train_absa.py) 已完成。
- [x] **模型合并**：LoRA weights merged to `merged_model` for vLLM inference.
- [x] **最终评估**：Generative ABSA Evaluation (evaluate_absa_vllm.py) 已完成。
  - Metrics: Aspect F1: 75.77%, Joint F1: 61.40%, Sentiment Acc: 81.14%.
- [ ] **报告撰写**：正在整合实验结果与截图资源。
  - ⚠️ **注意**: 当前磁盘上的 `crypto_absa_silver.json` (156K) 是**之前中断运行的残留文件**。当前全量任务正在内存中生成数据，**将在脚本运行结束时一次性写入**，请耐心等待。
- [ ] **数据转换**：待重构完成后，将 JSON 转为 Unsloth 训练格式 (脚本 `prepare_absa_dataset.py` 已就绪)。
- [ ] **最终微调**：Generative ABSA Fine-tuning (脚本 `train_absa.py` 已就绪)。
- [ ] **最终评估**：验证 Generative ABSA 有效性。
- [ ] **报告撰写**：完成纯文本课程报告。

## 执行者反馈或请求帮助
- **2025-12-29 (Parallel Execution)**:
  - **资源负载**: A800 显存充足，三任务并行顺畅。
  - **任务 1 (Data Construction)**: `construct_absa_data_vllm.py` 正在全量处理 5501 条测试数据，用于构建 Silver Standard ABSA 数据集。
  - **任务 2 (Baseline Eval)**: `evaluate_baseline_hf.py` 正在评估 Base 模型在简单情感分类上的 Zero-shot 性能。
  - **任务 3 (Training)**: `train_crypto.py` 继续运行，作为简单任务的对照组。
  - **下一步**: 待数据构建完成 -> 转换格式 -> 启动 Generative ABSA 微调。

## 经验教训
- 必须严格贴合原论文的核心任务 (ABSA)，简单的 Domain Adaptation 是不够的。
- 利用 LLM 的强大能力进行数据增强 (Pseudo-labeling) 是弥补数据集缺陷的有效手段。

## 经验教训
- Legacy 代码库在新环境中运行时，`transformers.modeling_utils` 中的许多工具类（如 `Conv1D`, `SequenceSummary`）已被移动或移除，手动适配成本过高且容易引入新 bug。
- **及时止损**: 遇到严重的环境兼容性阻力时，应果断切换到更现代、维护更好的框架（如 Unsloth）。
