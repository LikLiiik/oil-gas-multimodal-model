# 油气地球物理大模型 - 多模态特征融合

> **竞赛题目**：XH-202604 "透视"地下油气藏——油气地球物理大模型的多模态特征融合  
> **发榜单位**：中国石油勘探开发研究院  
> **竞赛平台**：第十九届"挑战杯"全国大学生课外学术科技作品竞赛

---

## 项目概述

本项目构建了一个能够融合 **3D 地震** 与 **1D 测井曲线** 的油气地球物理大模型，支持 Volve、RMOTC（Teapot Dome）、Penobscot 三个公开工区的真实数据预训练，并实现多模态统一表征与下游任务迁移。

### 核心挑战

- **地震数据**：3D 叠后地震体（空间-视觉模态）
- **测井数据**：沿井筒的多曲线 1D 序列（时序-物理模态）
- **跨工区训练**：不同工区坐标系、采样率、曲线命名需统一对齐
- **目标**：跨越模态与物理差异，实现统一特征表征与有利目标识别

---

## 模型架构

```
Seismic (B,1,D,H,W) ──> NCS Seismic Encoder ──> seismic_feat ──┐
                     (NCS-v1 2.5D/3D ViT)                          │
                                                                    ├──> CrossModalFusion ──> Pretrain / Task Heads
Well Log (B,C,L) ────> WLFM Well Log Encoder ──> well_feat ────────┘   (Coarse + Cross-Attn
                     (CNN + VQ + Transformer)                           + Gated Fusion)
```

也支持 legacy 骨干：`resnet3d` / `swin3d`（地震）与 `cnn_transformer`（测井），见 `config/config.yaml`。

### 关键组件

| 组件 | 技术方案 | 功能 |
|------|---------|------|
| **地震编码器** | NCS-v1 2.5D/3D ViT（默认） | 预训练 3D 地震表征 |
| **测井编码器** | WLFM（Well Log Foundation Model） | 测井序列建模 + 地质词汇 VQ |
| **跨模态融合** | Coarse Fusion + Bi-Cross-Attention + Gate | 多层次模态融合 |
| **预训练** | MSM + MWM + CMCL + SWM 四任务分阶段 | 自监督跨模态表征学习 |
| **下游任务** | 断层检测 / 储层预测 / 岩性分类 | 多任务预测头 |

---

## 项目结构

```
oil-gas-multimodal-model/
├── config/
│   ├── config.yaml                 # 主配置（骨干、融合、预训练参数）
│   └── model_config.py
├── data/
│   ├── prepare_volve_data.py       # Volve 元数据与井轨迹预处理
│   ├── prepare_rmotc_data.py       # RMOTC 预处理
│   ├── prepare_penobscot_data.py   # Penobscot 预处理
│   ├── volve_dataset.py            # Volve 单工区 Dataset
│   ├── field_dataset.py            # 通用工区 Dataset
│   ├── multimodal_dataset.py       # Volve + RMOTC + Penobscot 联合 Dataset
│   ├── well_seismic_tie.py         # 井-震物理对齐
│   ├── cbvs_io.py                  # Penobscot CBVS 地震读取
│   ├── synthetic_data.py           # 合成数据（调试/基线）
│   └── transforms.py
├── models/
│   ├── ncs_seismic_encoder.py      # NCS 地震编码器
│   ├── wlfm_well_log_encoder.py    # WLFM 测井编码器
│   ├── fusion_module.py
│   ├── prediction_heads.py
│   └── oil_gas_model.py
├── pretraining/                    # MSM / MWM / CMCL / SWM
├── training/
├── scripts/
│   ├── prepare_data.py             # 准备 Volve
│   ├── prepare_external_datasets.py# 准备 RMOTC / Penobscot
│   ├── download_volve_deviations.py
│   ├── train_pretrain_multi.py     # 三工区联合分阶段预训练（推荐）
│   ├── train_pretrain_volve.py     # Volve 单工区预训练
│   ├── train_pretrain.py           # 合成数据预训练（基线）
│   ├── train_finetune_volve.py
│   ├── train_finetune.py
│   ├── inference.py
│   └── evaluate.py
├── releases/
│   └── pretrain_multi/
│       └── manifest.json           # 预训练 release 元数据（权重不入库）
├── seismic/                        # Volve 3D 地震（本地，不入库）
├── checkpoints/                    # 训练 checkpoint（本地，不入库）
├── logs/                           # 训练日志（本地，不入库）
├── tests/
├── requirements.txt
└── README.md
```

---

## 数据与权重说明

**仓库中不包含大数据和模型权重**，克隆后需自行准备本地数据。以下目录/文件已在 `.gitignore` 中忽略：

| 类型 | 本地路径 | 说明 |
|------|---------|------|
| Volve 测井 | `data/Volve_Well_logs_pr_WELL/` | Equinor 公开数据集 |
| Volve 钻井 | `data/Volve_WITSML_Realtime_drilling_data/` | WITSML 实时数据 |
| Volve 技术文档 | `data/Volve_Well_technical_data/` | 井身结构等 |
| Volve 地震 | `seismic/*.segy` | 3D 叠后地震体 |
| RMOTC | `data/rmotc/` | 含 `rmotc.tar` 解压后数据 |
| Penobscot | `data/penobscot/` | 含 `Penobscot.zip` 解压后数据 |
| 预处理结果 | `data/prepared/`、`data/rmotc/prepared/`、`data/penobscot/prepared/` | 运行 prepare 脚本生成 |
| 训练权重 | `checkpoints/`、`releases/**/*.pt` | 本地保存，不推送 Git |

`releases/pretrain_multi/manifest.json` 仅记录 release 元信息（训练配置、样本数、checkpoint 文件名），实际 `.pt` 文件需本地训练获得或从团队共享存储获取。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将公开数据集下载到 `data/` 对应目录（见上表），然后运行预处理：

```bash
# Volve：扫描测井、对齐地震、生成 data/prepared/ 元数据
python scripts/prepare_data.py

# （可选）从 NPD/SODIR 下载井斜数据
python scripts/download_volve_deviations.py

# RMOTC + Penobscot：解压并生成 prepared 元数据
python scripts/prepare_external_datasets.py

# 仅处理单个工区
python scripts/prepare_external_datasets.py --dataset rmotc
python scripts/prepare_external_datasets.py --dataset penobscot
```

### 3. 运行测试

```bash
python tests/test_model.py
```

### 4. 多工区分阶段预训练（推荐）

Stage 1：MSM + MWM（单模态预训练）  
Stage 2：CMCL + SWM（跨模态对齐）

```bash
python scripts/train_pretrain_multi.py \
    --fields volve rmotc penobscot \
    --stage1_epochs 50 \
    --stage2_epochs 50 \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints/pretrain_multi
```

常用参数：

- `--fields volve rmotc penobscot`：选择参与训练的工区
- `--seismic_backbone ncs` / `--well_backbone wlfm`：编码器骨干
- `--embed_dim 192`：模型宽度
- `--resume` / `--stage2_from`：断点续训

### 5. Volve 单工区预训练

```bash
python scripts/train_pretrain_volve.py \
    --checkpoint_dir ./checkpoints/pretrain_staged
```

### 6. 合成数据基线（无需真实数据）

```bash
python scripts/train_pretrain.py --epochs 100 --batch_size 8 --use_synthetic
```

### 7. 微调与推理

```bash
# Volve 微调
python scripts/train_finetune_volve.py \
    --pretrained checkpoints/pretrain_multi/best_stage2.pt

# 通用下游任务微调
python scripts/train_finetune.py \
    --task fault_detection \
    --pretrained checkpoints/pretrain_multi/best_stage2.pt

# 推理
python scripts/inference.py \
    --checkpoint checkpoints/pretrain_multi/best_stage2.pt \
    --task fault_detection \
    --input synthetic
```

---

## 三阶段训练策略

1. **阶段 1 — 单模态预训练**：掩码地震建模 (MSM) + 掩码测井建模 (MWM)
   - 学习各模态内部表征

2. **阶段 2 — 跨模态对齐**：跨模态对比学习 (CMCL) + 地震-测井匹配 (SWM)
   - 建立井-震语义对应

3. **阶段 3 — 任务微调**：在标注数据上微调下游任务头
   - 断层检测、储层预测、岩性分类

多工区训练时，`CombinedMultimodalDataset` 仅使用 **verified geometry**（有真实井轨迹、可精确井-震对齐的样本），避免估算坐标引入噪声。

---

## 技术创新点

1. **NCS 预训练地震编码器** — 基于 NCS-v1 的 2.5D/3D 地震 ViT 表征
2. **WLFM 测井基础模型** — VQ 地质词汇 + Transformer 序列建模
3. **三工区联合预训练** — Volve + RMOTC + Penobscot 统一数据接口与归一化
4. **物理约束井-震对齐** — `well_seismic_tie` 沿井轨迹精确采样地震振幅
5. **四策略层次化融合** — Token / 实例 / 体素 / 特征多级融合
6. **分阶段渐进训练** — 模态内理解 → 模态间关联 → 下游迁移

---

## 评分标准对应

| 评分维度 | 权重 | 本项目覆盖 |
|---------|------|-----------|
| 技术创新性 | 30% | NCS + WLFM、物理井-震对齐、多工区联合预训练 |
| 落地可行性 | 40% | 真实公开数据 pipeline、完整训练/微调脚本 |
| 材料完整性 | 30% | 可运行代码、数据准备文档、release manifest |

---

## 依赖环境

- Python >= 3.9
- PyTorch >= 2.0
- CUDA（推荐，3D 卷积与 ViT 训练）
- 详见 `requirements.txt`
