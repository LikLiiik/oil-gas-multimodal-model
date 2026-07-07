# 油气地球物理大模型 - 多模态特征融合

> **竞赛题目**：XH-202604 "透视"地下油气藏——油气地球物理大模型的多模态特征融合  
> **发榜单位**：中国石油勘探开发研究院  
> **竞赛平台**：第十九届"挑战杯"全国大学生课外学术科技作品竞赛

---

## 项目概述

本项目构建了一个能够融合**3D地震图像**与**1D测井曲线**两种模态数据的油气地球物理大模型，实现多模态油气地球物理数据的统一表征和有利地质目标智能识别。

### 核心挑战

- **地震数据**：3D叠后地震图像体（空间-视觉模态）
- **测井数据**：沿井筒的1D多曲线序列（时序-物理模态）
- **目标**：跨越模态、空间分辨率和物理特性的差异，实现统一特征表征

---

## 模型架构

```
Seismic (B,1,D,H,W) ──> SeismicEncoder3D ──> seismic_feat ──┐
                     (CNN Stem + 3D Swin Transformer)         │
                                                              ├──> CrossModalFusion ──> Task Heads
Well Log (B,C,L) ────> WellLogEncoder1D ──> well_feat ──────┘   (Coarse + Cross-Attn    ├── Fault Detection
                     (CNN + RoPE Transformer                       + Gated Fusion)        ├── Reservoir Prediction
                       + Physics Encoding)                                                └── Lithology Classification
```

### 关键组件

| 组件 | 技术方案 | 功能 |
|------|---------|------|
| **地震编码器** | 3D CNN Stem + 3D Swin Transformer | 提取多尺度3D空间特征 |
| **测井编码器** | 多尺度1D CNN + RoPE Transformer + 物理约束 | 提取深度序列特征，融合岩石物理先验 |
| **跨模态融合** | Coarse Fusion + Bi-Cross-Attention + Adaptive Gate | 多层次模态融合 |
| **预训练** | MSM + MWM + CMCL + SWM 四任务联合 | 自监督跨模态表征学习 |
| **下游任务** | 断层检测 / 储层预测 / 岩性分类 | 多任务预测头 |

---

## 项目结构

```
油气大模型/
├── config/                     # 配置文件
│   ├── config.yaml             # 主配置文件
│   └── model_config.py         # Python配置类
├── data/                       # 数据模块
│   ├── synthetic_data.py       # 合成数据生成器
│   ├── dataset.py              # PyTorch数据集
│   └── transforms.py           # 数据增强
├── models/                     # 模型模块
│   ├── seismic_encoder.py      # 3D地震编码器
│   ├── well_log_encoder.py     # 测井编码器
│   ├── fusion_module.py        # 跨模态融合
│   ├── prediction_heads.py     # 下游任务头
│   └── oil_gas_model.py        # 完整模型封装
├── pretraining/                # 预训练模块
│   ├── msm_task.py             # 掩码地震建模
│   ├── mwm_task.py             # 掩码测井建模
│   ├── cmcl_task.py            # 跨模态对比学习
│   └── swm_task.py             # 地震-测井匹配
├── training/                   # 训练模块
│   ├── trainer.py              # 训练器
│   ├── losses.py               # 损失函数
│   └── metrics.py              # 评估指标
├── utils/                      # 工具函数
│   ├── helpers.py              # 辅助函数
│   └── visualization.py        # 可视化
├── scripts/                    # 执行脚本
│   ├── train_pretrain.py       # 预训练脚本
│   ├── train_finetune.py       # 微调脚本
│   ├── inference.py            # 推理脚本
│   └── evaluate.py             # 评估脚本
├── tests/
│   └── test_model.py           # 模型测试
├── requirements.txt
└── README.md
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 生成合成数据

```python
from data.synthetic_data import SyntheticDataGenerator

gen = SyntheticDataGenerator(seed=42)
sample = gen.generate_well_seismic_pair()

print(f"Seismic: {sample['seismic'].shape}")     # (1, 128, 256, 256)
print(f"Well log: {sample['well_log'].shape}")   # (512, 7)
```

### 运行测试

```bash
python tests/test_model.py
```

### 预训练

```bash
python scripts/train_pretrain.py --epochs 100 --batch_size 8 --use_synthetic
```

### 微调下游任务

```bash
# 断层检测
python scripts/train_finetune.py \
    --task fault_detection \
    --pretrained checkpoints/pretrain/pretrained_model_final.pt

# 储层预测
python scripts/train_finetune.py \
    --task reservoir_prediction \
    --pretrained checkpoints/pretrain/pretrained_model_final.pt

# 岩性分类
python scripts/train_finetune.py \
    --task lithology \
    --pretrained checkpoints/pretrain/pretrained_model_final.pt
```

### 推理

```bash
python scripts/inference.py \
    --checkpoint checkpoints/finetune/best_model_fault_detection.pt \
    --task fault_detection \
    --input synthetic
```

---

## 三阶段训练策略

1. **阶段1 - 单模态预训练**：掩码地震建模 (MSM) + 掩码测井建模 (MWM)
   - 利用大量无标注数据学习模态内表征
   
2. **阶段2 - 跨模态对齐**：跨模态对比学习 (CMCL) + 地震-测井匹配 (SWM)
   - 学习模态间的对应关系

3. **阶段3 - 任务微调**：在标注数据上微调下游任务
   - 断层检测、储层预测、岩性分类

---

## 技术创新点

1. **3D Swin Transformer for Seismic** - 首次将Swin Transformer扩展到3D地震数据
2. **物理约束测井编码器** - 显式建模岩石物理关系（DEN-POR, AC-POR, GR-泥质含量）
3. **四策略层次化融合** - Token/实例/体素/特征四级融合
4. **空间精确井震融合** - 使用grid_sample沿井轨迹精确采样
5. **三阶段渐进训练** - 逐层构建模态内理解到模态间关联

---

## 评分标准对应

| 评分维度 | 权重 | 本项目覆盖 |
|---------|------|-----------|
| 技术创新性 | 30% | 3D Swin ViT、物理约束编码、多级融合 |
| 落地可行性 | 40% | 完整训练框架、数据增强、多任务学习 |
| 材料完整性 | 30% | 可运行代码、合成数据、详细文档 |

---

## 依赖环境

- Python >= 3.9
- PyTorch >= 2.0
- CUDA (推荐) 用于3D卷积加速
- 详见 `requirements.txt`
