# 糖尿病智能诊断与分型系统 —— 多级多任务学习模型（代码与数据）

本仓库为论文附件，包含模型代码、训练好的权重、数据集说明、运行输出结果以及参考文献。

本仓库实现了一种**面向糖尿病的智能诊断与分型框架**，采用**多级（two-level）、多任务（multi-task）** 的机器学习 / 深度学习方案：

- **第一级**：在多个糖尿病诊断数据集上，同时训练 **多任务模型**（7 个数据集联合训练）与 **单任务模型**（3 个数据集独立训练）；
- **第二级**：在 `diabetes_dataset00.csv` 上训练 **类型模型**，完成糖尿病细分类型（12 类）诊断与治疗建议（4 类）预测。

模型以 PyTorch 实现，并结合 `imbalanced-learn` 处理医学数据中常见的类别不平衡问题。

---

## 目录

- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [数据集说明](#数据集说明)
- [模型架构](#模型架构)
- [环境依赖与安装](#环境依赖与安装)
- [快速开始](#快速开始)
- [性能结果](#性能结果)
- [模型权重文件](#模型权重文件)
- [鲁棒性与统计显著性](#鲁棒性与统计显著性)
- [参考文献](#参考文献)
- [注意事项与已知限制](#注意事项与已知限制)
- [许可证](#许可证)

---

## 核心特性

- **多级多任务学习**：第一级诊断 + 第二级分型，分阶段建模。
- **多任务 + 单任务双路**：既做 7 数据集联合多任务训练，也保留 3 个数据集的单任务模型。
- **糖尿病细分类型识别**：第二级类型模型支持 12 类糖尿病分型及 4 类治疗建议预测。
- **类别不平衡处理**：使用 `SMOTE`、`RandomOverSampler`、`RandomUnderSampler` 缓解正负样本失衡。
- **完整的评估体系**：报告 AUC（含 95% 置信区间）、灵敏度（Se）、特异度（Sp）、精确率（Precision）、F1、准确率（Accuracy）。
- **鲁棒性分析**：在注入噪声 / 异常值的条件下评估 F1 下降幅度。
- **统计显著性检验**：对第二级任务采用 Bootstrap 重采样，给出 F1 的 p 值与 95% 置信区间。

---

## 项目结构

```
附件1/
├── read me.txt                         # 作者原始说明（本 README 的基础）
├── 代码/
│   ├── 总模型代码.txt                   # 主入口：整合多任务/单任务/类型/测试/消融
│   ├── 单任务模型计算AUC代码.txt         # 第一级 3 个单任务数据集的 AUC 计算
│   ├── 多任务模型计算AUC代码.txt         # 第一级 7 个多任务数据集的 AUC 计算
│   ├── 类型模型计算AUC代码.txt           # 第二级 diabetes_dataset00.csv 的 AUC 计算
│   ├── 鲁棒性分析代码.txt               # 第一级 10 个数据集的鲁棒性分析（F1 为主）
│   ├── 统计显著性检验代码.txt           # 第二级 Bootstrap 显著性检验
│   ├── AUC结果.txt                      # 各模型 AUC / Se / Sp 等计算结果（训练日志）
│   ├── 鲁棒性分析代码结果.txt
│   ├── robustness_analysis.csv          # 鲁棒性分析输出的结构化结果
│   └── 固化文件输出/                    # 已训练好的模型权重（见下文）
└── 参考文献/                           # 25 篇糖尿病 / 机器学习相关 PDF
```

> 说明：代码以 `.txt` 形式存档（从 IDE / Notebook 导出）。运行前请将其另存为 `.py` 文件。

---

## 技术栈

| 类别 | 库 / 工具 |
|------|-----------|
| 编程语言 | Python 3 |
| 深度学习 | `torch`、`torch.nn`、`DataLoader`、`optim`、学习率调度（`ReduceLROnPlateau` / `CosineAnnealingLR`） |
| 传统机器学习 | `scikit-learn`（`StandardScaler`、`train_test_split`、评估指标） |
| 不平衡学习 | `imbalanced-learn`（`SMOTE`、`RandomOverSampler`、`RandomUnderSampler`） |
| 数据处理 | `pandas`、`numpy` |
| 可视化 | `matplotlib`、`seaborn` |
| 科学计算 | `scipy`、`joblib`（保存 / 加载 scaler） |

---

## 数据集说明

代码中使用 **13 个公开糖尿病相关 CSV 数据集**，原始路径硬编码在 `/personal/` 目录下，使用前请改为本地路径：

| 编号 | 文件名 | 用途 |
|------|--------|------|
| 1 | `Diabetes Simple Diagnosis.csv` | 多任务 |
| 2 | `diabetes.csv` | 多任务 |
| 3 | `diabetes_012_health_indicators_BRFSS2015.csv` | 多任务 |
| 4 | `diabetes_data.csv` | 单任务 |
| 5 | `diabetes_data_upload.csv` | 单任务 |
| 6 | `diabetes_dataset.csv` | 单任务 |
| 7 | `Diabetes_Dataset_With_18_Features.csv` | 多任务 |
| 8 | `diabetes_dataset00.csv` | **类型模型（分类/分型）** |
| 9 | `Diabetes_prediction.csv` | 多任务 |
| 10 | `diabetes_prediction_dataset.csv` | 多任务 |
| 11 | `Gestational Diabetes.csv` | 多任务 |
| 12 | `Gestational Diabetic Dat Set.csv` | 多任务 |
| 13 | `Healthcare-Diabetes.csv` | 多任务 |

> 其中 `diabetes_dataset00.csv` 为**分类（分型）模型**数据集，其余为**诊断模型**数据集。

---

## 模型架构

采用**两级框架**：

1. **第一级 —— 诊断**
   - **多任务模型**：在 7 个数据集（编号 1、2、3、7、10、12、13）上联合训练，共享表示、分别输出各数据集的诊断任务。
   - **单任务模型**：在 3 个数据集（编号 4、6、9）上独立训练，作为对照 / 补充。
2. **第二级 —— 分型**
   - **类型模型**：在 `diabetes_dataset00.csv` 上训练，输出 **12 类糖尿病分型** 与 **4 类治疗建议**。

**输入特征**（8 个初始筛选特征）：

| 类型 | 特征 |
|------|------|
| 数值型 | `Age`、`BMI`、`Glucose`、`BloodPressure`、`HbA1c_level` |
| 二值 / 类别型 | `Gender`、`Pregnancy`、`Smoking` |

> 代码内置了针对各数据集的列名映射（`feature_column_mapping`）与个性化预处理（如单位换算、缺失值替换为 `NaN`、剔除异常值 `9999999` 等）。

**糖尿病类型标签（12 类）**：
`CFRD`、`LADA`、`MODY`、`NDM`、`Prediabetic`、`Secondary Diabetes`、`Steroid-Induced Diabetes`、`Type 1 Diabetes`、`Type 2 Diabetes`、`Type 3c Diabetes`、`Unknown`、`Wolfram Syndrome`

**治疗建议标签（4 类）**：
`Insulin-Dependent`、`No Special Treatment`、`Oral Medications`、`Unknown Treatment`

---

## 环境依赖与安装

```bash
pip install torch scikit-learn imbalanced-learn pandas numpy matplotlib seaborn scipy joblib
```

> 建议使用虚拟环境（如 `conda` 或 `venv`）。PyTorch 需根据自身 CUDA 版本选择对应安装命令（CPU 版本亦可运行，代码会自动回退到 `cpu`）。

---

## 快速开始

1. **准备数据集**：将 13 个 CSV 放到本地目录，并修改 `总模型代码.txt` 等文件中 `pd.read_csv('/personal/...')` 的路径为本机路径。
2. **（可选）跳过训练直接预测**：`固化文件输出/` 中已提供训练好的 `.pth` 与 `.pkl` 文件，将其放到本地，并修改代码中加载固化文件的路径即可直接做病患预测。
3. **计算评估指标**：运行 `单任务模型计算AUC代码.txt` / `多任务模型计算AUC代码.txt` / `类型模型计算AUC代码.txt` 以复现 AUC / Se / Sp 等指标（总模型代码本身只输出 F1、精确率等，不含 AUC / Se / Sp 计算）。
4. **鲁棒性与显著性**：分别运行 `鲁棒性分析代码.txt` 与 `统计显著性检验代码.txt`。

> 代码中的模型固化输出路径（`固化文件输出/`）也请按本机位置调整。

---

## 性能结果

### 多任务模型（7 个诊断数据集，最佳 epoch 快照）

| 数据集 | 任务 | AUC | 95% CI | 灵敏度 Se | 特异度 Sp | F1 |
|--------|------|-----|--------|-----------|-----------|-----|
| 1 | Task 0 | 0.963 | 0.958–0.968 | 0.928 | 0.843 | 0.885 |
| 2 | Task 1 | 0.882 | 0.822–0.934 | 0.788 | 0.742 | 0.765 |
| 3 | Task 2 | 0.769 | 0.762–0.776 | 0.708 | 0.693 | 0.701 |
| 7 | Task 3 | 0.918 | 0.896–0.938 | 0.847 | 0.847 | 0.850 |
| 10 | Task 4 | 0.970 | 0.966–0.975 | 0.921 | 0.868 | 0.894 |
| 12 | Task 5 | 0.993 | 0.986–0.997 | 0.970 | 0.972 | 0.971 |
| 13 | Task 6 | 0.839 | 0.811–0.869 | 0.824 | 0.695 | 0.758 |

> 完整逐 epoch 训练日志见 `代码/AUC结果.txt`。单任务模型与类型模型的详细指标亦记录于该文件。

### 鲁棒性分析（`robustness_analysis.csv`）

以 F1 为主要指标，在注入噪声 / 异常值条件下评估第一级 10 个数据集的性能下降：

| 类型 | 数据集 | 样本数 | 基线 F1 | 噪声下降 % | 异常值下降 % |
|------|--------|--------|---------|-----------|-------------|
| MTL | 1 | 88,380 | 0.945 | 8.0 | 9.2 |
| MTL | 2 | 390 | 0.973 | 20.3 | 21.6 |
| MTL | 3 | 253,680 | 0.788 | 1.5 | 1.5 |
| MTL | 7 | 4,303 | 0.940 | 39.6 | 39.3 |
| MTL | 10 | 100,000 | 0.960 | 8.6 | 8.2 |
| MTL | 12 | 3,525 | 0.954 | 51.1 | 51.1 |
| MTL | 13 | 2,768 | 0.790 | 34.4 | 33.8 |
| Single | 4 | 1,879 | 0.829 | 45.6 | 45.6 |
| Single | 6 | 10,000 | 0.497 | 30.3 | 30.3 |
| Single | 9 | 1,000 | 0.568 | 74.9 | 73.0 |

---

## 模型权重文件

已固化输出的模型与 scaler 位于 `代码/固化文件输出/`：

| 文件 | 说明 |
|------|------|
| `best_single_task_3.pth` / `best_single_task_5.pth` / `best_single_task_8.pth` | 单任务最佳模型权重（任务 3 / 5 / 8） |
| `diabetes_multi_diagnosis_model.pth` | 多任务诊断模型权重 |
| `diabetes_type_model_final.pth` | 类型（分型）模型最终权重（约 13.9 MB） |
| `multi_scaler_0.pkl` … `multi_scaler_6.pkl` | 多任务 7 个数据集的标准化 scaler（7 个） |
| `single_task_scaler_3.pkl` / `single_task_scaler_5.pkl` / `single_task_scaler_8.pkl` | 单任务 3 个数据集的 scaler |
| `type_scaler.pkl` | 类型模型 scaler |

---

## 鲁棒性与统计显著性

- **鲁棒性分析**：针对第一级 10 个数据集，以 F1 为主要评估指标，考察在注入噪声与异常值后的性能保持能力（结果见 `robustness_analysis.csv`）。
- **统计显著性检验**：对第二级任务（`diabetes_dataset00.csv`）的训练结果，采用 **Bootstrap 重采样** 方法，针对每个训练周期的 F1 分数计算 **p 值** 与 **95% 置信区间**。

---

## 参考文献

`参考文献/` 目录包含 **25 篇** 糖尿病流行病学、机器学习 / 深度学习在糖尿病中应用的 PDF 文献（如 ESC 糖尿病指南、UK Biobank 脑影像、多任务学习、BERT、域适应等），为模型设计与对比方法提供理论支撑。

---

## 注意事项与已知限制

1. **路径硬编码**：数据集与固化文件的路径为作者本机路径（`/personal/` 与 `固化文件输出/`），**运行前必须改为本机路径**，否则会报错。
2. **代码格式**：代码以 `.txt` 存档，请先重命名为 `.py` 再运行。
3. **大文件**：本仓库含较大的模型权重与 PDF 文献，若托管到 GitHub 建议以 **Release 附件** 形式提供，或使用 Git LFS，避免污染 git 历史。
4. **复现性**：代码固定随机种子（`torch.manual_seed(42)`、`np.random.seed(42)`），但 AUC 结果文件含逐 epoch 日志，多次运行的指标会有轻微波动。

---


在此声明MIT仅用于学术等非商业用途。
