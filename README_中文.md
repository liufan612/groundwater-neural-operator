# 算子论文的数据生成与训练代码

已于 2026 年 10 月 22 日对照 `Manuscript_20261023.docx` 与 `Supporting_Information_20261023_lf.docx` 核对。本目录用于整理上传 GitHub 的源代码。

## 包含内容

1. 100 个基准情景的随机渗透系数生成、MODFLOW 6 模型构建、垂向细分、模拟及学习网格导出。
2. 四组源项情景 S00/S10/S01/S11，以及补充图 S2 的间歇释放情景生成。
3. U-Net、FNO、AFNO-style、UNO、DeepONet 五种主干。
4. 冻结水头条件化的两阶段训练、双分支训练、00/C/H/HC 物理配置及硬非负浓度约束。
5. 固定输入配置、依赖、运行说明和必要的代码检查。

`configs/source_templates.zip` 仅保存必要的几何、边界、初始条件与应力输入，不包含模拟输出。去除重复输入后的体积约 0.78 MB。所有模拟轨迹、处理后训练数据、模型权重和中间结果均由代码运行后生成，未放入交付包。

## 使用顺序

详见英文 [README](README.md) 中可直接执行的命令：

1. 安装 Python 3.11、CUDA 版 PyTorch 和项目依赖；配置 MODFLOW 6.8.0 的 `MF6_EXE`。
2. 运行 `python model/generate_dataset.py` 生成基准数据。
3. 运行 `python model/train.py head --model afno` 与 `python model/train.py guided-head`。
4. 按需运行 `train.py two-stage`、`train.py dual` 和另外四种主干。
5. 运行 `python model/generate_sources.py` 生成四组源项数据；这些是开发诊断情景，不参与重新训练。

## 上传 GitHub

将**本目录的内容**作为仓库根目录上传。`.gitignore` 已排除运行后生成的数据、缓存、日志和权重。仓库说明使用英文 README，中文文件作为操作说明。公开发布前由作者确定代码许可证；当前未擅自添加 MIT、GPL 等授权。

## 已知限制

旧笔记本不能确认可从记录的种子重建完全相同的原始强迫设置，因此保留了固定输入配置。新入口使用论文最终选定的超参数，不重新进行历史学习率搜索。训练仍需要 CUDA。完整数据重新模拟和正式长程训练未在本次整理中重跑；具体检查范围见 [复现说明](docs/REPRODUCTION.md)。

训练命令输出验证集指标；正式保留测试集评估、源项预测、SHAP 与绘图入口未包含在本包中。历史物理权重与残差尺度选择采用最终确定值。论文方法、模型参数量与源项设计的逐项对照见 [一致性审查记录](docs/MANUSCRIPT_CONSISTENCY_中文.md)。
