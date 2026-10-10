# FLGenNet

**基于频率分解的轻量低照度图像增强网络。** 通过拉普拉斯金字塔分离高低频，在低分辨率分支增强亮度与颜色，再引导高频细节重建。

[论文 / SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6482333) · [原项目](https://github.com/OuyangYifan0324/FLGen-Net) · [数据链接](docs/ORIGINAL_README.md) · [Apache-2.0](LICENSE)

> **FLGenNet: Frequency Decomposition-based Lightweight Generalization Network for Low-Light Image Enhancement**
>
> Yifan Ouyang, Jiaqi Liu, Ashwaq Qasem

## 公开论文

**第一作者：Yifan Ouyang（Ouyang, Yifan）**。公开页面的作者顺序为 Yifan Ouyang、Jiaqi Liu、Ashwaq Qasem。

| 信息 | 内容 |
| --- | --- |
| 标题 | FLGenNet: Frequency Decomposition-based Lightweight Generalization Network for Low-Light Image Enhancement |
| 公开平台 | SSRN 预印本 |
| 公开日期 | 2026 年 3 月 28 日 |
| DOI | [10.2139/ssrn.6482333](https://doi.org/10.2139/ssrn.6482333) |
| 阅读与下载 | [SSRN 公开论文页面](https://ssrn.com/abstract=6482333) |

论文入口指向作者的公开版本；本仓库尚未托管 PDF 副本。代码许可证不自动适用于论文。

```bibtex
@misc{ouyang2026flgennet,
  title = {FLGenNet: Frequency Decomposition-based Lightweight Generalization Network for Low-Light Image Enhancement},
  author = {Ouyang, Yifan and Liu, Jiaqi and Qasem, Ashwaq},
  year = {2026},
  howpublished = {SSRN preprint},
  doi = {10.2139/ssrn.6482333},
  url = {https://ssrn.com/abstract=6482333}
}
```

## 方法与效果

![FLGenNet 原始网络结构图](Figs/FLGenNet.jpg)

![原仓库提供的增强效果展示](Figs/result2.png)

以上图片来自原项目，并非本次整理重新推理所得；更多对比见 [完整效果图](Figs/result1.jpg)及 [LOL_result](LOL_result)。

## 开始使用

在项目根目录运行。原实现含固定 `.cuda()` 调用，当前需要 **NVIDIA CUDA 环境**；`num_gpu: 0` 不能保证 CPU/Mac 推理。

依赖包括 Python、PyTorch、与之匹配的 torchvision、NumPy、OpenCV、Pillow、PyYAML、tqdm、entmax、thop、TensorBoard；训练日志若启用 W&B 还需其客户端。历史仓库没有锁定版本，本次未安装或验证完整依赖组合。

1. 从[原数据说明](docs/ORIGINAL_README.md)准备数据，保持低照度输入和正常曝光 GT 同名配对。
2. 修改 `options/test/model/test.yml` 的两个数据目录；相对路径统一以项目根目录为基准，也可填写绝对路径。
3. 在 CUDA 环境执行：

```sh
python -m codes.test -opt options/test/model/test.yml
```

输出保存在 `results/FLGenNet_LOLv2/`，拼图依次为输入、增强结果、GT。默认配置使用 LOLv2 配对测试集；不要将输入本身当作 GT 生成误导性指标。

训练入口：

```sh
python -m codes.train -opt options/train/model/FLGNet.yml
```

训练前修改 train/val 数据目录；训练配置中的 epoch/迭代、损失和预取设置保持原项目设计，不代表已经重新验证。

## 随仓库权重

| 文件 | 大小 | 状态 |
| --- | --- | --- |
| [`experiments/lol2.pth`](experiments/lol2.pth) | 1,167,943 字节，约 1.11 MiB | 原仓库自带，本次未加载或重新训练 |

SHA-256：

```text
e91d367a2d540eca9d3ec7210edce5228eb3edfde6d5ab905e46775a3cdfd4cc
```

文件名对应 LOLv2，但训练来源、参数格式与当前架构兼容性尚未通过加载检查。测试配置使用严格加载；不匹配会明确报错。上述哈希只确认文件内容，不能证明模型效果。

## 本次整理与验证

修复配置中的模型/网络类名称、FLG.py 自动发现、缺失模块导入、跨平台路径拼接和权重路径重复前缀。**神经网络架构文件 `codes/models/archs/FLG.py` 保持原样**，既有 Git 历史与许可证完整保留。

```sh
python -m unittest discover -s tests -v
```

本地检查覆盖 Python 语法、注册发现、配置类匹配和相对/绝对路径处理；不依赖 PyTorch，不读取权重。**尚未完成真实训练、推理、checkpoint 兼容性验证或 PSNR/SSIM 复测**，因此这里不填写未经验证的成绩。历史提供的效果图保留作为项目资料。

原始数据说明完整保存在 [docs/ORIGINAL_README.md](docs/ORIGINAL_README.md)，代码沿用 Apache-2.0 许可。
