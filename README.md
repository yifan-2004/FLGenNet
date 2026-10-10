# FLGenNet

**Frequency Decomposition-based Lightweight Generalization Network for Low-Light Image Enhancement.** A Laplacian pyramid separates low- and high-frequency components, enhances luminance and color at low resolution, and guides detail reconstruction.

**Yifan Ouyang, Jiaqi Liu, Ashwaq Qasem**

[Paper](https://ssrn.com/abstract=6482333) · [DOI](https://doi.org/10.2139/ssrn.6482333) · [Citation](docs/PAPER.md) · [Dataset setup](docs/ORIGINAL_README.md)

## Architecture and examples

![FLGenNet architecture](Figs/FLGenNet.jpg)

![FLGenNet enhancement examples](Figs/result2.png)

[More comparisons](Figs/result1.jpg) · [Example outputs](LOL_result). These are existing project figures; no new inference results are reported in this update.

## Test and train

Use an **NVIDIA CUDA environment**. CPU/Mac inference is not supported by the current hard-coded CUDA paths. Dependencies include PyTorch, torchvision, NumPy, OpenCV, Pillow, PyYAML, tqdm, entmax, thop and TensorBoard; versions are not pinned.

Prepare paired low-light and normally exposed images with matching filenames. Edit dataset paths in the corresponding configuration, then run from the repository root:

```bash
python -m codes.test -opt options/test/model/test.yml
python -m codes.train -opt options/train/model/FLGNet.yml
```

Test outputs go to `results/FLGenNet_LOLv2/`. Comparisons show input, enhanced image and ground truth. Use genuine ground truth for PSNR/SSIM.

## Checkpoint and checks

[`experiments/lol2.pth`](experiments/lol2.pth) is included (about 1.11 MiB). Its compatibility has not been verified by loading; no new training or benchmark run is claimed.

Maintenance fixes cover configuration names, registry discovery, imports, portable paths and checkpoint loading. Run the dependency-free checks:

```bash
python -m unittest discover -s tests -v
```

## License

[Apache-2.0](LICENSE) · [Source](https://github.com/OuyangYifan0324/FLGen-Net)
