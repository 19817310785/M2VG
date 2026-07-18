# Reproducibility notes

This document records the technical setup while the anonymous submission README
and framework figure are being prepared.

## Environment

The reference environment is Python 3.8, CUDA 11.8, PyTorch 2.0.0, and
Torchvision 0.15.1. Create it with:

```bash
conda env create -f environment.yml
conda activate m2vg
```

Install the repository dependencies and the external detection components:

```bash
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule update --init
pip install -e .
cd ..
pip install -e .
```

## External artifacts

Datasets, pretrained weights, and trained checkpoints are deliberately excluded
from version control. Place them in these paths before training or evaluation:

```text
data/
pretrain_weights/
work_dir/
```

Prepare the MS-COCO `train2014` images from the
[official COCO download page](https://cocodataset.org/#download). Download the
referring-expression and foreground annotations from the
[annotation release](https://huggingface.co/datasets/Dmmm997/InstanceVG_Data)
and arrange them under `data/seqtr_type/` as required by the configuration
files.

Download the BEiT-3 pretrained backbone files and tokenizer from the
[official BEiT-3 release](https://github.com/microsoft/unilm/tree/master/beit3)
and place them under `pretrain_weights/`.

Place the final M2VG checkpoints under `work_dir/` and use the matching M2VG
configuration:

| M2VG configuration | Compatible checkpoint |
| --- | --- |
| `configs/gres/M2VG-grefcoco.py` | `M2VG-grefcoco.pth` |
| `configs/refcoco/M2VG-B-refcoco/M2VG-B-refcoco.py` | `M2VG-B-refcoco.pth` |
| `configs/refcoco/M2VG-L-refcoco/M2VG-L-refcoco.py` | `M2VG-L-refcoco.pth` |
| `configs/refzom/M2VG-refzom.py` | `M2VG-refzom.pth` |
| `configs/rrefcoco/M2VG-rrefcoco.py` | `M2VG-rrefcoco.pth` |

## Entry points

```bash
bash tools/dist_train.sh [PATH_TO_CONFIG] [NUM_GPUS]
bash tools/dist_test.sh [PATH_TO_CONFIG] [NUM_GPUS] --load-from [CHECKPOINT]
```

Available task configurations include:

- `configs/gres/M2VG-grefcoco.py`
- `configs/refcoco/M2VG-B-refcoco/M2VG-B-refcoco.py`
- `configs/refcoco/M2VG-L-refcoco/M2VG-L-refcoco.py`
- `configs/refzom/M2VG-refzom.py`
- `configs/rrefcoco/M2VG-rrefcoco.py`
