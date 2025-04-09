# pytorch-proVLAE
![MIT LICENSE](https://img.shields.io/badge/LICENSE-MIT-blue)
[![Format Code](https://github.com/suzuki-2001/pytorch-proVLAE/actions/workflows/black-format.yaml/badge.svg)](https://github.com/suzuki-2001/pytorch-proVLAE/actions/workflows/black-format.yaml)
[![Validate Mamba Environment](https://github.com/suzuki-2001/pytorch-proVLAE/actions/workflows/validate-mamba-env.yaml/badge.svg)](https://github.com/suzuki-2001/pytorch-proVLAE/actions/workflows/validate-mamba-env.yaml)

</br>

This is a PyTorch implementation of the paper [PROGRESSIVE LEARNING AND DISENTANGLEMENT OF HIERARCHICAL REPRESENTATIONS](https://openreview.net/forum?id=SJxpsxrYPS) by Zhiyuan et al, [ICLR 2020](https://iclr.cc/virtual_2020/poster_SJxpsxrYPS.html).
The official code for proVLAE, implemented in TensorFlow, is available [here](https://github.com/Zhiyuan1991/proVLAE).

</br>

<img src="./md/shapes3d.gif" width="100%">

⬆︎ Visualization of results when traversing the latent space of pytorch-proVLAE trained on four datasets: 3D Shapes.

&nbsp;

## Installation
We recommend using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) (via [miniforge](https://github.com/conda-forge/miniforge)) for faster installation of dependencies, but you can also use [conda](https://docs.anaconda.com/miniconda/miniconda-install/).
```bash
git clone https://github.com/suzuki-2001/pytorch-proVLAE.git
cd pytorch-proVLAE

mamba env create -f env.yaml # or conda
mamba activate torch-provlae
```

&nbsp;

## Usage
You can train pytorch-proVLAE with the following command. Sample hyperparameters and train configuration are provided in [scripts directory](./scripts/).
If you have a checkpoint file from a pythorch-proVLAE training, setting the mode argument to "traverse" allows you to inspect the latent traversal. Please ensure that the parameter settings match those used for the checkpoint file when running this mode.

</br>

```bash
# training with distributed data parallel
# we tested NVIDIA V100 PCIE 16GB+32GB, NVIDIA A6000 48GB x2
torchrun --nproc_per_node=2 --master_port=29501 src/train.py \
    --distributed \
    --mode seq_train \
    --dataset shapes3d \
    --optim adamw \
    --num_ladders 3 \
    --batch_size 128 \
    --num_epochs 15 \
    --learning_rate 5e-4 \
    --beta 8 \
    --z_dim 3 \
    --coff 0.5 \
    --pre_kl \
    --hidden_dim 32 \
    --fade_in_duration 5000 \
    --output_dir ./output/shapes3d/ \
    --data_path ./data
```
&nbsp;

## License
This repository is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details. This follows the licensing of the [original implementation license](https://github.com/Zhiyuan1991/proVLAE/blob/master/LICENSE) by Zhiyuan.

&nbsp;

***
*This repository is a contribution to [AIST (National Institute of Advanced Industrial Science and Technology)](https://www.aist.go.jp/) project.

[Human Informatics and Interaction Research Institute](https://unit.aist.go.jp/hiiri/), [Neuronrehabilitation Research Group](https://unit.aist.go.jp/hiiri/nrehrg/) \
Shosuke Suzuki, Ryusuke Hayashi
