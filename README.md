# pytorch-proVLAE
![MIT LICENSE](https://img.shields.io/badge/LICENSE-MIT-blue)
[![Format Code](https://github.com/suzuki-2001/pytorch-proVLAE/actions/workflows/black-format.yaml/badge.svg)](https://github.com/suzuki-2001/pytorch-proVLAE/actions/workflows/black-format.yaml)
[![Validate Mamba Environment](https://github.com/suzuki-2001/pytorch-proVLAE/actions/workflows/validate-mamba-env.yaml/badge.svg)](https://github.com/suzuki-2001/pytorch-proVLAE/actions/workflows/validate-mamba-env.yaml)

</br>

This is a PyTorch implementation of the paper [PROGRESSIVE LEARNING AND DISENTANGLEMENT OF HIERARCHICAL REPRESENTATIONS](https://openreview.net/forum?id=SJxpsxrYPS) by Zhiyuan et al, [ICLR 2020](https://iclr.cc/virtual_2020/poster_SJxpsxrYPS.html).
The official code for proVLAE, implemented in TensorFlow, is available [here](https://github.com/Zhiyuan1991/proVLAE).

</br>

<table>
<tr>
<td><img src="./md/shapes3d.gif" width="100%"></td>
<td><img src="./md/mnist.gif" width="100%"></td>
</tr>
<tr>
<td><img src="./md/ident3d.gif" width="100%"></td>
<td><img src="./md/mpi3d.gif" width="100%"></td>
</tr>
</table>

⬆︎ Visualization of results when traversing the latent space of pytorch-proVLAE trained on four datasets: 3D Shapes (top-left), MNIST (top-right), 3DIdent (bottom-left), and MPI3D (bottom-right). The results for the last two datasets are preliminary, with hyperparameter tuning still in progress.

&nbsp;

#### Model Architecture 
> Figure 1: Progressive learning of hierarchical representations. White blocks and solid lines are VAE
> models at the current progression. α is a fade-in coefficient for blending in the new network component. Gray circles and dash line represents (optional) constraining of the future latent variables.

![figure-1 in pro-vlae paper](./md/provlae-figure1.png)

⬆︎ The figure from Zhiyuan et al. (ICLR 2020) illustrates the ladder architecture of the VAE and the progressive learning of hierarchical representations.

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

</br>

- ### Hyper Parameters

| Argument | Default | Description |
|----------|---------------|-------------|
| `z_dim` | 3 | Dimension of latent variables |
| `num_ladders` | 3 | Number of ladders (hierarchies) in pro-VLAE |
| `beta` | 8.0 | β parameter for pro-VLAE |
| `learning_rate` | 5e-4 | Learning rate |
| `fade_in_duration` | 5000 | Number of steps for fade-in period |
| `image_size` | 64 | Input image size |
| `chn_num` | 3 | Number of input image channels |
| `train_seq` | 1 | Current training sequence number (`indep_train mode` only) |
| `batch_size` | 100 | Batch size |
| `num_epochs` | 1 | Number of epochs |
| `hidden_dim` | 32 | Hidden layer dimension |
| `coff` | 0.5 | Coefficient for KL divergence |
| `pre_kl` | True | Use inactive ladder loss |
| `use_kl_annealing` | False | Enable KL annealing (see [haofuml/cyclical_annealing](https://github.com/haofuml/cyclical_annealing)) |
| `kl_annealing_mode` | "linear" | KL cycle annealing mode (linear, sigmoid, cosine) |
| `cycle_period` | 4 | The number of annealing |
| `max_kl_weight` | 1.0 | Max KL weight |
| `min_kl_weight` | 0.1 | Min KL weight |

</br>

- ### Training Parameters

| Argument | Default | Description |
|----------|---------------|-------------|
| `dataset` | "shapes3d" | Dataset to use (mnist, fashionmnist, dsprites, shapes3d, mpi3d, ident3d,celeba, flowers102, dtd, imagenet) |
| `data_path` | "./data" | Path to dataset storage |
| `output_dir` | "outputs" | Output directory |
| `checkpoint_dir` | "checkpoints" | Checkpooints results directory |
| `recon_dir` | "reconstructions" | Reconstructions results directory |
| `traverse_dir` | "travesals" | Traversal results directory |
| `mode` | "seq_train" | Execution mode ("seq_train", "indep_train", "traverse") |
| `compile_mode` | "default" | PyTorch compilation mode |
| `on_cudnn_benchmark` | True | Enable/disable cuDNN benchmark |
| `optim` | "adam" | Optimization algorithm (adam, adamw, sgd, lamb, diffgrad, madgrad) |
| `distributed` | False | enable distributed data parallel |
| `num_workers` | 4 | Number of workers for data loader |
| `use_wandb` | False | Enable wandb tracking and logging |
| `wandb_project` | "provlae" | Wandb project name |

</br>

Mode descriptions:
- `seq_train`: Sequential training from ladder 1 to `num_ladders`
- `indep_train`: Independent training of specified `train_seq` ladder
- `traverse`: Visualize latent space using trained model (need checkpoints)

&nbsp;

## PyTorch optimization options

- __Fast training__: performance tuning referred to in [PyTorch Performance Tuning Guide - Szymon Migacz, NVIDIA](https://t.co/7CIDWfrI0J).
    - [torch.backends.cudnn.benchmark](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.benchmark)
    - [Automatic Mixed Precision (FP16/BF16)](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
    - [Asynchronous GPU Copies](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
    - [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
    - [Tensor Float 32 (>= Ampere)](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)
    - [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html)


- __Optimizer__: DiffGrad, Lamb, MADGRAD is implemented by [jettify/pytorch-optimizer](https://github.com/jettify/pytorch-optimizer), other optimizer based [torch.optim package](torch.serialization.add_safe_globals([set])).
    - Adam
    - AdamW
    - SGD
    - [DiffGrad](https://arxiv.org/abs/1909.11015)
    - [Lamb](https://arxiv.org/abs/1904.00962)
    - [MADGRAD](https://arxiv.org/abs/2101.11075)


&nbsp;

## Dataset
Here is a lists of datasets used in the original proVLAE paper, along with additional disentanglement datasets. The datasets are automatically downloaded and preprocessed when you specify the dataset name in the `--dataset` argument except for ImageNet. 

### Datasets used in the original proVLAE paper
1. [MNIST](https://yann.lecun.com/exdb/mnist/): `mnist`
2. [Disentanglement testing Sprites dataset (dSprites)](https://github.com/google-deepmind/dsprites-dataset): `dsprites`
3. [3D Shapes](https://github.com/google-deepmind/3d-shapes): `shapes3d`
4. [Large-scale CelebFaces Attributes (CelebA)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): `celeba`

### Additional Disentanglement Datasets
1. [MPI3D Disentanglement Datasets](https://github.com/rr-learning/disentanglement_dataset?tab=readme-ov-file): `mpi3d`
2. [3DIdent](https://paperswithcode.com/dataset/3dident): `ident3d`

### Other Datasets (testing in progress)
1. [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): `fashionmnist`
2. [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/): `dtd`
3. [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/): `flowers102`
4. [ImageNet](https://www.image-net.org/): `imagenet`

&nbsp;

## Work in Progress
Hyperparameter optimization (beta, coff, fade-in duration, learning rates) and implementation of disentanglement metrics (MIG for detecting factor splitting, MIG-sup for factor entanglement) are currently under development. Benchmark results will be provided in future updates.

&nbsp;

## License
This repository is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details. This follows the licensing of the [original implementation license](https://github.com/Zhiyuan1991/proVLAE/blob/master/LICENSE) by Zhiyuan.

&nbsp;

***
*This repository is a contribution to [AIST (National Institute of Advanced Industrial Science and Technology)](https://www.aist.go.jp/) project.

[Human Informatics and Interaction Research Institute](https://unit.aist.go.jp/hiiri/), [Neuronrehabilitation Research Group](https://unit.aist.go.jp/hiiri/nrehrg/) \
Shosuke Suzuki, Ryusuke Hayashi
