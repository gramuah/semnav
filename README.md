![](imgs/SemNavimg.png)
# SemNav: Semantic Segmentation for Visual Semantic Navigation

Code for our paper: [SemNav: Semantic Segmentation for Visual Semantic Navigation]().

**Authors:** Rafael Flor Rodríguez-Rabadán, Carlos Gutiérrez Álvarez, Roberto Javier López Sastre.  
[Group Page](https://gram.web.uah.es/)

## Overview

SemNav is a visual semantic navigation model that achieves successful navigations through imitation learning, using semantic segmentation images as input to the model.

This paradigm advances the state of the art in both simulated environments and real-world scenarios by leveraging scene information, reducing the domain gap between training in simulation and deployment in reality. Our findings demonstrate that using semantic segmentation significantly improves state-of-the-art results in the ObjectNav task, relying solely on imitation learning.

Read more in the [paper](#).

---

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/example/semnav.git

conda create -n semnav python=3.9 cmake=3.18.0
```

### Install Habitat-Sim

```bash
git clone --depth 1 --branch v0.2.2 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim/
pip install -r requirements.txt
python setup.py install --headless

pip3 install torch torchvision torchaudio
```

### Install Habitat-Lab

```bash
pip install gym==0.22.0 urllib3==1.25.11 numpy==1.25.0 pillow==9.2.0
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
python setup.py develop --install
conda install protobuf  
```

---

## Docker Setup

### Build the Docker Image

Navigate to the directory containing the `Dockerfile` and execute:

```bash
docker build -t semnav:latest -f docker/Dockerfile .
```

### Run the Docker Container

```bash
docker run --gpus all -it --rm --name semnav_container semnav:latest
```

### Access the Running Container

```bash
docker exec -it semnav_container /bin/bash
```

### Stop the Container

```bash
docker stop semnav_container
```

The Dockerfile sets up the complete environment, including:
- CUDA and cuDNN for GPU support
- Conda for environment management
- Habitat-Sim and Habitat-Lab for simulation tasks
- Essential Python libraries: PyTorch, torchvision, torchaudio

Ensure the entry script `/home/your_username/entrypoint.sh` is present and executable.

---

## Datasets

We provide two datasets, **SemNav 40** and **SemNav 1630**, for leveraging semantic segmentation information:

- **SemNav 1630**: Built using human-annotated semantic labels from [HM3D Semantics](https://github.com/facebookresearch/habitat-lab/tree/main/habitat/data/datasets/hm3d_semantics).
- **SemNav 40**: Derived by mapping these annotations to the 40 categories of [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

| Dataset      | Download Link |
|-------------|--------------|
| **SemNav 40**  | [Download](#) |
| **SemNav 1630** | [Download](#) |


Additionally, download the **ObjectNav HM3D episode dataset** from [this link](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets).

---

## Pretrained Checkpoints

We provide multiple trained configurations. The **pretrained_ckpt** directory contains checkpoints for the **SemNav 40** dataset in two setups:
- **Only Semantic**
- **Semantic+RGB**

---

## Training

To train a model from scratch, run:

```bash
sbatch scripts/1-objectnav-il.sh objectnav_hm3d_hd
```

The training dataset is available in the [PirlNav repository](https://github.com/Ram81/pirlnav?tab=readme-ov-file).

Modify the training configuration in:
```
configs/experiments/il_objectnav.yaml
```

### Policy Options

- **`SEMANTIC_ObjectNavILMAEPolicy`**: Uses only semantic segmentation.
- **`SEMANTIC_RGB_ObjectNavILMAEPolicy`**: Uses both semantic segmentation and RGB.
- **`RGB_ObjectNavILMAEPolicy`**: Uses only RGB.

Pretrained visual encoder weights can be downloaded from the [PirlNav repository](https://github.com/Ram81/pirlnav?tab=readme-ov-file).

---

## Evaluation

Run the evaluation with:

```bash
sbatch scripts/1-objectnav-il-eval.sh /path/to/checkpoint
```

To evaluate pretrained models, select a checkpoint from **pretrained_ckpt**.

---

For further information, refer to our [paper](#) or visit our [Group Page](https://gram.web.uah.es/).

