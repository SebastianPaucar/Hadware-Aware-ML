# Hardware-Aware ML for CMS Level-1 Trigger

Compute nodes from the Center for High Performance Computing at the University of Utah will be used for this project. The environment hosts the following configuration:

| Component    |            Version               |
|--------------|----------------------------------|
| OS           | RockyLinux 8.10 (Green Obsidian) |

The project is configured within the /scratch/ high-capacity storage space under the `ha-ml/` directory. The `apptainer` module is loaded from the server `Lmod` stack to manage containers.

```bash
[u6059911@notchpeak1:~]$ mkdir -p /scratch/general/vast/$USER/ha-ml
[u6059911@notchpeak1:~]$ cd /scratch/general/vast/$USER/ha-ml
[u6059911@notchpeak1:ha-ml]$ module load apptainer/1.4.1
```

`AlmaLinux 9.7` is required as an LXPLUS-like container environment to run CERN-related fast-ML packages while pinning exact versions of TensorFlow/Keras, CUDA/cuDNN, and HGQ2. Please remember the recommended backend requirements for HGQ2 to function properly:

|  Component   |   Version  |
|--------------|------------|
|   `python`   |  `>=3.11`  |
|   `keras`    |    `>=3`   |
| `tensorlfow` |  `>=2.16`  |
|    `jax`     | `>=0.4.28` |
|   `torch`    |  `>=2.5.0` |

The attached `environment.yaml` satisfies these requirements.

## Set up the GPU-aware container

Configure a Apptainer container from a minimal AlmaLinux `def` file under active development:

```bash
(base) [u6059911@notchpeak1:ha-ml]$ cat alma9-dev.def 
Bootstrap: docker
From: almalinux:9

%setup
    mkdir -p ${APPTAINER_ROOTFS}/uufs
    mkdir -p ${APPTAINER_ROOTFS}/scratch
    mkdir -p ${APPTAINER_ROOTFS}/home

%post
    dnf install -y --allowerasing --nobest curl git wget which tar vim

%runscript
    exec bash
(base) [u6059911@notchpeak1:ha-ml]$ apptainer build --fakeroot --fix-perms -s alma9-dev alma9-dev.def
```

Apptainer flags used:

* `--fakeroot`: Maps the host user UID to the root UID (0) inside the container.
* `--fix-perms`: Ensures consistent read/write/execute permissions inside the container after the build completes.
* `-s` or `--sandbox`: Builds the container as a writable directory-based filesystem instead of a compressed immutable `.sif` image. Useful for iterative development and debugging.

Run the container and verify the operating system:

```bash
[u6059911@notchpeak1:ha-ml]$ apptainer shell --writable --fakeroot --nv --bind /uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-nehalem/gcc-8.5.0/cuda-12.5.0-7pt27rceb2uulofwak7zo3xkzspxgkg2:/usr/local/cuda  alma9-dev
Apptainer> cat /etc/os-release
NAME="AlmaLinux"
VERSION="9.7 (Moss Jungle Cat)"
ID="almalinux"
ID_LIKE="rhel centos fedora"
VERSION_ID="9.7"
PLATFORM_ID="platform:el9"
PRETTY_NAME="AlmaLinux 9.7 (Moss Jungle Cat)"
ANSI_COLOR="0;34"
LOGO="fedora-logo-icon"
CPE_NAME="cpe:/o:almalinux:almalinux:9::baseos"
HOME_URL="https://almalinux.org/"
DOCUMENTATION_URL="https://wiki.almalinux.org/"
BUG_REPORT_URL="https://bugs.almalinux.org/"

ALMALINUX_MANTISBT_PROJECT="AlmaLinux-9"
ALMALINUX_MANTISBT_PROJECT_VERSION="9.7"
REDHAT_SUPPORT_PRODUCT="AlmaLinux"
REDHAT_SUPPORT_PRODUCT_VERSION="9.7"
SUPPORT_END=2032-06-01
```

Apptainer flags used:

* `--writable`: Remounts the container root filesystem with read-write permissions, persisting all modifications directly to the sandbox image on disk.
* `--nv`: Enables NVIDIA GPU passthrough within the container while exposing the host driver version directly.
* `--bind`: It mounts a complete CUDA 12.5 toolkit from the host CHPC Spack installation into `/usr/local/cuda` inside the container.

Download Conda, configure it properly in the `PATH`, and initialize it:

```bash
Apptainer> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /opt/
Apptainer> bash /opt/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
Apptainer> export PATH="/opt/conda/bin:$PATH"
Apptainer> source /opt/conda/etc/profile.d/conda.sh
```

Set up the `ha-ml` Conda environment and install GPU-aware PyTorch with statically linked CUDA 12.4 compute kernels, along with the cuDNN shared libraries loaded by PyTorch at runtime:

```bash
Apptainer> conda env create -f /opt/environment.yml
Apptainer> conda activate ha-ml
(ha-ml) Apptainer> pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
(ha-ml) Apptainer> pip install nvidia-cudnn-cu12==9.3.0.75
```

Create mount-point directories for NVIDIA-related files inside the container:

```bash
(ha-ml) Apptainer> mkdir -p /usr/local/cuda
(ha-ml) Apptainer> mkdir -p /usr/share/egl/egl_external_platform.d
(ha-ml) Apptainer> mkdir -p /usr/share/nvidia
(ha-ml) Apptainer> mkdir -p /usr/share/glvnd/egl_vendor.d
(ha-ml) Apptainer> mkdir -p /usr/share/vulkan/implicit_layer.d
(ha-ml) Apptainer> mkdir -p /var/run/nvidia-persistenced
(ha-ml) Apptainer> touch /usr/bin/nvidia-smi
(ha-ml) Apptainer> touch /usr/bin/nvidia-debugdump
(ha-ml) Apptainer> touch /usr/bin/nvidia-persistenced
(ha-ml) Apptainer> touch /usr/bin/nvidia-cuda-mps-control
(ha-ml) Apptainer> touch /usr/bin/nvidia-cuda-mps-server
(ha-ml) Apptainer> touch /usr/share/nvidia/nvoptix.bin
(ha-ml) Apptainer> touch /var/run/nvidia-persistenced/socket
(ha-ml) Apptainer> touch /usr/share/vulkan/implicit_layer.d/nvidia_layers.json
(ha-ml) Apptainer> touch /usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json
(ha-ml) Apptainer> touch /usr/share/egl/egl_external_platform.d/10_nvidia_wayland.json
(ha-ml) Apptainer> touch /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

Expose CUDA toolkit binaries and register CUDA/cuDNN shared libraries required for TensorFlow and PyTorch GPU runtime resolution:

```bash
(ha-ml) Apptainer> export PATH=/usr/local/cuda/bin:$PATH
(ha-ml) Apptainer> export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
(ha-ml) Apptainer> export LD_LIBRARY_PATH=/opt/conda/envs/ha-ml/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

Verify NVIDIA driver visibility, GPU passthrough, and CUDA toolkit availability inside the container:

```bash
(ha-ml) Apptainer> nvidia-smi
Sat May 16 00:24:03 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GT 1030         On  |   00000000:86:00.0 Off |                  N/A |
| 38%   34C    P8            N/A  /   19W |     146MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         1536396      G   chimerax                                142MiB |
+-----------------------------------------------------------------------------------------+
(ha-ml) Apptainer> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Apr_17_19:19:55_PDT_2024
Cuda compilation tools, release 12.5, V12.5.40
Build cuda_12.5.r12.5/compiler.34177558_0
```

Verify framework-level GPU acceleration and confirm CUDA/cuDNN runtime compatibility for TensorFlow and PyTorch:

```bash
(ha-ml) Apptainer> python -c "
import tensorflow as tf
import torch

print('TensorFlow:', tf.__version__,
      '| CUDA:', tf.sysconfig.get_build_info().get('cuda_version'),
      '| cuDNN:', tf.sysconfig.get_build_info().get('cudnn_version'),
      '| GPU:', tf.config.list_physical_devices('GPU'))

print('PyTorch:', torch.__version__,
      '| CUDA:', torch.version.cuda,
      '| cuDNN:', torch.backends.cudnn.version(),
      '| GPU:', torch.cuda.is_available())
"
TensorFlow: 2.21.0 | CUDA: 12.5.1 | cuDNN: 9 | GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
PyTorch: 2.5.1+cu124 | CUDA: 12.4 | cuDNN: 90300 | GPU: True
```

The output output confirms that:

* TensorFlow successfully detects the GPU through CUDA 12.5.1 and cuDNN 9.
* PyTorch successfully detects the GPU with CUDA 12.4 and cuDNN 9.3.0.
* GPU passthrough and runtime library resolution are functioning correctly inside the container.

# Set up the Keras backend

Configure Keras 3 to use TensorFlow as the backend runtime instead of JAX or PyTorch.

```bash
(ha-ml) Apptainer> export KERAS_BACKEND=tensorflow
```

# Quick Setup

Consider:

* `alma9-haml.def`: Apptainer Docker-based manifest that configures the full GPU-aware `ha-ml` AlmaLinux environment end-to-end.
* `set-up-sandbox.sh`: Sets up the fakeroot sandbox associated with `alma9-haml.def` for continuous development.
* `set-up-sif.sh`: Builds the resulting persistent, immutable `.sif` image from the sandbox.

To set up the environment properly, follow these steps:
 
**1. Set up the sandbox:**

```bash
nohup bash set-up-sandbox.sh > build_sandbox.log 2>&1 &
```

To enter the container through the sandbox for development:

```bash
apptainer shell --writable --fakeroot --nv --bind /uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-nehalem/gcc-8.5.0/cuda-12.5.0-7pt27rceb2uulofwak7zo3xkzspxgkg2:/usr/local/cuda alma9-haml
```

**2. Build the image:**

```bash
nohup bash set-up-sif.sh > build_sif.log 2>&1 &
```

To enter the container through the `.sif` image:

```bash
apptainer shell --nv --bind /uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-nehalem/gcc-8.5.0/cuda-12.5.0-7pt27rceb2uulofwak7zo3xkzspxgkg2:/usr/local/cuda alma9-haml.sif
```

