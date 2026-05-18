```bash
[u6059911@notchpeak1:~]$ mkdir -p /scratch/general/vast/$USER/ha-ml
[u6059911@notchpeak1:~]$ cd /scratch/general/vast/$USER/ha-ml
[u6059911@notchpeak1:ha-ml]$ module load apptainer/1.4.1
```

```bash
(base) [u6059911@notchpeak1:ha-ml]$ cat alma9-dev.def 
Bootstrap: docker
From: almalinux:9

%setup
    mkdir -p ${APPTAINER_ROOTFS}/uufs
    mkdir -p ${APPTAINER_ROOTFS}/scratch
    mkdir -p ${APPTAINER_ROOTFS}/home

%post
    dnf install -y --allowerasing --nobest curl git wget which tar

%runscript
    exec bash
```

```bash
[u6059911@notchpeak1:ha-ml]$ apptainer shell --writable --fakeroot --nv \
>   --bind /uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-nehalem/gcc-8.5.0/cuda-12.5.0-7pt27rceb2uulofwak7zo3xkzspxgkg2:/usr/local/cuda \
>   alma9-dev
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
(ha-ml) Apptainer> dnf install -y vim
```


```bash
Apptainer> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /opt/
Apptainer> bash /opt/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
Apptainer> export PATH="/opt/conda/bin:$PATH"
Apptainer> conda env create -f /opt/environment.yml
Apptainer> source /opt/conda/etc/profile.d/conda.sh
Apptainer> conda activate ha-ml
(ha-ml) Apptainer> pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
(ha-ml) Apptainer> pip install nvidia-cudnn-cu12==9.3.0.75
```

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
(ha-ml) Apptainer> export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
(ha-ml) Apptainer> export PATH=/usr/local/cuda/bin:$PATH
```

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


```bash
(ha-ml) Apptainer> export LD_LIBRARY_PATH=/opt/conda/envs/ha-ml/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
(ha-ml) Apptainer> echo $LD_LIBRARY_PATH 
/opt/conda/envs/ha-ml/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64::/.singularity.d/libs:/usr/local/cuda-10.1/lib64
```