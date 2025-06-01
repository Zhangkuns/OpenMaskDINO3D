# 基础镜像，选择带有 CUDA 支持的官方镜像
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6



# 设置环境变量
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# 安装依赖工具、SSH 服务和 Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    openssh-server

# 配置 SSH 服务
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session    required     pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# 设置 SSH 默认端口（可以根据需要调整）
EXPOSE 2222

RUN conda install torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install torch-scatter
RUN pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-deps

# 配置 PyPI 使用清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn \

# 激活 Conda 环境并安装 PyTorch 和相关库
RUN pip install pandas==1.5.3 \
    scipy==1.12.0 \
    transformers==4.46.3 \
    einops==0.6.1 \
    plyfile==1.0.1 \
    trimesh==3.23.1 \
    peft==0.9.0 \
    termcolor==2.3.0 \
    pycocoevalcap==1.2 \
    sentencepiece==0.2.0 \
    protobuf==4.25.3 \
    packaging==24.0 \
    wandb \
    accelerate==1.1.1 \
    mmengine==0.10.3

RUN pip install bitsandbytes

CMD ["/usr/sbin/sshd", "-D"]

