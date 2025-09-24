FROM nvcr.io/nvidia/cuda-dl-base:25.04-cuda12.9-devel-ubuntu24.04

ARG USER=docker_user
ARG USER_ID=1000
ARG GROUP_ID=1001

ARG DEBIAN_FRONTEND=interactive


# Environment variables
ENV CUDA_HOME=/usr/local/cuda \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=jazzy \
    PIP_BREAK_SYSTEM_PACKAGES=1

# --- Fix MPI issue ---
RUN mkdir -p /opt/hpcx/ompi/lib/x86_64-linux-gnu \
    && ln -s /opt/hpcx/ompi /opt/hpcx/ompi/lib/x86_64-linux-gnu/openmpi \
    && dpkg-reconfigure libc-bin

# --- System setup ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python-is-python3 \
    git \
    openssh-client \
    wget \
    vim \
    curl \
    libeigen3-dev \
    build-essential \
    locales \
    software-properties-common \
    lsb-release \
    gnupg2 && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN add-apt-repository universe

##### ROS2 setup #####
RUN apt-get update && apt-get install locales -y
RUN locale-gen en_US.UTF-8 en_US
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

RUN apt-get update && apt install software-properties-common -y
RUN add-apt-repository universe -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt upgrade -y
RUN apt update && apt install ros-dev-tools ros-$ROS_DISTRO-desktop ros-${ROS_DISTRO}-rqt-tf-tree -y
RUN apt install -y ros-$ROS_DISTRO-tf2-geometry-msgs 
RUN rosdep init
RUN rosdep update 

##### SSH setup #####
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

##### NVIM setup #####

COPY ./setup_nvim_for_docker.bash /.setup_nvim_for_docker.bash
RUN chmod +x /.setup_nvim_for_docker.bash
RUN /.setup_nvim_for_docker.bash
RUN rm /.setup_nvim_for_docker.bash
RUN apt-get install tmux tmuxinator -y

# GTSAM deps

RUN apt-get install libboost-all-dev -y
RUN apt-get install libtbb-dev -y

WORKDIR /root/deps

RUN --mount=type=ssh git clone git@github.com:borglab/gtsam.git -b 4.2
RUN cd gtsam && git submodule update --init --recursive
WORKDIR /root/deps/gtsam/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
RUN make -j$(nproc)
RUN make install


WORKDIR /root/deps_ws/
COPY ./complementary_deps.repos ./complementary_deps.repos
RUN --mount=type=ssh vcs import < complementary_deps.repos

RUN rosdep install -y --from-paths ./src --ignore-src 
RUN source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo "source /root/deps_ws/install/setup.bash" >> ~/.bashrc
# --- Python environment setup ---
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip3 install networkx==3.1
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch \
    torchvision


# --- CLIP and Detectron2 setup ---
ARG TORCH_CUDA_ARCH_LIST="7.5;7.0+PTX"
ENV FORCE_CUDA="1"
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git@18f69583391e5040043ca4f4bebd2c60f0ebfde0'
RUN pip3 install 'git+https://github.com/openai/CLIP.git'

# --- Clone repositories ---
RUN apt-get update && apt-get install -y \
    libepoxy-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libglew-dev \
    cmake \
    build-essential \
    git
# Mount the SSH keys for cloning private repositories
RUN --mount=type=ssh git clone -b ros2-jazzy git@github.com:snt-arg/visual_sgraphs.git
RUN --mount=type=ssh git clone -b ros2-jazzy git@github.com:snt-arg/scene_segment_ros.git

RUN apt-get update && \
    apt-get install -y ros-jazzy-depth-image-proc ros-jazzy-backward-ros \
    ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-pcl-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-image-geometry \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-tf2-eigen \
    ros-${ROS_DISTRO}-tf2-sensor-msgs \
    ros-${ROS_DISTRO}-tf2-tools \
    ros-${ROS_DISTRO}-message-filters

#install g2o

# WORKDIR /root/deps
# RUN --mount=type=ssh git clone git@github.com:RainerKuemmerle/g2o.git
# WORKDIR /root/deps/g2o/build
# RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DG2O_USE_OPENGL=OFF -DG2O_USE_CSPARSE=ON -DG2O_USE_CHOLMOD=ON -DG2O_USE_CUDA=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# RUN make -j$(nproc)
# RUN make install

# Install fzf and modern tools
RUN apt-get update && apt-get install -y \
    fzf \
    tree \
    htop \
    ncdu \
    bat \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && \
#     ~/.fzf/install --all

# Install Bash-it
RUN git clone --depth=1 https://github.com/Bash-it/bash-it.git ~/.bash_it && \
    ~/.bash_it/install.sh

# Create custom bashrc with all your settings
RUN echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> ~/.bashrc && \
    echo 'alias ls="ls --color=auto"' >> ~/.bashrc && \
    echo 'alias ll="ls -alF --color=auto"' >> ~/.bashrc && \
    echo 'alias cb="colcon build --symlink-install"' >> ~/.bashrc && \
    echo 'alias si="source install/setup.bash"' >> ~/.bashrc && \
    echo 'alias jazy="rsource /opt/ros/jazzy/setup.bash"' >> ~/.bashrc && \
    echo 'export CLICOLOR=1' >> ~/.bashrc && \
    echo 'export TERM=xterm-256color' >> ~/.bashrc && \
    echo 'source /usr/share/bash-completion/completions/fzf' >> ~/.bashrc && \
    echo 'source /usr/share/doc/fzf/examples/key-bindings.bash' >> ~/.bashrc && \
    echo 'export HISTSIZE=10000' >> ~/.bashrc && \
    echo 'export HISTFILESIZE=20000' >> ~/.bashrc && \
    echo 'export HISTCONTROL=ignoredups:erasedups' >> ~/.bashrc && \
    echo 'shopt -s histappend' >> ~/.bashrc
# echo 'export FZF_CTRL_T_OPTS="--preview '\''bat --color=always --style=header,grid --line-range :300 {}'\'"' >> ~/.bashrc

# Copy and run the fzf configuration script
COPY fzf_config.sh /tmp/fzf_config.sh
RUN chmod +x /tmp/fzf_config.sh && /tmp/fzf_config.sh && rm /tmp/fzf_config.sh

# VSCode setup
RUN apt-get update && apt-get install -y \
    wget \
    gpg \
    && wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg \
    && install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/ \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list \
    && apt-get update \
    && apt-get install -y code


WORKDIR /root/workspace/src
RUN --mount=type=ssh git clone git@github.com:snt-arg/situational_graphs.git -b ros2-jazzy-multifloor
WORKDIR /root/workspace/
RUN source /root/deps_ws/install/setup.bash && rosdep install -y --from-paths ./src --ignore-src 
RUN export MAKEFLAGS="-j 1"
# RUN source /root/deps_ws/install/setup.bash && colcon build --executor sequential --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DUSE_RGB_CLOUD=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo  
#export ros variables in bashrc
WORKDIR /root

RUN sudo apt-get install -y libg2o-dev
RUN sudo apt-get install -y ros-jazzy-rmw-cyclonedds-cpp
RUN sudo apt-get install -y ros-jazzy-cyclonedds
RUN echo "source /root/.complementary_bashrc" >> ~/.bashrc
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod a+x /entrypoint.sh
ENTRYPOINT ["bash", "/entrypoint.sh"]
