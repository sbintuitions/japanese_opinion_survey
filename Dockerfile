FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04 as base

### Install python 3.10 and set it as default python interpreter
RUN  apt update &&  apt install software-properties-common -y && \
add-apt-repository ppa:deadsnakes/ppa -y &&  apt update && \
apt install curl python3.10 build-essential vim git -y && \
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
apt install python3.10-venv python3.10-dev -y && \
curl -Ss https://bootstrap.pypa.io/get-pip.py | python3.10 && \
apt-get clean && rm -rf /var/lib/apt/lists/

RUN pip install poetry
RUN pip install --upgrade pip

# --------------------------------------------------------------
# Install rust (*** 追加した部分!!! ***)
#
# NOTE: Mac PC で build する場合のみ Rust が必要な模様
# --------------------------------------------------------------
ENV PATH=$PATH:/root/.cargo/bin
RUN curl https://sh.rustup.rs -sSf > /rust.sh && sh /rust.sh -y \
    && rustup install stable

WORKDIR /workspace/japanese_opinion_survey
COPY ./src ./src
COPY ./pyproject.toml ./pyproject.toml
RUN poetry install
ENV PYTHONPATH "${PYTHONPATH}:./"
