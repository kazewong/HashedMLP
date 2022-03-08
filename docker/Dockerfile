FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt-get update && apt-get install -y openssh-server && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /workspace/requirements.txt
RUN conda install h5py && conda clean --all
RUN pip install -r /workspace/requirements.txt --no-cache-dir

