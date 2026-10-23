FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    bzip2 \
    wget \
    openmpi-bin \
    openmpi-common \
    libopenmpi3 \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Required to run mpirun as root inside a container
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV LD_LIBRARY_PATH="/opt/miniconda/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/openmpi/lib:${LD_LIBRARY_PATH}"

# Ubuntu 22.04 ships libmpi .so.40; conda packages expect .so.20
RUN cp /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.40 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.20 && \
    cp /usr/lib/x86_64-linux-gnu/libmpi.so.40 /usr/lib/x86_64-linux-gnu/libmpi.so.20

# micromamba: single ~5 MB binary replaces ~800 MB Miniconda
ENV MAMBA_ROOT_PREFIX=/opt/miniconda \
    PATH=$PATH:/opt/miniconda/bin
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba && \
    micromamba shell init -s bash --root-prefix $MAMBA_ROOT_PREFIX

SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]

RUN micromamba install -y -n base -c conda-forge -c omnia \
    python=3.12 "numpy<2" \
    openmm cuda-version=12.2 cuda-nvcc cuda-nvrtc cuda-runtime \
    openmmforcefields openff-toolkit \
    openmmtools \
    rdkit \
    ambertools \
    pdbfixer \
    propka \
    "mdtraj<1.11" \
    mdanalysis \
    biopython \
    pandas \
    mpi4py \
    && micromamba clean -a -y

# Setup sandcat package
WORKDIR "/sandcat"
COPY . /sandcat
RUN pip install -e ./
