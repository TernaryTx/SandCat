#!/usr/bin/env bash
# Install GROMACS 2021.3 + PLUMED 2.8.0 (with OPES module) on a Linux GPU host.
#
# Tested on: Ubuntu 22.04, CUDA 11.8 (e.g. p3/p4 instances with NVIDIA drivers
# already installed by the host image or manually).
#
# Run as root or with sudo. Installs to /opt/gromacs and /opt/plumed.
# Adds PATH and LD_LIBRARY_PATH exports to /etc/profile.d/gromacs_plumed.sh
# so they are available in all future shells.
#
# Usage:
#   chmod +x install_gromacs_plumed.sh
#   sudo ./install_gromacs_plumed.sh

set -euo pipefail

PLUMED_VERSION="2.8.0"
GROMACS_VERSION="2021.3"
PLUMED_PREFIX="/opt/plumed"
GROMACS_PREFIX="/opt/gromacs"
CUDA_ROOT="/usr/local/cuda"
BUILD_DIR="$(mktemp -d)"

echo "=== Installing build dependencies ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
    build-essential \
    libfftw3-dev \
    libopenmpi-dev \
    openmpi-bin \
    wget \
    curl \
    git \
    zlib1g-dev \
    libbz2-dev \
    python3 \
    python3-dev \
    unzip \
    ca-certificates

echo "=== Installing CMake 3.28.6 ==="
curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.sh \
    -o /tmp/cmake.sh
sh /tmp/cmake.sh --prefix=/usr/local --skip-license
rm /tmp/cmake.sh

echo "=== Building PLUMED ${PLUMED_VERSION} with OPES module ==="
cd "$BUILD_DIR"
wget https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/plumed-${PLUMED_VERSION}.tgz
tar xf plumed-${PLUMED_VERSION}.tgz
cd plumed-${PLUMED_VERSION}
./configure \
    --prefix="${PLUMED_PREFIX}" \
    CXX=mpicxx CC=mpicc \
    --enable-mpi \
    --enable-fftw \
    --enable-modules=+opes
make -j"$(nproc)"
make install
cd "$BUILD_DIR"
rm -rf plumed-${PLUMED_VERSION} plumed-${PLUMED_VERSION}.tgz

export PATH="${PLUMED_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${PLUMED_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "=== Building GROMACS ${GROMACS_VERSION} patched with PLUMED ==="
cd "$BUILD_DIR"
wget https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz
tar xf gromacs-${GROMACS_VERSION}.tar.gz
cd gromacs-${GROMACS_VERSION}
plumed patch -p -e gromacs-${GROMACS_VERSION}
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${GROMACS_PREFIX}" \
    -DGMX_BUILD_OWN_FFTW=OFF \
    -DGMX_FFT_LIBRARY=fftw3 \
    -DGMX_GPU=CUDA \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_ROOT}" \
    -DGMX_MPI=ON \
    -DGMX_OPENMP=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DREGRESSIONTEST_DOWNLOAD=OFF \
    -DGMX_SIMD=AVX2_256
make -j"$(nproc)"
make install
cd "$BUILD_DIR"
rm -rf gromacs-${GROMACS_VERSION} gromacs-${GROMACS_VERSION}.tar.gz

echo "=== Writing environment setup to /etc/profile.d/gromacs_plumed.sh ==="
cat > /etc/profile.d/gromacs_plumed.sh <<EOF
export PATH="${GROMACS_PREFIX}/bin:${PLUMED_PREFIX}/bin:\${PATH}"
export LD_LIBRARY_PATH="${GROMACS_PREFIX}/lib:${PLUMED_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
# allow MPI to run as root (common in single-node containers)
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
EOF

rmdir "$BUILD_DIR"

echo ""
echo "=== Installation complete ==="
echo "PLUMED ${PLUMED_VERSION} -> ${PLUMED_PREFIX}"
echo "GROMACS ${GROMACS_VERSION} -> ${GROMACS_PREFIX}"
echo ""
echo "Open a new shell or run: source /etc/profile.d/gromacs_plumed.sh"
echo "Then verify with: gmx_mpi --version && plumed --version"
