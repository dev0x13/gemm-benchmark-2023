FROM ubuntu:22.04

ARG EIGEN_VERSION=3.4.0
ARG OPENBLAS_VERSION=0.3.25
ARG MKL_VERSION=2024.0.0.49673
ARG MOJO_VERSION=0.6.0
ARG MODULAR_AUTH_TOKEN=

WORKDIR /app

RUN apt-get update -q \
  && apt-get install -y \
    wget \
    tar \
    gpg \
    apt-transport-https \
    clang \
    libomp-dev \
    make \
    python3-venv

# Install Eigen
RUN mkdir eigen \
  && wget -qO- https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz | tar xz --strip-components=1 -C eigen
ENV EIGENROOT=/app/eigen

# Install Intel MKL
RUN wget -q https://registrationcenter-download.intel.com/akdlm/IRC_NAS/86d6a4c1-c998-4c6b-9fff-ca004e9f7455/l_onemkl_p_${MKL_VERSION}_offline.sh \
  && chmod +x l_onemkl_p_${MKL_VERSION}_offline.sh \
  && ./l_onemkl_p_${MKL_VERSION}_offline.sh -a -s --eula accept \
  && rm l_onemkl_p_${MKL_VERSION}_offline.sh
ENV MKLROOT=/opt/intel/oneapi/mkl/latest

# Install OpenBLAS
RUN mkdir openblas \
  && wget -qO- https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz | tar xz --strip-components=1 -C openblas \
  && (cd openblas && make -j $(nproc) && make install) \
  && rm -rf openblas
ENV OPENBLASROOT=/opt/OpenBLAS

# Install Modular CLI (needed for Mojo installation)
RUN keyring_location=/usr/share/keyrings/modular-installer-archive-keyring.gpg \
  && wget -qO- 'https://dl.modular.com/bBNWiLZX5igwHXeu/installer/gpg.0E4925737A3895AD.key' | gpg --dearmor >> ${keyring_location} \
  && wget -qO- 'https://dl.modular.com/bBNWiLZX5igwHXeu/installer/config.deb.txt?distro=debian&codename=wheezy' > /etc/apt/sources.list.d/modular-installer.list \
  && apt-get update -q \
  && apt-get install -y modular

# Install Mojo
RUN if [ -n "${MODULAR_AUTH_TOKEN}" ]; then \
   modular auth "${MODULAR_AUTH_TOKEN}" && modular install --install-version ${MOJO_VERSION} mojo; else \
   echo "Warning: MODULAR_AUTH_TOKEN is not set, not installing Mojo"; fi
ENV MODULAR_HOME="/root/.modular"
ENV PATH="/root/.modular/pkg/packages.modular.com_mojo/bin:$PATH"

# Clean apt cache
RUN apt-get autoremove -yq \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY src/ /app

ENTRYPOINT ["/bin/bash", "-c", "/app/run_benchmark.sh"]
