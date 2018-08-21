FROM ubuntu:16.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:gophers/archive

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/go-1.10/bin:/go/bin
ENV GOPATH=/go

RUN mkdir -p /go/src

RUN apt-get update && apt-get install --fix-missing -y --no-install-recommends \
      linux-libc-dev \
      libavcodec-dev  \
      libavcodec-ffmpeg56 \
      manpages-dev \
      libopenmpi-dev \
      git \
      curl \
      golang-1.10-go \
      build-essential \
      cmake \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      protobuf-compiler \
      cmake \
      wget

RUN git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch
WORKDIR /pytorch
RUN git checkout v0.4.1
RUN git submodule update --init --recursive
RUN mkdir build
WORKDIR /pytorch/build

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN echo "deb https://apt.repos.intel.com/mkl all main" > /etc/apt/sources.list.d/intel-mkl.list
RUN apt-get install -y apt-transport-https
RUN apt-get update && apt-get install -y intel-mkl-64bit-2018.3-051
RUN echo "/opt/intel/compilers_and_libraries_2018.3.222/linux/mkl/lib/intel64_lin/" > /etc/ld.so.conf.d/intel_mkl.conf

RUN cmake .. -DBLAS="MKL" -DUSE_MKLDNN=1 -DBUILD_PYTHON=OFF
RUN mkdir -p /home/ml/projects/
RUN ln -s /pytorch /home/ml/projects/pytorch
RUN make -j `nproc` install
RUN ldconfig
