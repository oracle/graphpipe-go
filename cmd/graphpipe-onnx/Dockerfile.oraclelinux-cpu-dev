FROM oraclelinux:7-slim

RUN yum install --enablerepo ol7_optional_latest -y \
    tar \
    gzip \
    make \
    git \
    curl \
    golang \
    protobuf-devel \
    hdf5-devel \
    cmake \
    && rm -rf /var/cache/yum/*

RUN rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
RUN yum install -y intel-mkl-2018.2-046

ENV GOPATH=/go

RUN mkdir -p /go/src

RUN git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch
WORKDIR /pytorch
RUN git checkout v0.4.1
RUN git submodule update --init --recursive
RUN mkdir build
WORKDIR /pytorch/build

RUN yum install --enablerepo ol7_developer_EPEL -y \
    cmake3 \
    gcc-c++ \
    gflags-devel \
    glog-devel

RUN echo "/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/" > /etc/ld.so.conf.d/intel_mkl.conf
RUN ldconfig

RUN cmake3 .. -DBLAS="MKL" -DBUILD_PYTHON=OFF
RUN make -j `nproc` install
RUN ldconfig

ENV LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/:/usr/local/lib:/usr/lib64/:/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/
