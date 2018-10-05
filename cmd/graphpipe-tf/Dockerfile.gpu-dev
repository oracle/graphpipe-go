FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:gophers/archive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    patch \
    golang-1.10-go \
    python-dev

RUN curl http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub | apt-key add -
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    libnccl2=2.3.5-2+cuda9.0 \
    libnccl-dev=2.3.5-2+cuda9.0


RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg |  apt-key add -
RUN apt-get update && apt-get install -y bazel

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/go-1.10/bin:/go/bin

ENV GOPATH=/go
RUN mkdir -p /go/src
RUN go get -u github.com/kardianos/govendor

RUN git clone https://github.com/tensorflow/tensorflow /tensorflow
WORKDIR /tensorflow
RUN git pull && git checkout v1.11.0

ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1
ENV TF_NEED_MKL 1
ENV TF_DOWNLOAD_MKL 1

RUN ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/libnccl.so.2
ENV NCCL_INSTALL_PATH="/usr/"

RUN tensorflow/tools/ci_build/builds/configured GPU
RUN bazel build --config=mkl --config=cuda --config=monolithic //tensorflow:libtensorflow.so
RUN cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
RUN cp /root/.cache/bazel/_bazel_root/*/external/mkl_linux/lib/* /usr/local/lib
RUN ldconfig
RUN ln -s /tensorflow/tensorflow /usr/local/include/tensorflow
