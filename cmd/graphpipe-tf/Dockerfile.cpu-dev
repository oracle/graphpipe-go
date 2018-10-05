FROM ubuntu:16.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

RUN apt-get update && apt install -y software-properties-common
RUN add-apt-repository ppa:gophers/archive
RUN apt-get update && apt-get install -y \
    linux-libc-dev \
    manpages-dev \
    python-dev \
    golang-1.10-go \
    git \
    curl \
    patch

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/go-1.10/bin:/go/bin

RUN go get -u github.com/kardianos/govendor
RUN git clone https://github.com/tensorflow/tensorflow /tensorflow
WORKDIR /tensorflow
RUN git pull && git checkout v1.11.0

RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg |  apt-key add -
RUN apt-get update &&  apt-get install -y bazel

RUN yes "" | ./configure

RUN bazel build --config=mkl --config=monolithic //tensorflow:libtensorflow.so
RUN cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
RUN cp /root/.cache/bazel/_bazel_root/*/external/mkl_linux/lib/* /usr/local/lib
RUN ldconfig
RUN ln -s /tensorflow/tensorflow /usr/local/include/tensorflow
