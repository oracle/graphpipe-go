FROM oraclelinux:7-slim

RUN yum install --enablerepo ol7_optional_latest -y tar gzip make git curl golang \
  && rm -rf /var/cache/yum/*

RUN curl https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo > /etc/yum.repos.d/vbatts-bazel-epel-7.repo 
RUN yum install -y bazel
RUN go get -u github.com/kardianos/govendor

RUN git clone https://github.com/tensorflow/tensorflow /tensorflow
WORKDIR /tensorflow
RUN git pull && git checkout v1.11.0

RUN yum install --enablerepo ol7_optional_latest -y which patch && rm -rf /var/cache/yum/*
RUN yum install --enablerepo ol7_optional_latest -y gcc-c++ && rm -rf /var/cache/yum/*

RUN yes "" | ./configure
RUN bazel build --config=mkl --config=monolithic //tensorflow:libtensorflow.so
RUN cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
RUN cp /root/.cache/bazel/_bazel_root/*/external/mkl_linux/lib/* /usr/local/lib
RUN ldconfig
RUN ln -s /tensorflow/tensorflow /usr/local/include/tensorflow

