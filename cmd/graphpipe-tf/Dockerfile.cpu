FROM ubuntu:16.04

ENV GOPATH=/go
RUN apt-get update && apt-get install -y ca-certificates libstdc++6
COPY graphpipe-tf /
COPY lib/libiomp5.so /usr/local/lib
COPY lib/libmklml_intel.so /usr/local/lib
COPY lib/libtensorflow.so /usr/local/lib

RUN ldconfig
ENTRYPOINT ["/graphpipe-tf"]
