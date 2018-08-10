FROM oraclelinux:7-slim
ENV LD_LIBRARY_PATH=/usr/local/lib
COPY graphpipe-tf /
COPY lib/libiomp5.so /usr/local/lib
COPY lib/libmklml_intel.so /usr/local/lib
COPY lib/libtensorflow.so /usr/local/lib
ENTRYPOINT ["/graphpipe-tf"]
