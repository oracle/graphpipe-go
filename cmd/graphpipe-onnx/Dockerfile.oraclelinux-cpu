FROM oraclelinux:7-slim

RUN rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
RUN yum install -y intel-mkl-gnu-rt-2018.2-199-2018.2-199.x86_64
RUN yum install -y protobuf
RUN yum install --enablerepo ol7_developer_EPEL -y glog libgomp
ENV LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/:/usr/local/lib:/usr/lib64/:/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin/

COPY lib/libcaffe2.so /usr/local/lib/libcaffe2.so
COPY graphpipe-onnx /
ENTRYPOINT ["/graphpipe-onnx"]
