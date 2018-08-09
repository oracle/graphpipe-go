NAME=graphpipe-tf
RUN_TYPE?=cpu
TF_DIR=/usr/local
SUDO=sudo
LIBTF_IMAGE=libtensorflow
GRAPHPIPE_TF_IMAGE=graphpipe-tf
DOCKER=docker

ifeq ('$(USE_SUDO)', '0')
	SUDO=
endif

ifeq ('$(RUN_TYPE)', 'gpu')
	DOCKER=nvidia-docker
	LD_STUBS=LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$$LD_LIBRARY_PATH
endif

sha = $(shell git rev-parse --short HEAD | tr -d ' \n')
ifeq ($(VERSION),)
VERSION = $(shell git describe --tags --match 'v*.*.*' 2> /dev/null  | tr -d 'v \n')
realv = $(shell echo $(VERSION) | cut -d- -f1)
ifneq ($(VERSION),$(realv))
commits = $(shell echo $(VERSION) | cut -d- -f2)
VERSION := $(realv).$(commits).$(sha)
endif
endif
dirty = $(shell git diff --shortstat 2> /dev/null | tail -n1 | tr -d ' \n')
ifneq ($(dirty),)
VERSION := $(VERSION).dev
endif

.PHONY: $(NAME)

$(NAME):
	$(LD_STUBS) go build -ldflags '-X "main.ver=$(VERSION)" -X "main.sha=$(sha)"'

in-docker:
	docker run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
		-w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-tf \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		-v $$(dirname $(SSH_AUTH_SOCK)):$$(dirname $(SSH_AUTH_SOCK)) \
		-e SSH_AUTH_SOCK=$(SSH_AUTH_SOCK) \
		-e USE_SUDO=0 \
		-e GOPATH=/go \
		-e CHOWN=$$(id -u):$$(id -g) \
		$(LIBTF_IMAGE):$(RUN_TYPE)-dev \
		/usr/bin/make install-govendor deps $(RUN_TYPE) copy-files-$(RUN_TYPE) graphpipe-tf fix-perms

cpu:
	# pass

copy-files-cpu:
	mkdir -p lib
	cp /root/.cache/bazel/_bazel_root/*/external/mkl_linux/lib/* lib/
	cp /usr/local/lib/libtensorflow* lib/

oraclelinux-cpu:
	# pass

copy-files-oraclelinux-cpu:
	mkdir -p lib
	cp /root/.cache/bazel/_bazel_root/*/external/mkl_linux/lib/* lib/
	cp /usr/local/lib/libtensorflow* lib/

gpu:
	ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

copy-files-gpu:
	# pass

devshell:
	docker run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
        -w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-go \
        -p 9000:9000 \
        -e http_proxy=$(http_proxy) \
        -e https_proxy=$(https_proxy) \
        -e GOPATH=/go \
        -e CHOWN=$$(id -u):$$(id -g) \
        $(LIBTF_IMAGE):$(RUN_TYPE)-dev \
        /bin/bash

dev-container:
	docker build \
		-f Dockerfile.$(RUN_TYPE)-dev \
		-t $(LIBTF_IMAGE):$(RUN_TYPE)-dev \
		--build-arg http_proxy=$(http_proxy) \
		--build-arg https_proxy=$(https_proxy) \
		.

devserver:
	$(DOCKER) run -it --rm \
        -v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
        -w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-tf \
        -e http_proxy=$(http_proxy) \
        -e https_proxy=$(https_proxy) \
        -p 9000:9000 \
        -e GOPATH=/go \
        -e CHOWN=$$(id -u):$$(id -g) \
        $(LIBTF_IMAGE):$(RUN_TYPE)-dev \
        ./graphpipe-tf --model=https://objectstorage.us-phoenix-1.oraclecloud.com/n/bmcskeppareuser/b/c4/o/squeezenet.pb \
		--listen=0.0.0.0:9000

runtime-container:
	docker build \
		-f Dockerfile.$(RUN_TYPE) \
		-t $(GRAPHPIPE_TF_IMAGE):$(RUN_TYPE) \
		--build-arg http_proxy=$(http_proxy) \
		--build-arg https_proxy=$(https_proxy) \
		.

runserver:
	$(DOCKER) run -it --rm \
        -e http_proxy=$(http_proxy) \
        -e https_proxy=$(https_proxy) \
        -p 9000:9000 \
        $(GRAPHPIPE_TF_IMAGE):$(RUN_TYPE) \
        --model=https://objectstorage.us-phoenix-1.oraclecloud.com/n/bmcskeppareuser/b/c4/o/squeezenet.pb \
		--listen=0.0.0.0:9000

install-go:
	apt install golang-go

install-govendor:
	@if [ ! -e $(GOPATH)/bin/govendor ]; then \
		go get -u github.com/kardianos/govendor; \
	fi

govendor:
	@if [ ! -e $(GOPATH)/bin/govendor ]; then \
		echo "You need govendor: go get -u github.com/kardianos/govendor" && exit 1; \
	fi

graphpipe-go-deps:
	cd ../../ && make deps

go-deps: govendor
	$(GOPATH)/bin/govendor sync -v

install-tensorflow:
	@if [ ! -e /usr/local/include/tensorflow/c/c_api.h ]; then \
		curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-$(RUN_TYPE)-linux-x86_64-1.8.0.tar.gz" | \
			$(SUDO) tar -C $(TF_DIR) -xz; \
	fi

tensorflow:
	@if [ ! -e /usr/local/include/tensorflow/c/c_api.h ]; then \
		echo "You need tensorflow installed: https://www.tensorflow.org/install/install_go" && exit 1; \
	fi

fix-perms:
	@if [ -n "$${CHOWN}" ]; then \
		chown -R $${CHOWN} lib vendor $(NAME) ../../vendor; \
	fi

PROTOC=protoc
TF=$(GOPATH)/src/github.com/tensorflow/tensorflow
PROTOS = $(TF)/tensorflow/core/protobuf/config.proto \
	$(TF)/tensorflow/core/protobuf/debug.proto \
	$(TF)/tensorflow/core/protobuf/cluster.proto \
	$(TF)/tensorflow/core/protobuf/rewriter_config.proto

protobuf:
	go get github.com/gogo/protobuf/proto
	go get github.com/gogo/protobuf/gogoproto
	go get github.com/tensorflow/tensorflow/tensorflow/go
	cd protoc-gen-gogotf && go install
	PATH=$(GOPATH)/bin:$(PATH) $(PROTOC) -I$(TF)/ $(TF)/tensorflow/core/framework/*.proto --gogotf_out=internal/
	PATH=$(GOPATH)/bin:$(PATH) $(PROTOC) -I$(TF)/ $(PROTOS) --gogotf_out=internal/
	sed -i 's;github.com/tensorflow;github.com/oracle/graphpipe-go/cmd/graphpipe-tf/internal/github.com/tensorflow;g' internal/github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/*.pb.go

deps: graphpipe-go-deps go-deps

all: $(NAME)
