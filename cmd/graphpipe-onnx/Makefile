NAME=graphpipe-onnx
LIBCAFFE_IMAGE=libcaffe2
GRAPHPIPE_IMAGE=graphpipe-onnx
RUN_TYPE?=cpu
DOCKER?=docker
ONNX_URL?=https://objectstorage.us-phoenix-1.oraclecloud.com/n/bmcskeppareuser/b/c4/o/squeezenet.onnx
ONNX_VALUE_INPUTS_URL?=https://objectstorage.us-phoenix-1.oraclecloud.com/n/bmcskeppareuser/b/c4/o/squeezenet.value_inputs.json

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

ifeq ('$(RUN_TYPE)', 'gpu')
LD_STUBS=LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$$LD_LIBRARY_PATH
GP_ENABLE_CUDA=1
CGO_LDFLAGS=/usr/local/cuda/lib64/libcurand.so /usr/local/lib/libcaffe2_gpu.so /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so
CGO_CXXFLAGS=-DGP_ENABLE_CUDA
DOCKER=nvidia-docker
endif

.PHONY: $(NAME)

in-docker:
	docker inspect --type=image $(LIBTF_IMAGE):$(RUN_TYPE)-dev > /dev/null ||  make dev-container
	docker run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
		-w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-onnx \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		-e "CGO_LDFLAGS=$(CGO_LDFLAGS)" \
		-e "CGO_CXXFLAGS=$(CGO_CXXFLAGS)" \
		-e "CGO_LDFLAGS_ALLOW=.*" \
		-e GOPATH=/go \
		-e CHOWN=$$(id -u):$$(id -g) \
		$(LIBCAFFE_IMAGE):$(RUN_TYPE)-dev \
		/usr/bin/make $(RUN_TYPE) install-govendor deps graphpipe-onnx copy-files-$(RUN_TYPE) fix-perms


$(NAME):
	go build -ldflags '-X "main.ver=$(VERSION)" -X "main.sha=$(sha)"'

local: $(NAME)

devshell:
	$(DOCKER) run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
		-w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-onnx \
		-p 9000:9000 \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		-e GOPATH=/go \
		$(LIBCAFFE_IMAGE):$(RUN_TYPE)-dev \
		/bin/bash

devserver:
	$(DOCKER) run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
		-w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-onnx \
		-p 9000:9000 \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		-e GOPATH=/go \
		$(LIBCAFFE_IMAGE):$(RUN_TYPE)-dev \
		./graphpipe-onnx --model=$(ONNX_URL) --value-inputs=$(ONNX_VALUE_INPUTS_URL) --listen=0.0.0.0:9000

runserver:
	$(DOCKER) run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
		-p 9000:9000 \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		-e GOPATH=/go \
		$(GRAPHPIPE_IMAGE):$(RUN_TYPE) \
		--model=$(ONNX_URL) --value-inputs=$(ONNX_VALUE_INPUTS_URL) --listen=0.0.0.0:9000

copy-files-cpu:
	mkdir -p lib
	cp /pytorch/build/lib/libcaffe2.so lib/

copy-files-gpu:
	mkdir -p lib
	cp /pytorch/build/lib/libcaffe2.so lib/
	cp /pytorch/build/lib/libcaffe2_gpu.so lib/

copy-files-oraclelinux-cpu:
	mkdir -p lib
	cp /pytorch/build/lib/libcaffe2.so lib/

fix-perms:
	@if [ -n "$${CHOWN}" ]; then \
		chown -R $${CHOWN} lib/libca* vendor $(NAME) ../../vendor; \
	fi

cpu:
	# pass

oraclelinux-cpu:
	# pass

gpu:
	ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

dev-container:
	docker build \
		-f Dockerfile.$(RUN_TYPE)-dev \
		-t $(LIBCAFFE_IMAGE):$(RUN_TYPE)-dev \
		--build-arg http_proxy=$(http_proxy) \
		--build-arg https_proxy=$(https_proxy) \
		.

runtime-container:
	docker build \
		-f Dockerfile.$(RUN_TYPE) \
		-t $(GRAPHPIPE_IMAGE):$(RUN_TYPE) \
		--build-arg http_proxy=$(http_proxy) \
		--build-arg https_proxy=$(https_proxy) \
		.

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

deps: graphpipe-go-deps go-deps

all: dev-container in-docker runtime-container
