NAME=graphpipe-onnx
LIBCAFFE_IMAGE=libcaffe2
GRAPHPIPE_IMAGE=graphpipe-onnx
RUN_TYPE?=gpu

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
	#DOCKER=nvidia-docker
	LD_STUBS=LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$$LD_LIBRARY_PATH
endif

.PHONY: $(NAME)

$(NAME):
	go build -ldflags '-X "main.ver=$(VERSION)" -X "main.sha=$(sha)"'

in-docker:
	docker run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
		-w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-onnx \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		-e GOPATH=/go \
		-e CHOWN=$$(id -u):$$(id -g) \
		$(LIBCAFFE_IMAGE):$(RUN_TYPE)-dev \
		/usr/bin/make $(RUN_TYPE) graphpipe-onnx fix-perms

fix-perms:
	@if [ -n "$${CHOWN}" ]; then \
		chown -R $${CHOWN} vendor $(NAME) ../../vendor; \
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

shell:
	docker run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
		-w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-onnx \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		$(LIBCAFFE_IMAGE):$(RUN_TYPE)-dev \
		/bin/sh


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

all: $(NAME)

