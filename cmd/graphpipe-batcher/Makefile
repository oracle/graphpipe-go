NAME=graphpipe-batcher
RUN_TYPE?=cpu
TF_DIR=/usr/local
SUDO=sudo
LIBECHO_IMAGE=graphpipe-batcher-builder
GRAPHPIPE_ECHO_IMAGE=graphpipe-batcher
DOCKER=docker

ifeq ('$(USE_SUDO)', '0')
	SUDO=
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
		-w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-batcher \
		-e http_proxy=$(http_proxy) \
		-e https_proxy=$(https_proxy) \
		-v $$(dirname $(SSH_AUTH_SOCK)):$$(dirname $(SSH_AUTH_SOCK)) \
		-e SSH_AUTH_SOCK=$(SSH_AUTH_SOCK) \
		-e USE_SUDO=0 \
		-e GOPATH=/go \
		-e CHOWN=$$(id -u):$$(id -g) \
		$(LIBECHO_IMAGE):$(RUN_TYPE)-dev \
		/usr/bin/make install-govendor deps $(RUN_TYPE) graphpipe-batcher fix-perms

cpu:
	# pass

oraclelinux-cpu:
	# pass

devshell:
	docker run -it --rm \
		-v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
        -w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-batcher \
        -p 10000:10000 \
        -e http_proxy=$(http_proxy) \
        -e https_proxy=$(https_proxy) \
        -e GOPATH=/go \
        -e CHOWN=$$(id -u):$$(id -g) \
        $(LIBECHO_IMAGE):$(RUN_TYPE)-dev \
        /bin/bash

dev-container:
	docker build \
		-f Dockerfile.$(RUN_TYPE)-dev \
		-t $(LIBECHO_IMAGE):$(RUN_TYPE)-dev \
		--build-arg http_proxy=$(http_proxy) \
		--build-arg https_proxy=$(https_proxy) \
		.

devserver:
	$(DOCKER) run -it --rm \
        -v $(PWD)/../../:/go/src/github.com/oracle/graphpipe-go \
        -w /go/src/github.com/oracle/graphpipe-go/cmd/graphpipe-batcher \
        -e http_proxy=$(http_proxy) \
        -e https_proxy=$(https_proxy) \
        -p 10000:10000 \
        -e GOPATH=/go \
        -e CHOWN=$$(id -u):$$(id -g) \
        $(LIBECHO_IMAGE):$(RUN_TYPE)-dev \
        ./graphpipe-batcher \
		--target-url=http://127.0.0.1:9000 \
		--inputs=input_1 \
		--outputs=loss/Softmax \
		--listen=0.0.0.0:10000

runtime-container:
	docker build \
		-f Dockerfile.$(RUN_TYPE) \
		-t $(GRAPHPIPE_ECHO_IMAGE):$(RUN_TYPE) \
		--build-arg http_proxy=$(http_proxy) \
		--build-arg https_proxy=$(https_proxy) \
		.

runserver:
	$(DOCKER) run -it --rm \
        -e http_proxy=$(http_proxy) \
        -e https_proxy=$(https_proxy) \
        -p 10000:10000 \
        $(GRAPHPIPE_ECHO_IMAGE):$(RUN_TYPE) \
		--target-url=http://127.0.0.1:9000 \
		--inputs=input_1 \
		--outputs=loss/Softmax \
		--listen=0.0.0.0:10000

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

fix-perms:
	@if [ -n "$${CHOWN}" ]; then \
		chown -R $${CHOWN} vendor $(NAME) ../../vendor; \
	fi

deps: graphpipe-go-deps go-deps

all: dev-container in-docker runtime-container
