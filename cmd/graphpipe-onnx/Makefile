NAME=graphpipe-onnx

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

$(NAME): deps
	go build -ldflags '-X "main.ver=$(VERSION)" -X "main.sha=$(sha)"'

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

