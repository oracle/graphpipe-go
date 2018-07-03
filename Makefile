.PHONY: install-govendor deps all

deps: govendor
	$(GOPATH)/bin/govendor sync

install-govendor:
	go get -u github.com/kardianos/govendor

govendor:
	@if [ ! -e $(GOPATH)/bin/govendor ]; then \
		echo "You need govendor: go get -u github.com/kardianos/govendor" && exit 1; \
	fi

all: deps
