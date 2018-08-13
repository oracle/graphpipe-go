FROM ubuntu:16.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:gophers/archive

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/go-1.10/bin:/go/bin
ENV GOPATH=/go

RUN mkdir -p /go/src

RUN apt-get update && apt-get install --fix-missing -y --no-install-recommends \
      golang-1.10-go

RUN apt-get update && apt-get install --fix-missing -y --no-install-recommends \
      make \
      git
