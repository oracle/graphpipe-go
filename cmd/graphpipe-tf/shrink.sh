#!/usr/bin/env bash

smith -i slim.tar.gz
smith upload -d -v -r 'https://odxml:26467dafb0f2cae81e98aff21a291e0a754ef528616ba6f1adbe03cb8c333ce3@wcr.io/odxml/tureen-tensorflow:latest' -i slim.tar.gz
