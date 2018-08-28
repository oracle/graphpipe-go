# graphpipe-go - GraphPipe for go

[![wercker status](https://app.wercker.com/status/4c0651ec25ddf2f9a5c5cd6a3727265b/s/master "wercker status")](https://app.wercker.com/project/byKey/4c0651ec25ddf2f9a5c5cd6a3727265b) [![License: Unlicense](https://img.shields.io/badge/license-UPL-blue.svg)](https://opensource.org/licenses/UPL)

`graphpipe-go` provides a variety of functions to help you easily serve
and access ml models using the very speedy [`GraphPipe`](https://oracle.github.io/graphpipe/) protocol.

Additionally, this package provides reference server implementations
for common ML model formats, including:

* [ONNX and Caffe2 Model Server](https://github.com/oracle/graphpipe-go/tree/master/cmd/graphpipe-onnx)
* [Tensorflow Model Server](https://github.com/oracle/graphpipe-go/tree/master/cmd/graphpipe-tf)

For an overview of GraphPipe, read our [project documentation](https://oracle.github.io/graphpipe/)

If you are interested in learning a bit more about how the go servers and
clients work, read on:

## Client API
Most users integrating GraphPipe into their application will be interested
making client calls.  To make remote calls, we provide three different APIs

### `Remote`

```
// Remote is the simple function for making a remote model request with a
// single input and output and no config.  It performs introspection and
// automatic type conversion on its input and output.  It will use the server
// defaults for input and output.
func Remote(uri string, in interface{}) (interface{}, error)
```

### `MultiRemote`

```
// MultiRemote is the complicated function for making a remote model request.
// It supports multiple inputs and outputs, custom clients, and config strings.
// If inputNames or outputNames is empty, it will use the default inputs and
// outputs from the server.  For multiple inputs and outputs, it is recommended
// that you Specify inputNames and outputNames so that you can control
// input/output ordering.  MultiRemote also performs type introspection for
// inputs and outputs.
func MultiRemote(client *http.Client, uri string, config string, ins []interface{}, inputNames, outputNames []string) ([]interface{}, error)
```

### `MultiRemoteRaw`
```
// MultiRemoteRaw is the actual implementation of the remote model
// request using NativeTensor objects. The raw call is provided
// for requests that need optimal performance and do not need to
// be converted into native go types.
func MultiRemoteRaw(client *http.Client, uri string, config string, inputs []*NativeTensor, inputNames, outputNames []string) ([]*NativeTensor, error) {
```
In similar fashion to the serving model, the client for making remote
calls is made up of three functions, Remote, MultiRemote, and
MultiRemoteRaw.

The functions range from simple to complex. The first three of those will
convert your native Go types into tensors and back, while the last one uses
`graphpipe` tensors throughout.

## Model Serving API

There are two Serve functions, Serve and ServeRaw, that both create
standard Go http listeners and support caching with BoltDB.

### Serve

For applications where the manipulation of tensors will mostly be in go, Serve
provides a wrapper to your `apply` function that allows you to work with
standard go types rather than explicit `graphpipe` data objects and converts
between them for you. From the Go docs:

```
// Serve offers multiple inputs and outputs and converts tensors
// into native datatypes based on the shapes passed in to this function
// plus any additional shapes implied by your apply function.
// If cache is true, will attempt to cache using cache.db as cacheFile
func Serve(listen string, cache bool, apply interface{}, inShapes, outShapes [][]int64) error {}
```

As an example, here is a simple way to construct a graphpipe identity server,
which can receive a graphpipe network request, and echo it back to the client:

```
package main

import (
    "github.com/Sirupsen/logrus"
    graphpipe "github.com/oracle/graphpipe-go"
)

func main() {
    logrus.SetLevel(logrus.InfoLevel)
    useCache := false           // toggle caching on/off
    inShapes := [][]int64(nil)  // Optionally set input shapes
    outShapes := [][]int64(nil) // Optionally set output shapes
    if err := graphpipe.Serve("0.0.0.0:9000", useCache, apply, inShapes, outShapes); err != nil {
        logrus.Errorf("Failed to serve: %v", err)
    }
}

func apply(requestContext *graphpipe.RequestContext, ignore string, in interface{}) interface{} {
    return in // using the graphpipe.Serve interface, graphpipe automatically converts
              // go native types to tensors.
}
```

You can find a docker-buildable example of this server [here](https://github.com/oracle/graphpipe-go/tree/master/cmd/graphpipe-echo).

### ServeRaw

For applications that will be passing the tensors directly to another system
for processing and don't need conversion to standard Go types, ServeRaw
provides a lower-level interface.  For examples of apps that use ServeRaw, see
[graphpipe-tf](https://github.com/oracle/graphpipe-go/tree/master/cmd/graphpipe-tf)
and
[graphpipe-onnx](https://github.com/oracle/graphpipe-go/tree/master/cmd/graphpipe-onnx).

As you might expect, Serve uses ServeRaw underneath the hood.
