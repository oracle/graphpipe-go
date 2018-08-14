# graphpipe-tf - Serve TensorFlow Models via Graphpipe

The headlines are true! You can serve your caffe2/ONNX models via graphpipe
easily using this server.

If all you want to do is deploy models using GraphPipe, we recommend you read 
our [project documentation](https://oracle.github.io/graphpipe/).  If you are
interested in hacking on `graphpipe-onnx`, read on.

## Development Quickstart
Because of the relative complexity of system configuration when dealing with machine
learning libraries, our dev and build system for graphpipe-tf is 100% docker-driven.

Our build system can output images in three configurations:

* *cpu* (default) - create an Ubuntu-based build for cpu inference.  In this configuration, the BLAS backend is MKL.
* *oraclelinux-cpu* - same as cpu, but using oraclelinux as a base image.
* *gpu* - create an Ubuntu-based build for gpu inference.  If no physical gpu is present, inference falls back to 
  MKL cpu inference

You can switch between these configurations by setting the *RUN_TYPE* environment variable.

```
    > export RUN_TYPE=gpu
```

In order to support streamlined development and deployment, each build configuration
has 2 containers: one for development, and one for deployment.
```
    > make dev-container # creates the base dev-container
    > make in-docker # compiles the server inside the dev-container
    > make runtime-container # compiles the runtime-container and injects build artifacts
```

Additionally, you can build all three of these steps at the same time:
```
    > make all
```

During development, it is usually sufficient to run the server from the development image.
An example invocation of a development server can be invoked like this:
```
    > make devserver  # observe the docker command that is output, and tweak it for your own testing
```

Similarly, you can invoke a test instance of the deployment
```
    > make runserver  # observe the docker command that is output, and tweak it for your own testing
```

If things seem broken, try dropping into a shell in your dev-container to figure things out:

```
    > make devshell
```

## Proxies
If you are behind a proxy, set the *http_proxy* and *https_proxy* environment variables so our build system
can forward this configuration to docker.

## Running the server
The graphpipe-onnx binary has the following options:

```
  Required Flags for ONNX Models:
    -m, --model string          ONNX model to load.   Accepts local file or http(s) url.
        --value-inputs string   value_inputs.json for the model.  Accepts local file or http(s) url.

  Required Flags for Caffe2 Models:
        --init-net string       init_net file to load
        --predict-net string    predict_net file to load.  Accepts local file or http(s) url.
        --value-inputs string   value_inputs.json for the model.  Accepts local file or http(s) url.

  Optional Flags:
        --cache                 enable results caching
        --cache-dir string      directory for local cache state (default "~/.graphpipe")
        --disable-cuda          disable Cuda
        --engine-count int      number of caffe2 graph engines to create (default 1)
    -h, --help                  help for graphpipe-caffe2
    -l, --listen string         listen string (default "127.0.0.1:9000")
        --profile string        profile and write profiling output to this file
    -v, --verbose               enable verbose o
```

The exact flags you need depends on the type of model you have.  If you are invoking an onnx model, you need
to specify --model and --value-inputs:
```
./graphpipe-onnx --model=mymodel.onnx --value-inputs=value-inputs.json
```

For caffe2 models, you must specify 3 inputs, --init-net, --predict-net, and --value-inputs:
```
./graphpipe-onnx --init-net=my-init-net.pb --predict-net=my-predict-net.pb --value-inputs=value_inputs.json
```

## Environment Variables
For convenience, the key parameters of the service can be configured with environment variables,

```
    GP_OUTPUTS                comma seprated default inputs
    GP_INPUTS                 comma seprated default outputs
    GP_MODEL                  ONNX model to load.  Accepts local file or http(s) url.
    GP_CACHE                  enable results caching
    GP_INIT_NET               init_net file to load. Accepts local file or http(s) url.
    GP_PREDICT_NET            predict_net file to load. Accepts local file or http(s) url.
    GP_VALUE_INPUTS           value_inputs.json file to load. Accepts local file or http(s) url.
```


## Troubleshooting

### govendor can't fetch private libs
The in-docker setup _should_ forward `ssh-agent`s correctly if you
have it set up on your systems. Don't forget to `ssh-add` your key!

This link might be helpful: https://developer.github.com/v3/guides/using-ssh-agent-forwarding/

### proxies :(
Proxying should be forwarded for all our commands, but you may need
to configure your docker runtime to use them as well. Probably lives
(or needs to be created) at:

  `/etc/systemd/system/docker.service.d/http-proxy.conf`

