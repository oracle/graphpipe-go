# graphpipe-echo - A simple tensor echo server written in golang

This is a bare-bones demonstration of how to create a bare-bones graphpipe
server using the graphpipe api.  It uses the same build conventions as
graphpipe-tf and graphpipe-onnx, so those looking to dive into hacking either
of those servers may find it helpful to start here.

Like the other cmd projects in this repo, building, testing, and deployment
are all handled in docker.

graphpipe-echo has 2 supported configurations:

* *cpu* (default) - create an Ubuntu-based build.
* *oraclelinux-cpu* - same as cpu, but using oraclelinux as a base image.


In order to support streamlined development and deployment, each build configuration
has 2 containers: one for development, and one for deployment.
```
    > make dev-container # creates the base dev container
    > make in-docker # compiles the server inside the docker image
    > make runtime-container # compiles the deployment container, and injects build artifacts into this container
```

Additionally, you can build all three of these steps at the same time:
```
    > make all
```

During development, it is usually sufficient to run the server from the development image.
A development server can be invoked like this:
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
