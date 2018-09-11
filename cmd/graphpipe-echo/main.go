package main

import (
	"os"

	"github.com/Sirupsen/logrus"
	graphpipe "github.com/oracle/graphpipe-go"
	"github.com/spf13/cobra"
)

type options struct {
	listen string
}

func runEchoServer(listen string) {
	useCache := false           // toggle caching on/off
	inShapes := [][]int64(nil)  // Optionally set input shapes
	outShapes := [][]int64(nil) // Optionally set output shapes
	if err := graphpipe.Serve(listen, useCache, apply, inShapes, outShapes); err != nil {
		logrus.Errorf("Failed to serve: %v", err)
	}
}

func main() {
	var opts options
	var cmdExitCode int
	logrus.SetLevel(logrus.InfoLevel)

	cmd := cobra.Command{
		Use:   "graphpipe-echo",
		Short: "graphpipe-echo - echoing ml requests",
		Run: func(cmd *cobra.Command, args []string) {
			runEchoServer(opts.listen)
		},
	}

	f := cmd.Flags()
	f.StringVarP(&opts.listen, "listen", "l", "127.0.0.1:9000", "listen string")

	cmd.Execute()
	os.Exit(cmdExitCode)
}

func apply(requestContext *graphpipe.RequestContext, ignore string, in interface{}) interface{} {
	return in // using the graphpipe.Serve interface, graphpipe automatically converts go native types to tensors.
}
