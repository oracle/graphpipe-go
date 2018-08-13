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
	return in // using the graphpipe.Serve interface, graphpipe automatically converts go native types to tensors.
}
