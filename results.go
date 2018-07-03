package graphpipe

import (
	"fmt"

	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

func getOutputNames(req *graphpipefb.InferRequest) []string {
	outputNames := make([]string, req.OutputNamesLength())
	for i := 0; i < req.OutputNamesLength(); i++ {
		outputNames[i] = string(req.OutputNames(i))
	}
	return outputNames
}

func getInputMap(c *appContext, req *graphpipefb.InferRequest) (map[string]*NativeTensor, error) {
	inputMap := map[string]*NativeTensor{}
	for i := 0; i < req.InputTensorsLength(); i++ {
		name := ""
		if i < req.InputNamesLength() {
			name = string(req.InputNames(i))
		}
		if name == "" {
			name = c.defaultInputs[i]
		}
		tensor := &graphpipefb.Tensor{}
		if !req.InputTensors(tensor, i) {
			return nil, fmt.Errorf("Could not init tensor")
		}
		inputMap[name] = TensorToNativeTensor(tensor)
	}
	return inputMap, nil
}

func getResults(c *appContext, requestContext *RequestContext, req *graphpipefb.InferRequest) ([]*NativeTensor, error) {
	inputMap, err := getInputMap(c, req)
	if err != nil {
		return nil, err
	}
	outputNames := getOutputNames(req)
	return c.apply(requestContext, string(req.Config()), inputMap, outputNames)
}
