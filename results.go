package graphpipe

import (
	"fmt"

	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

func getOutputNames(c *appContext, req *graphpipefb.InferRequest) ([]string, error) {
	if req.OutputNamesLength() == 0 {
		if len(c.defaultOutputs) == 0 {
			return nil, fmt.Errorf("no default outputs available -  please specify one or more outputs")
		}
		return c.defaultOutputs, nil
	}
	outputNames := make([]string, req.OutputNamesLength())
	for i := 0; i < req.OutputNamesLength(); i++ {
		name := string(req.OutputNames(i))
		if name == "" {
			return nil, fmt.Errorf("Could not init output names")
		}
		outputNames[i] = name
	}
	return outputNames, nil
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
	outputNames, err := getOutputNames(c, req)
	if err != nil {
		return nil, err
	}
	return c.apply(requestContext, string(req.Config()), inputMap, outputNames)
}
