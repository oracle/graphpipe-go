/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/

package graphpipe

import (
	"fmt"
	"net/http"

	"github.com/Sirupsen/logrus"
	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

// Remote is the simple function for making a remote model request with a
// single input and output and no config.  It performs introspection and
// automatic type conversion on its input and output.  It will use the server
// defaults for input and output.
func Remote(uri string, in interface{}) (interface{}, error) {
	res, err := MultiRemote(http.DefaultClient, uri, "", []interface{}{in}, nil, nil)
	if len(res) != 1 {
		return nil, fmt.Errorf("%d outputs were returned - one was expected", len(res))
	}
	return res[0], err
}

// MultiRemote is the complicated function for making a remote model request. It
// supports multiple inputs and outputs, custom http clients, and config
// strings. If inputNames or outputNames is empty, it will use the default
// inputs and outputs from the server.  For multiple inputs and outputs, it is
// recommended that you Specify inputNames and outputNames so that you can
// control input/output ordering.  MultiRemote also performs type introspection
// for inputs and outputs.
func MultiRemote(httpClient *http.Client, uri string, config string, ins []interface{}, inputNames, outputNames []string) ([]interface{}, error) {
	inputs := make([]*NativeTensor, len(ins))
	for i := range ins {
		nt := &NativeTensor{}
		err := nt.InitSimple(ins[i])
		inputs[i] = nt
		if err != nil {
			logrus.Errorf("Failed to convert input %d: %v", i, err)
			return nil, err
		}
	}

	client := HttpClient {
		NetHttpClient: httpClient,
		Uri: uri,
	}	
	outputs, err := MultiRemoteRaw(client, config, inputs, inputNames, outputNames)
	if err != nil {
		logrus.Errorf("Failed to MultiRemoteRaw: %v", err)
		return nil, err
	}
	natives := make([]interface{}, len(outputs))
	for i := range outputs {
		var err error
		natives[i], err = NativeTensorToNative(outputs[i])
		if err != nil {
			logrus.Errorf("Failed to convert output %d: %v", i, err)
			return nil, err
		}
	}
	return natives, nil
}

// MultiRemoteRaw is the actual implementation of the remote model
// request using NativeTensor objects. The raw call is provided
// for requests that need optimal performance and do not need to
// be converted into native go types. It also allows for other types of
// Clients.
func MultiRemoteRaw(client Client, config string, inputs []*NativeTensor, inputNames, outputNames []string) ([]*NativeTensor, error) {
	b := client.builder()
	buf := buildInferRequest(b, config, inputs, inputNames, outputNames)
	body, err := client.call(b, buf)
	if err != nil {
		return nil, err
	}
	
	res := graphpipefb.GetRootAsInferResponse(body, 0)

	rval := make([]*NativeTensor, res.OutputTensorsLength())
	for i := 0; i < res.OutputTensorsLength(); i++ {
		tensor := &graphpipefb.Tensor{}
		if !res.OutputTensors(tensor, i) {
			err := fmt.Errorf("Bad output tensor #%d", i)
			return nil, err
		}
		rval[i] = TensorToNativeTensor(tensor)
	}
	
	return rval, nil
}

func buildInferRequest(b *fb.Builder, config string, inputs[]*NativeTensor, inputNames, outputNames []string) []byte {
	inStrs := make([]fb.UOffsetT, len(inputNames))
	outStrs := make([]fb.UOffsetT, len(outputNames))

	for i := range inStrs {
		inStrs[i] = b.CreateString(inputNames[i])
	}

	for i := range outStrs {
		outStrs[i] = b.CreateString(outputNames[i])
	}

	graphpipefb.InferRequestStartInputNamesVector(b, len(inStrs))
	for i := len(inStrs) - 1; i >= 0; i-- {
		offset := inStrs[i]
		b.PrependUOffsetT(offset)
	}

	inputNamesOffset := b.EndVector(len(inStrs))

	graphpipefb.InferRequestStartOutputNamesVector(b, len(outStrs))
	for i := len(outStrs) - 1; i >= 0; i-- {
		offset := outStrs[i]
		b.PrependUOffsetT(offset)
	}
	outputNamesOffset := b.EndVector(len(outStrs))

	inputOffsets := make([]fb.UOffsetT, len(inputs))
	for i := 0; i < len(inputs); i++ {
		inputOffsets[i] = inputs[i].Build(b)
	}

	graphpipefb.InferRequestStartInputTensorsVector(b, len(inputs))
	for i := len(inputOffsets) - 1; i >= 0; i-- {
		offset := inputOffsets[i]
		b.PrependUOffsetT(offset)
	}
	inputTensors := b.EndVector(len(inputs))

	configString := b.CreateString(config)

	graphpipefb.InferRequestStart(b)
	graphpipefb.InferRequestAddInputNames(b, inputNamesOffset)
	graphpipefb.InferRequestAddOutputNames(b, outputNamesOffset)
	graphpipefb.InferRequestAddInputTensors(b, inputTensors)
	graphpipefb.InferRequestAddConfig(b, configString)
	inferRequestOffset := graphpipefb.InferRequestEnd(b)
	graphpipefb.RequestStart(b)
	graphpipefb.RequestAddReqType(b, graphpipefb.ReqInferRequest)
	graphpipefb.RequestAddReq(b, inferRequestOffset)
	requestOffset := graphpipefb.RequestEnd(b)

	return Serialize(b, requestOffset)
}

