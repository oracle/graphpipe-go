package graphpipe

import (
	"bytes"
	"fmt"
	"io/ioutil"
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

// MultiRemote is the complicated function for making a remote model request.
// It supports multiple inputs and outputs, custom clients, and config strings.
// If inputNames or outputNames is empty, it will use the default inputs and
// outputs from the server.  For multiple inputs and outputs, it is recommended
// that you Specify inputNames and outputNames so that you can control
// input/output ordering.  MultiRemote also performs type introspection for
// inputs and outputs.
func MultiRemote(client *http.Client, uri string, config string, ins []interface{}, inputNames, outputNames []string) ([]interface{}, error) {
	inputs := make([]*NativeTensor, len(ins))
	for i := range ins {
		var err error
		nt := &NativeTensor{}
		err = nt.InitSimple(ins[i])
		inputs[i] = nt
		if err != nil {
			logrus.Errorf("Failed to convert input %d: %v", i, err)
			return nil, err
		}
	}

	outputs, err := MultiRemoteRaw(client, uri, config, inputs, inputNames, outputNames)
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
// be converted into native go types.
func MultiRemoteRaw(client *http.Client, uri string, config string, inputs []*NativeTensor, inputNames, outputNames []string) ([]*NativeTensor, error) {
	b := fb.NewBuilder(1024)

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

	buf := Serialize(b, requestOffset)

	rq, err := http.NewRequest("POST", uri, bytes.NewReader(buf))
	if err != nil {
		logrus.Errorf("Failed to create request: %v", err)
		return nil, err
	}

	// send the request
	rs, err := client.Do(rq)
	if err != nil {
		logrus.Errorf("Failed to send request: %v", err)
		return nil, err
	}
	defer rs.Body.Close()

	body, err := ioutil.ReadAll(rs.Body)
	if err != nil {
		logrus.Errorf("Failed to read body: %v", err)
		return nil, err
	}
	if rs.StatusCode != 200 {
		return nil, fmt.Errorf("Remote failed with %d: %s", rs.StatusCode, string(body))
	}

	res := graphpipefb.GetRootAsInferResponse(body, 0)

	rval := make([]*NativeTensor, res.OutputTensorsLength())

	for i := 0; i < res.OutputTensorsLength(); i++ {
		tensor := &graphpipefb.Tensor{}
		if !res.OutputTensors(tensor, i) {
			err := fmt.Errorf("Bad input tensor")
			return nil, err
		}
		nt := TensorToNativeTensor(tensor)
		rval[i] = nt
	}

	return rval, nil
}
