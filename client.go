package graphpipe

import (
	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

// BuildInferRequest constructs an InferRequest flatbuffer from NativeTensor
func BuildInferRequest(config string, inputTensors []*NativeTensor, inputs, outputs []string) *fb.Builder {
	b := fb.NewBuilder(1024)
	inStrs := make([]fb.UOffsetT, len(inputs))
	outStrs := make([]fb.UOffsetT, len(outputs))

	for i := range inputs {
		inStr := b.CreateString(inputs[i])
		inStrs[i] = inStr
	}
	for i := range outputs {
		outStr := b.CreateString(outputs[i])
		outStrs[i] = outStr
	}
	graphpipefb.InferRequestStartInputNamesVector(b, len(inputs))

	for _, offset := range inStrs {
		b.PrependUOffsetT(offset)
	}
	inputNamesOffset := b.EndVector(len(inputs))
	graphpipefb.InferRequestStartOutputNamesVector(b, len(outputs))
	for _, offset := range outStrs {
		b.PrependUOffsetT(offset)
	}
	outputNamesOffset := b.EndVector(len(outputs))

	inputOffsets := make([]fb.UOffsetT, len(inputTensors))
	for i := 0; i < len(inputTensors); i++ {
		tp := inputTensors[i]
		inputOffsets[i] = tp.Build(b)
	}

	graphpipefb.InferRequestStartInputTensorsVector(b, 1)
	for _, offset := range inputOffsets {
		b.PrependUOffsetT(offset)
	}
	inputTensorsOffset := b.EndVector(1)

	configString := b.CreateString(config)
	graphpipefb.InferRequestStart(b)
	graphpipefb.InferRequestAddInputNames(b, inputNamesOffset)
	graphpipefb.InferRequestAddOutputNames(b, outputNamesOffset)
	graphpipefb.InferRequestAddInputTensors(b, inputTensorsOffset)
	graphpipefb.InferRequestAddConfig(b, configString)
	inferRequestOffset := graphpipefb.InferRequestEnd(b)
	b.Finish(inferRequestOffset)
	return b
}

// ParseInferResponse constructs a NativeTensor from flatbuffer
func ParseInferResponse(inferResponse *graphpipefb.InferResponse) []*NativeTensor {
	tensors := []*NativeTensor{}

	for i := 0; i < inferResponse.OutputTensorsLength(); i++ {

		t := graphpipefb.Tensor{}
		inferResponse.OutputTensors(&t, i)
		shape := []int64{}
		for j := 0; j < t.ShapeLength(); j++ {
			shape = append(shape, t.Shape(j))
		}
		nt := &NativeTensor{}
		nt.InitWithData(t.DataBytes(), shape, t.Type())
		tensors = append(tensors, nt)

	}
	return tensors
}

// BuildMetadataRequest constructs flatbuffer from NativeMetadataRequest
func BuildMetadataRequest() *fb.Builder {
	b := fb.NewBuilder(0)
	graphpipefb.MetadataRequestStart(b)
	metaReq := graphpipefb.MetadataRequestEnd(b)

	graphpipefb.RequestStart(b)
	graphpipefb.RequestAddReq(b, metaReq)
	graphpipefb.RequestAddReqType(b, graphpipefb.ReqMetadataRequest)
	req := graphpipefb.RequestEnd(b)
	b.Finish(req)
	return b
}

func parseIO(io *graphpipefb.IOMetadata) NativeIOMetadata {
	nio := NativeIOMetadata{}
	nio.Name = string(io.Name())
	nio.Description = string(io.Description())
	nio.Type = io.Type()
	for i := 0; i < io.ShapeLength(); i++ {
		nio.Shape = append(nio.Shape, io.Shape(i))
	}
	return nio
}

// ParseMetadataResponse constructs a NativeMetadataRequest from flatbuffer
func ParseMetadataResponse(metadataResponse *graphpipefb.MetadataResponse) *NativeMetadataResponse {
	nm := &NativeMetadataResponse{}
	nm.Version = string(metadataResponse.Version())
	nm.Server = string(metadataResponse.Server())
	nm.Description = string(metadataResponse.Description())

	for i := 0; i < metadataResponse.InputsLength(); i++ {
		io := &graphpipefb.IOMetadata{}
		metadataResponse.Inputs(io, i)
		nm.Inputs = append(nm.Inputs, parseIO(io))
	}
	for i := 0; i < metadataResponse.OutputsLength(); i++ {
		io := &graphpipefb.IOMetadata{}
		metadataResponse.Outputs(io, i)
		nm.Outputs = append(nm.Outputs, parseIO(io))
	}
	return nm
}
