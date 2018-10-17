/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/

package graphpipe

import (
	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

// NativeIOMetadata holds information describing the format of
// the model being served.
type NativeIOMetadata struct {
	Name        string
	Description string
	Shape       []int64
	Type        uint8
}

// NativeMetadataResponse is the response format used by the
// server.
type NativeMetadataResponse struct {
	Name        string
	Version     string
	Server      string
	Description string
	Inputs      []NativeIOMetadata
	Outputs     []NativeIOMetadata
}

// Build does all the heavy lifting of building out flatbuffers.
func (meta *NativeMetadataResponse) Build(b *fb.Builder) fb.UOffsetT {
	inputOffsets := make([]fb.UOffsetT, len(meta.Inputs))
	for i := 0; i < len(meta.Inputs); i++ {
		io := meta.Inputs[i]
		graphpipefb.IOMetadataStartShapeVector(b, len(io.Shape))
		for j := len(io.Shape) - 1; j >= 0; j-- {
			b.PrependInt64(io.Shape[j])
		}
		endShape := b.EndVector(len(io.Shape))
		name := b.CreateString(io.Name)
		desc := b.CreateString(io.Description)
		graphpipefb.IOMetadataStart(b)
		graphpipefb.IOMetadataAddShape(b, endShape)
		graphpipefb.IOMetadataAddName(b, name)
		graphpipefb.IOMetadataAddType(b, io.Type)
		graphpipefb.IOMetadataAddDescription(b, desc)
		inputOffsets[i] = graphpipefb.IOMetadataEnd(b)
	}
	outputOffsets := make([]fb.UOffsetT, len(meta.Outputs))

	for i := 0; i < len(meta.Outputs); i++ {
		io := meta.Outputs[i]
		graphpipefb.IOMetadataStartShapeVector(b, len(io.Shape))
		for j := len(io.Shape) - 1; j >= 0; j-- {
			b.PrependInt64(io.Shape[j])
		}
		endShape := b.EndVector(len(io.Shape))
		name := b.CreateString(io.Name)
		desc := b.CreateString(io.Description)
		graphpipefb.IOMetadataStart(b)
		graphpipefb.IOMetadataAddShape(b, endShape)
		graphpipefb.IOMetadataAddName(b, name)
		graphpipefb.IOMetadataAddType(b, io.Type)
		graphpipefb.IOMetadataAddDescription(b, desc)
		outputOffsets[i] = graphpipefb.IOMetadataEnd(b)
	}

	graphpipefb.MetadataResponseStartInputsVector(b, len(meta.Inputs))
	for i := len(meta.Inputs) - 1; i >= 0; i-- {
		b.PrependUOffsetT(inputOffsets[i])
	}
	inputs := b.EndVector(len(meta.Inputs))

	graphpipefb.MetadataResponseStartOutputsVector(b, len(meta.Outputs))
	for i := len(meta.Outputs) - 1; i >= 0; i-- {
		b.PrependUOffsetT(outputOffsets[i])
	}
	outputs := b.EndVector(len(meta.Outputs))

	desc := b.CreateString(meta.Description)
	version := b.CreateString(meta.Version)
	server := b.CreateString(meta.Server)
	name := b.CreateString(meta.Name)
	graphpipefb.MetadataResponseStart(b)
	graphpipefb.MetadataResponseAddDescription(b, desc)
	graphpipefb.MetadataResponseAddVersion(b, version)
	graphpipefb.MetadataResponseAddName(b, name)
	graphpipefb.MetadataResponseAddServer(b, server)
	graphpipefb.MetadataResponseAddInputs(b, inputs)
	graphpipefb.MetadataResponseAddOutputs(b, outputs)
	return graphpipefb.MetadataResponseEnd(b)
}
