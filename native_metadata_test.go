package graphpipe

import (
	"testing"

	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

func createIOMeta(name string) NativeIOMetadata {
	resp := NativeIOMetadata{}
	resp.Name = name
	resp.Description = name
	resp.Type = graphpipefb.TypeFloat32
	resp.Shape = []int64{1, 2}
	return resp

}

func TestNativeMetadataResponse(t *testing.T) {
	resp := &NativeMetadataResponse{}
	resp.Name = "name"
	resp.Version = "0.01"
	resp.Server = "servy"
	resp.Description = "mydesc"
	resp.Inputs = append(resp.Inputs, createIOMeta("input0"))
	resp.Inputs = append(resp.Inputs, createIOMeta("input1"))
	resp.Outputs = append(resp.Outputs, createIOMeta("output0"))
	resp.Outputs = append(resp.Outputs, createIOMeta("output1"))
	b := fb.NewBuilder(1024)
	resp.Build(b)
}
