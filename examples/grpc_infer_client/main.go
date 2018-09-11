package main

import (
	"fmt"
	fb "github.com/google/flatbuffers/go"
	graphpipe "github.com/oracle/graphpipe-go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

var addr = "127.0.0.1:9000"

/// Example of a client that talks to graphpipe-echo
func main() {
	inputTensors := []*graphpipe.NativeTensor{}

	nt := &graphpipe.NativeTensor{}
	v := []int64{2, 2}
	err := nt.InitSimple(v)
	inputTensors = append(inputTensors, nt)
	b := graphpipe.BuildInferRequest("", inputTensors, nil, nil)

	conn, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithCodec(fb.FlatbuffersCodec{}))
	if err != nil {
		fmt.Printf("Failed to connect: %v", err)
	}
	defer conn.Close()
	client := graphpipefb.NewGraphpipeServiceClient(conn)
	inferResponse, err := client.Infer(context.Background(), b)
	outputTensors := graphpipe.ParseInferResponse(inferResponse)
	fmt.Println(outputTensors[0].Type)
}
