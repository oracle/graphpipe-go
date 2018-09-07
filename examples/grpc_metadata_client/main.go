package main

import (
	"fmt"
	fb "github.com/google/flatbuffers/go"
	graphpipe "github.com/oracle/graphpipe-go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

var addr = "127.0.0.1:9001"

/// Example of a client that requests metadata from a graphpipe server
func main() {
	b := graphpipe.BuildMetadataRequest()

	conn, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithCodec(fb.FlatbuffersCodec{}))
	if err != nil {
		fmt.Printf("Failed to connect: %v", err)
	}
	defer conn.Close()
	client := graphpipefb.NewGraphpipeServiceClient(conn)
	out, err := client.Metadata(context.Background(), b)
	response := graphpipe.ParseMetadataResponse(out)
	fmt.Println(response)
}
