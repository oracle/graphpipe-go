package main

import (
	"fmt"
	graphpipe "github.com/oracle/graphpipe-go"
)

var addr = "grpc+http://127.0.0.1:9000"

/// Example of a client that requests metadata from a graphpipe server
func main() {
	meta, err := graphpipe.Metadata(addr)

	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println(meta)
	}
}
