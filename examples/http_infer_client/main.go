package main

import (
	"fmt"
	graphpipe "github.com/oracle/graphpipe-go"
)

var addr = "http://127.0.0.1:9000"

/// Example of a client that talks to graphpipe-echo
func main() {
	v := []int64{2, 2}
	output, err := graphpipe.Remote(addr, v)

	if err != nil {
		fmt.Printf("Failed to call remote: %v", err)
	} else {
		echoData := output.([]int64)
		fmt.Println(echoData)
	}
}
