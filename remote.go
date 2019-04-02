/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/

package graphpipe

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"

	"github.com/Sirupsen/logrus"
	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

// Remote is the simple function for making a remote model request with a
// single input and output and no config.  It performs introspection and
// automatic type conversion on its input and output.  It will use the server
// defaults for input and output.
func Remote(uri string, in interface{}) (interface{}, error) {
	scheme, host, err := parseURL(uri)

	if err != nil {
		return nil, err
	}
	switch scheme {
	case "http":
		res, err := MultiRemote(http.DefaultClient, uri, "", []interface{}{in}, nil, nil)
		if len(res) != 1 {
			return nil, fmt.Errorf("%d outputs were returned - one was expected", len(res))
		}
		return res[0], err
	case "grpc+http":
		conn, err := getDefaultGRPCClient(host)
		if err != nil {
			fmt.Printf("Failed to connect: %v", err)
			return nil, err
		}
		defer conn.Close()
		client := graphpipefb.NewGraphpipeServiceClient(conn)
		res, err := MultiRemote(client, uri, "", []interface{}{in}, nil, nil)
		if len(res) != 1 {
			return nil, fmt.Errorf("%d outputs were returned - one was expected", len(res))
		}
		return res[0], err
	default:
		msg := fmt.Sprintf("Unhandled scheme: %s", scheme)
		logrus.Errorf(msg)
		return nil, errors.New(msg)
	}

}

func getDefaultGRPCClient(host string) (*grpc.ClientConn, error) {
	conn, err := grpc.Dial(host, grpc.WithInsecure(), grpc.WithCodec(fb.FlatbuffersCodec{}))
	if err != nil {
		fmt.Printf("Failed to connect: %v", err)
		return nil, err
	}
	return conn, nil
}

func grpcMetadata(host string) (*NativeMetadataResponse, error) {
	b, offset := BuildMetadataRequest()
	b.Finish(offset)
	conn, err := getDefaultGRPCClient(host)
	if err != nil {
		fmt.Printf("Failed to connect: %v", err)
		return nil, err
	}
	defer conn.Close()
	client := graphpipefb.NewGraphpipeServiceClient(conn)
	out, err := client.Metadata(context.Background(), b)
	response := ParseMetadataResponse(out)
	return response, nil
}

func httpMetadata(uri string) (*NativeMetadataResponse, error) {
	client := http.DefaultClient

	b, offset := BuildMetadataRequest()

	graphpipefb.RequestStart(b)
	graphpipefb.RequestAddReqType(b, graphpipefb.ReqMetadataRequest)
	graphpipefb.RequestAddReq(b, offset)
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
	res := graphpipefb.GetRootAsMetadataResponse(body, 0)

	return ParseMetadataResponse(res), err
}

// Metadata is used to fetch metadata from a remote server
func Metadata(uri string) (*NativeMetadataResponse, error) {
	scheme, host, err := parseURL(uri)
	if err != nil {
		return nil, err
	}

	switch scheme {
	case "http":
		return httpMetadata(uri)
	case "grpc+http":
		return grpcMetadata(host)
	default:
		msg := fmt.Sprintf("Unhandled scheme: %s", scheme)
		logrus.Errorf(msg)
		return nil, errors.New(msg)
	}
}

// MultiRemote is the complicated function for making a remote model request.
// It supports multiple inputs and outputs, custom clients, and config strings.
// If inputNames or outputNames is empty, it will use the default inputs and
// outputs from the server.  For multiple inputs and outputs, it is recommended
// that you Specify inputNames and outputNames so that you can control
// input/output ordering.  MultiRemote also performs type introspection for
// inputs and outputs.
func MultiRemote(clientInterface interface{}, uri string, config string, ins []interface{}, inputNames, outputNames []string) ([]interface{}, error) {
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

	outputs, err := MultiRemoteRaw(clientInterface, uri, config, inputs, inputNames, outputNames)
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
func MultiRemoteRaw(clientInterface interface{}, uri string, config string, inputs []*NativeTensor, inputNames, outputNames []string) ([]*NativeTensor, error) {
	scheme, _, err := parseURL(uri)
	if err != nil {
		return nil, err
	}
	b, inferRequestOffset := BuildInferRequest(config, inputs, inputNames, outputNames)
	switch scheme {
	case "http":
		client := clientInterface.(*http.Client)

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

		inferResponse := graphpipefb.GetRootAsInferResponse(body, 0)

		outputTensors := ParseInferResponse(inferResponse)

		return outputTensors, nil
	case "grpc+http":
		client := clientInterface.(graphpipefb.GraphpipeServiceClient)
		b.Finish(inferRequestOffset)
		inferResponse, err := client.Infer(context.Background(), b)
		if err != nil {
			return nil, err
		}
		outputTensors := ParseInferResponse(inferResponse)
		return outputTensors, nil
	default:
		msg := fmt.Sprintf("Unhandled scheme in MultiRemoteRaw %s", scheme)
		return nil, errors.New(msg)
	}
}
