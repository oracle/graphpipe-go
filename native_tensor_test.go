/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/

package graphpipe

import (
	"testing"

	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

func TestNativeTensorFlatInt64(t *testing.T) {
	nt := &NativeTensor{}
	v := []int64{2, 2}
	err := nt.InitSimple(v)
	if err != nil {
		t.Fatal(err)
	}
	if len(nt.Shape) != 1 {
		t.Fatal("Wrong shape")
	}
	if len(nt.Data) != 8*2 {
		t.Fatal("Wrong data length")
	}
	if nt.Type != graphpipefb.TypeInt64 {
		t.Fatal("Wrong data type")
	}

	b := fb.NewBuilder(1024)
	nt.Build(b)
}

func TestNativeTensorString(t *testing.T) {
	nt := &NativeTensor{}
	v := []string{"foo", "bar", "baz"}
	err := nt.InitSimple(v)
	if err != nil {
		t.Fatal(err)
	}
	if len(nt.Shape) != 1 {
		t.Fatal("Wrong shape")
	}
	if nt.Shape[0] != 3 {
		t.Fatal("Wrong item count")
	}
	if nt.Type != graphpipefb.TypeString {
		t.Fatal("Wrong data type")
	}

	b := fb.NewBuilder(1024)
	nt.Build(b)
}

func TestInitWithData(t *testing.T) {
	nt := &NativeTensor{}
	dataLen := int(255)
	data := make([]byte, dataLen)

	for i := range data {
		data[i] = byte(i % dataLen)
	}

	shape := make([]int64, 1)
	shape[0] = int64(dataLen)
	err := nt.InitWithData(data, shape, graphpipefb.TypeInt8)
	if err != nil {
		t.Fatal(err)
	}

	builder := fb.NewBuilder(1024)
	nt.Build(builder)
}

func TestInitWithInvalidData(t *testing.T) {
	nt := &NativeTensor{}
	dataLen := int(255)
	data := make([]byte, dataLen)

	for i := range data {
		data[i] = byte(i % dataLen)
	}

	shape := make([]int64, 1)
	shape[0] = int64(dataLen) - 1 // invalid shape
	err := nt.InitWithData(data, shape, graphpipefb.TypeInt8)
	if err == nil {
		t.Fatal("Expecting an error to be returned for mis-shaped data")
	}
}

func TestInitWithStringVals(t *testing.T) {
	nt := &NativeTensor{}
	strings := []string{"a", "b", "c", "d"}

	shape := make([]int64, 2)
	shape[0] = 2
	shape[1] = 2
	err := nt.InitWithStringVals(strings, shape)
	if err != nil {
		t.Fatal(err)
	}

	builder := fb.NewBuilder(1024)
	nt.Build(builder)
}

func TestInitWithInvalidStringVals(t *testing.T) {
	nt := &NativeTensor{}
	strings := []string{"a", "b", "c", "d"}

	shape := make([]int64, 2)
	shape[0] = 2
	shape[1] = 3 // Too big!
	err := nt.InitWithStringVals(strings, shape)
	if err == nil {
		t.Fatal("Expecting an error to be returned for mis-shaped data")
	}
}
