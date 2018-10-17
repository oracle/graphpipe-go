/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/

package graphpipe

import (
	"reflect"

	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

var errorInterface = reflect.TypeOf((*error)(nil)).Elem()

func nativeToTensor(val interface{}) (*graphpipefb.Tensor, error) {
	b := fb.NewBuilder(1024)
	x, err := BuildTensorSafe(b, val)

	if err != nil {
		return nil, err
	}

	temp := Serialize(b, x)
	tensor := graphpipefb.GetRootAsTensor(temp, 0)
	return tensor, nil
}

func getMethodTypeShapes(method interface{}) (*graphpipefb.Tensor, []*graphpipefb.Tensor, []*graphpipefb.Tensor) {
	types := []int32{}
	mtyp := reflect.ValueOf(method).Type()
	nin := mtyp.NumIn() - 2 // minus two to skip RequestContext and config
	inputTensors := make([]*graphpipefb.Tensor, nin)
	for i := 0; i < nin; i++ {
		typ := mtyp.In(i + 2) // plus two to skip the RequestContext and config
		shape, _, _, dt, _ := ShapeType(reflect.Zero(typ))
		types = append(types, int32(dt))
		for j := range shape {
			if shape[j] == 0 {
				shape[j] = -1
			}
		}
		inputTensors[i], _ = nativeToTensor(shape)
	}
	outputTensors := []*graphpipefb.Tensor{}
	for i := 0; i < mtyp.NumOut(); i++ {
		typ := mtyp.Out(i)
		if typ.Implements(errorInterface) {
			continue
		}
		shape, _, _, dt, _ := ShapeType(reflect.Zero(typ))
		types = append(types, int32(dt))
		for j := range shape {
			if shape[j] == 0 {
				shape[j] = -1
			}
		}
		tensor, _ := nativeToTensor(shape)
		outputTensors = append(outputTensors, tensor)
	}
	typesTensor, _ := nativeToTensor(types)
	return typesTensor, inputTensors, outputTensors
}
