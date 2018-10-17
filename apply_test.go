/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/

package graphpipe

import (
	"fmt"
	"reflect"
	"testing"

	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

func TestSimpleApplyString(t *testing.T) {
	v := []string{"foo", "bar", "baz"}
	tmp, _ := nativeToTensor(v)
	tensor := TensorToNativeTensor(tmp)
	if err := testSimpleApplyFunc(t, tensor, applyInterface); err != nil {
		t.Fatal(err)
	}
	if err := testSimpleApplyFunc(t, tensor, applyString); err != nil {
		t.Fatal(err)
	}
	if err := testSimpleApplyFunc(t, tensor, applyInterfaceError); err != nil {
		t.Fatal(err)
	}
	if err := testSimpleApplyFunc(t, tensor, applyStringError); err != nil {
		t.Fatal(err)
	}
}

func TestSimpleApplyFloat(t *testing.T) {
	v := []float32{1., 2., 3.}
	tmp, _ := nativeToTensor(v)
	tensor := TensorToNativeTensor(tmp)
	if err := testSimpleApplyFunc(t, tensor, applyInterface); err != nil {
		t.Fatal(err)
	}
	if err := testSimpleApplyFunc(t, tensor, applyFloat); err != nil {
		t.Fatal(err)
	}
	if err := testSimpleApplyFunc(t, tensor, applyInterfaceError); err != nil {
		t.Fatal(err)
	}
	if err := testSimpleApplyFunc(t, tensor, applyFloatError); err != nil {
		t.Fatal(err)
	}
}

func TestSimpleApplyMulti(t *testing.T) {
	tmp, _ := nativeToTensor([][]string{{"a", "b", "c"}, {"d", "e", "f"}, {"g", "h", "i"}})
	tensor := TensorToNativeTensor(tmp)
	if err := testSimpleApplyFunc(t, tensor, applyMulti); err != nil {
		t.Fatal(err)
	}
}

func testSimpleApplyFunc(t *testing.T, tensor *NativeTensor, f interface{}) error {
	return testSimpleApplyFuncBounded(t, tensor, f, nil, nil)
}

func testSimpleApplyFuncBounded(t *testing.T, tensor *NativeTensor, f interface{}, inshape, outshape [][]int64) error {
	opts := BuildSimpleApply(f, inshape, outshape)
	inputs := map[string]*NativeTensor{"input0": tensor}
	outputNames := []string{"output0"}
	rc := RequestContext{builder: fb.NewBuilder(1024)}

	res, err := opts.Apply(&rc, "", inputs, outputNames)
	if err != nil {
		return err
	}

	offset := res[0].Build(rc.builder)
	buf := Serialize(rc.builder, offset)
	t2 := graphpipefb.GetRootAsTensor(buf, 0)
	a, _ := NativeTensorToNative(tensor)
	b, _ := TensorToNative(t2)
	if a == nil || !reflect.DeepEqual(a, b) {
		return fmt.Errorf("tps %v and %v are not equal", a, b)
	}
	return nil
}

func TestSimpleApplyShapes(t *testing.T) {
	tmp, _ := nativeToTensor([][]string{{"a", "b", "c"}, {"d", "e", "f"}, {"g", "h", "i"}})
	tensor := TensorToNativeTensor(tmp)
	shape := [][]int64{{-1, 3}}
	if err := testSimpleApplyFuncBounded(t, tensor, applyMulti, shape, nil); err != nil {
		t.Fatal(err)
	}
	if err := testSimpleApplyFuncBounded(t, tensor, applyMulti, nil, shape); err != nil {
		t.Fatal(err)
	}
	shape = [][]int64{{2, -1}}
	if err := testSimpleApplyFuncBounded(t, tensor, applyMulti, shape, nil); err == nil {
		t.Fatal("Sending wrong size failed to error")
	}
	/*
		if err := testSimpleApplyFuncBounded(t, tensor, applyMulti, nil, shape); err == nil {
			t.Fatal("Returning wrong size failed to error")
		}
	*/
}

func applyInterface(_ *RequestContext, config string, in interface{}) interface{} {
	return in
}

func applyString(_ *RequestContext, config string, in []string) []string {
	return in
}

func applyFloat(_ *RequestContext, config string, in []float32) []float32 {
	return in
}

func applyInterfaceError(_ *RequestContext, config string, in interface{}) (interface{}, error) {
	return in, nil
}

func applyStringError(_ *RequestContext, config string, in []string) ([]string, error) {
	return in, nil
}

func applyFloatError(_ *RequestContext, config string, in []float32) ([]float32, error) {
	return in, nil
}

func applyMulti(_ *RequestContext, config string, in [][]string) [][]string {
	return in
}
