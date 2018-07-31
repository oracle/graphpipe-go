package graphpipe

import (
	"reflect"
	"testing"

	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

func TestGetMethodTypeShapes(t *testing.T) {
	s := int32(graphpipefb.TypeString)
	sm := func(_ *RequestContext, _ string, i []string) []string { return i }
	testGetMethodTypeShapes(t, sm, []int32{s, s}, []int64{-1}, []int64{-1})
	sm2 := func(_ *RequestContext, _ string, i [][]string) [][]string { return i }
	testGetMethodTypeShapes(t, sm2, []int32{s, s}, []int64{-1, -1}, []int64{-1, -1})
	sm3 := func(_ *RequestContext, _ string, i [][3]string) [][3]string { return i }
	testGetMethodTypeShapes(t, sm3, []int32{s, s}, []int64{-1, 3}, []int64{-1, 3})
	f := int32(graphpipefb.TypeFloat32)
	fm := func(_ *RequestContext, _ string, i []float32) []float32 { return i }
	testGetMethodTypeShapes(t, fm, []int32{f, f}, []int64{-1}, []int64{-1})
}

func testGetMethodTypeShapes(t *testing.T, method interface{}, expectedTypes []int32, expectedInput, expectedOutput []int64) {
	types, input, output := getMethodTypeShapes(method)
	checkTypes, _ := TensorToNative(types)
	if !reflect.DeepEqual(checkTypes, expectedTypes) {
		t.Fatalf("types %v and %v are not equal", checkTypes, expectedTypes)
	}
	checkInput, _ := TensorToNative(input[0])
	if !reflect.DeepEqual(checkInput, expectedInput) {
		t.Fatalf("inputs %v and %v are not equal", checkInput, expectedInput)
	}
	checkOutput, _ := TensorToNative(output[0])
	if !reflect.DeepEqual(checkOutput, expectedOutput) {
		t.Fatalf("outputs %v and %v are not equal", checkOutput, expectedOutput)
	}
}

func TestConversion(t *testing.T) {
	testConvert(t, []string{"foo", "bar", "baz"})
	testConvert(t, [][]string{{"foo"}, {"bar"}, {"baz"}})
	testConvert(t, []int32{1, 2, 3})
	testConvert(t, [][]int32{{1}, {2}, {3}})
	testConvert(t, []float32{1.0, 2.0, 3.0})
	testConvert(t, [][]float32{{1.0}, {2.0}, {3.0}})
}

func testConvert(t *testing.T, val interface{}) {
	b := fb.NewBuilder(1024)
	x, err := BuildTensorSafe(b, val)

	if err != nil {
		t.Fatal("Couldn't build tensor")
	}

	temp := Serialize(b, x)
	tensor := graphpipefb.GetRootAsTensor(temp, 0)
	res, err := TensorToNative(tensor)

	if err != nil {
		t.Fatal("Couldn't build native")
	}

	if !reflect.DeepEqual(val, res) {
		t.Fatalf("values %v and %v are not equal", val, res)
	}
}
