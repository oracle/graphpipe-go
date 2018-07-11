package graphpipe

import (
	"errors"
	"reflect"

	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

type NativeTensor struct {
	Type       uint8
	Shape      []int64
	StringVals []string
	Data       []byte
}

func (nt *NativeTensor) InitSimple(val interface{}) error {
	v := reflect.ValueOf(val)
	shape, num, size, dt, err := ShapeType(v)
	nt.Shape = shape
	nt.Type = dt
	if nt.Type == graphpipefb.TypeString {
		strs := make([]string, num)
		extractStrs(v, strs)
		nt.StringVals = strs

	} else {
		nt.Data, err = getDataSafe(v, int(num*size))
	}
	return err
}

func (nt *NativeTensor) InitWithData(data []byte, shape []int64, dt uint8) error {
	totalBytes := 1
	meta := types[dt]
	for _, v := range shape {
		totalBytes *= int(v)
	}
	totalBytes *= int(meta.size)
	if totalBytes != len(data) {
		return errors.New("Data length does not match shape/dtype")
	}
	nt.Shape = shape
	nt.Type = dt
	nt.Data = data
	return nil
}

func (nt *NativeTensor) InitWithStringVals(stringVals []string, shape []int64) error {
	totalItems := 1
	for _, v := range shape {
		totalItems *= int(v)
	}
	if len(stringVals) != totalItems {
		return errors.New("Data length does not match shape/dtype")
	}
	nt.Shape = shape
	nt.Type = graphpipefb.TypeString
	nt.StringVals = stringVals
	return nil
}

func (nt *NativeTensor) Build(b *fb.Builder) fb.UOffsetT {
	if nt.Type == graphpipefb.TypeString {
		return BuildStringTensorRaw(b, nt.StringVals, nt.Shape)
	} else {
		return BuildDataTensorRaw(b, nt.Data, nt.Shape, nt.Type)
	}
}
