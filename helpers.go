package graphpipe

import (
	"fmt"
	"math"
	"reflect"
	"unsafe"

	fb "github.com/google/flatbuffers/go"

	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

// Serialize writes a builder object to a byte array
func Serialize(b *fb.Builder, obj fb.UOffsetT) []byte {
	b.Finish(obj)
	return b.FinishedBytes()
}

// TensorToNativeTensor is a converter between the flatbuffer Tensor
// objects and the easier to use NativeTensor objects.
func TensorToNativeTensor(t *graphpipefb.Tensor) *NativeTensor {
	shape := make([]int64, t.ShapeLength())
	for i := 0; i < t.ShapeLength(); i++ {
		shape[i] = t.Shape(i)
	}
	nt := &NativeTensor{}
	if t.Type() == graphpipefb.TypeString {
		vals := make([]string, t.StringValLength())
		for i := 0; i < len(vals); i++ {
			vals[i] = string(t.StringVal(i))
		}
		nt.InitWithStringVals(vals, shape)
	} else {
		nt.InitWithData(t.DataBytes(), shape, t.Type())
	}
	return nt
}

// NativeTensorToNative is a converter between NativeTensors and raw
// arrays of arrays (of arrays of arrays) of numbers.
func NativeTensorToNative(t *NativeTensor) (interface{}, error) {
	if int(t.Type) > len(types) {
		return nil, fmt.Errorf("Unknown type: %d", t.Type)
	}

	meta := types[t.Type]
	typ := meta.typ
	if typ.Kind() == reflect.Struct {
		return nil, fmt.Errorf("Invalid type: %d", t.Type)
	}
	shapeLen := len(t.Shape)
	shape := make([]int, shapeLen)
	elems := 1
	for i := 0; i < shapeLen; i++ {
		typ = reflect.SliceOf(typ)
		shape[i] = int(t.Shape[i])
		elems *= shape[i]
	}
	if shapeLen == 0 {
		elems = 0
	}

	var data interface{}
	if t.Type == graphpipefb.TypeString {
		num := len(t.StringVals)
		if num != elems {
			return nil, fmt.Errorf("Incorrect number of elements in (%d) != (%d)",
				num, elems)
		}
		strs := make([]string, num)
		for i := 0; i < num; i++ {
			strs[i] = t.StringVals[i]
		}
		data = strs
	} else {
		if len(t.Data) != elems*int(meta.size) {
			return nil, fmt.Errorf("Incorrect number of elements in (%d) != (%d)",
				len(t.Data)/int(meta.size), elems)
		}
		data = meta.conv(t.Data)
	}

	return sliceData(typ, reflect.ValueOf(data), shape).Interface(), nil
}

// TensorToNative converts a tensor object into a native go slice. It can be
// cast to a proper type. For example:
// x := TensorToNative(t).([][]float32)
func TensorToNative(t *graphpipefb.Tensor) (interface{}, error) {
	if int(t.Type()) > len(types) {
		return nil, fmt.Errorf("Unknown type: %d", t.Type())
	}

	meta := types[t.Type()]
	typ := meta.typ
	if typ.Kind() == reflect.Struct {
		return nil, fmt.Errorf("Invalid type: %d", t.Type())
	}
	shapeLen := t.ShapeLength()
	shape := make([]int, shapeLen)
	elems := 1
	for i := 0; i < shapeLen; i++ {
		typ = reflect.SliceOf(typ)
		shape[i] = int(t.Shape(i))
		elems *= shape[i]
	}
	if shapeLen == 0 {
		elems = 0
	}

	var data interface{}
	if t.Type() == graphpipefb.TypeString {
		num := t.StringValLength()
		if num != elems {
			return nil, fmt.Errorf("Incorrect number of elements in (%d) != (%d)",
				num, elems)
		}
		strs := make([]string, num)
		for i := 0; i < num; i++ {
			strs[i] = string(t.StringVal(i))
		}
		data = strs
	} else {
		if t.DataLength() != elems*int(meta.size) {
			return nil, fmt.Errorf("Incorrect number of elements in (%d) != (%d)",
				t.DataLength()/int(meta.size), elems)
		}
		data = meta.conv(t.DataBytes())
	}

	return sliceData(typ, reflect.ValueOf(data), shape).Interface(), nil
}

// BuildTensorSafe builds a flatbuffer tensor from a native go slice or array.
// This version is safe to use with any array or slice. It will return an error
// if val contains a jagged slice.
// Note that this returns a flatbuffers offset type so that it can be
// used as part of building a larger object. The object can be serialized
// with the Serialize function.
func BuildTensorSafe(b *fb.Builder, val interface{}) (fb.UOffsetT, error) {
	return buildTensor(b, val, getDataSafe)
}

// BuildTensorContiguous builds a flatbuffer tensor from a native go slice or
// array. Using this on a non-contiguous slice or a jagged slice can result
// in memory corruption.
// Note that this returns a flatbuffers offset type so that it can be
// used as part of building a larger object. The object can be serialized
// with the Serialize function.
func BuildTensorContiguous(b *fb.Builder, val interface{}) (fb.UOffsetT, error) {
	return buildTensor(b, val, getDataContiguous)
}

// BuildTensorNonContiguous builds a flatbuffer tensor from a native go slice.
// This should be used if the slice is not contiguous in memory. Using this on
// a jagged slice can result in memory corruption.
// Note that this returns a flatbuffers offset type so that it can be
// used as part of building a larger object. The object can be serialized
// with the Serialize function.
func BuildTensorNonContiguous(b *fb.Builder, val interface{}) (fb.UOffsetT, error) {
	return buildTensor(b, val, getDataNonContiguous)
}

// BuildDataTensorRaw builds a data tensor from a byte slice. Validity is not
// checked, so passing a data slice that is not shape.numElems * dt.size will
// result in a tensor that is unusable by the receiver.
func BuildDataTensorRaw(b *fb.Builder, data []byte, shape []int64, dt uint8) fb.UOffsetT {
	dataFb := b.CreateByteVector(data)
	return buildTensorRaw(b, dataFb, 0, shape, dt)
}

// BuildStringTensorRaw builds a string tensor from a string slice.
func BuildStringTensorRaw(b *fb.Builder, strs []string, shape []int64) fb.UOffsetT {
	nstrs := len(strs)
	if shape == nil {
		shape = []int64{int64(nstrs)}
	}
	outs := make([]fb.UOffsetT, nstrs)
	for i := range strs {
		outs[i] = b.CreateByteVector([]byte(strs[i]))
	}
	graphpipefb.TensorStartStringValVector(b, nstrs)
	for i := len(outs) - 1; i >= 0; i-- {
		b.PrependUOffsetT(outs[i])
	}
	stringValFb := b.EndVector(nstrs)
	return buildTensorRaw(b, 0, stringValFb, shape, graphpipefb.TypeString)
}

type converter func([]byte) interface{}

func toUint8(b []byte) interface{} {
	return b
}

func toInt8(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]int8)(ptr))[:len(b)]
}

func toUint16(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]uint16)(ptr))[:len(b)/2]
}

func toInt16(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]int16)(ptr))[:len(b)/2]
}

func toUint32(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]uint32)(ptr))[:len(b)/4]
}

func toInt32(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]int32)(ptr))[:len(b)/4]
}

func toUint64(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]uint64)(ptr))[:len(b)/8]
}

func toInt64(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]int64)(ptr))[:len(b)/8]
}

func toFloat32(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]float32)(ptr))[:len(b)/4]
}

func toFloat64(b []byte) interface{} {
	ptr := unsafe.Pointer(&b[0])
	return (*(*[math.MaxUint32]float64)(ptr))[:len(b)/8]
}

var types = []struct {
	typ  reflect.Type
	size int64
	conv converter
}{
	{reflect.TypeOf(struct{}{}), 0, nil}, // null type
	{reflect.TypeOf(uint8(0)), 1, toUint8},
	{reflect.TypeOf(int8(0)), 1, toInt8},
	{reflect.TypeOf(uint16(0)), 2, toUint16},
	{reflect.TypeOf(int16(0)), 2, toInt16},
	{reflect.TypeOf(uint32(0)), 4, toUint32},
	{reflect.TypeOf(int32(0)), 4, toInt32},
	{reflect.TypeOf(uint64(0)), 8, toUint64},
	{reflect.TypeOf(int64(0)), 8, toInt64},
	{reflect.TypeOf(struct{}{}), 2, nil}, // no native float16
	{reflect.TypeOf(float32(0)), 4, toFloat32},
	{reflect.TypeOf(float64(0)), 8, toFloat64},
	{reflect.TypeOf(""), -1, nil},
}

func sliceData(typ reflect.Type, data reflect.Value, shape []int) reflect.Value {
	if len(shape) > 1 {
		ret := reflect.MakeSlice(typ, shape[0], shape[0])
		sub := shape[1:]
		size := data.Len() / shape[0]
		elem := typ.Elem()
		for i := 0; i < shape[0]; i++ {
			o := i * size
			ret.Index(i).Set(sliceData(elem, data.Slice(o, o+size), sub))
		}
		return ret
	}
	return data
}

// ShapeType returns shape, num, size, and dt for a reflect.Value. The value
// of size will be -1 (unknown) for a string type
func ShapeType(val reflect.Value) (shape []int64, num int64, size int64, dt uint8, err error) {
	typ := val.Type()
	num = int64(1)
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		l := int64(val.Len())
		num *= l
		shape = append(shape, l)
		typ = typ.Elem()
		if val.Len() > 0 {
			val = val.Index(0)
		} else {
			val = reflect.Zero(typ)
		}
		if typ.Kind() == reflect.Interface {
			val = reflect.ValueOf(val.Interface())
			typ = val.Type()
		}
	}
	if len(shape) == 0 {
		num = 0
	}
	for dt, t := range types {
		if typ.Kind() == t.typ.Kind() {
			return shape, num, t.size, uint8(dt), nil
		}
	}
	return shape, -1, -1, graphpipefb.TypeNull, fmt.Errorf("unsupported dtype %v", typ)
}

// extractStrs extracts strs from nested strings in row major order
func extractStrs(val reflect.Value, out []string) {
	typ := val.Type()
	if typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		outLen := len(out)
		rows := val.Len()
		cols := outLen / rows
		for i := 0; i < rows; i++ {
			o := i * cols
			extractStrs(val.Index(i), out[o:o+cols])
		}
	} else {
		out[0] = val.String()
	}
}

// pointerToData  gets a pointer to the underlying data of a slice
// or array
func pointerToData(val reflect.Value) unsafe.Pointer {
	typ := val.Type()
	for typ.Kind() == reflect.Slice {
		typ = typ.Elem()
		if val.Len() > 0 {
			val = val.Index(0)
		} else {
			val = reflect.Zero(typ)
		}
		if typ.Kind() == reflect.Interface {
			val = reflect.ValueOf(val.Interface())
			typ = val.Type()
		}
	}
	if typ.Kind() == reflect.Array {
		x := reflect.New(typ)
		x.Elem().Set(val)
		val = x
	} else {
		val = val.Addr()
	}
	return unsafe.Pointer(val.Pointer())
}

// fillContiguous fills a contiguous byte array from nested values
func fillContiguous(val reflect.Value, out []byte) {
	nestedKind := val.Index(0).Type().Kind()
	nested := nestedKind == reflect.Slice
	outLen := len(out)
	rows := val.Len()
	cols := outLen / rows
	for i := 0; i < rows; i++ {
		o := i * cols
		if nested {
			fillContiguous(val.Index(i), out[o:o+cols])
		} else {
			data, _ := getDataContiguous(val.Index(i), cols)
			copy(out[o:o+cols], data)
		}
	}
}

// checkData checks nested slices for valid data. Returns true if the data
// is contiguous
func checkData(val reflect.Value, size int) (bool, error) {
	if val.Len() == 0 {
		return true, nil
	}
	rows := val.Len()
	cols := size / rows
	contiguous := true
	if val.Index(0).Type().Kind() == reflect.Slice {
		num := val.Index(0).Len()
		var base uintptr
		if cols != 0 {
			base = uintptr(pointerToData(val))
		}
		for i := 0; i < rows; i++ {
			if num != val.Index(i).Len() {
				return false, fmt.Errorf("Nested slice is the wrong size")
			}
			var err error
			contiguous, err = checkData(val.Index(i), cols)
			if err != nil {
				return false, err
			}
			if base != 0 {
				ptr := uintptr(pointerToData(val.Index(i)))
				if ptr-base != uintptr(i*cols) {
					contiguous = false
				}
			}
		}
	}
	return contiguous, nil
}

func getDataContiguous(val reflect.Value, size int) ([]byte, error) {
	if size == 0 {
		return []byte{}, nil
	}
	ptr := pointerToData(val)
	data := (*(*[math.MaxUint32]byte)(ptr))[:size]
	return data, nil
}

func getDataNonContiguous(val reflect.Value, size int) ([]byte, error) {
	data := make([]byte, size)
	fillContiguous(val, data)
	return data, nil
}

func getDataSafe(val reflect.Value, size int) ([]byte, error) {
	contiguous, err := checkData(val, size)
	if err != nil {
		return nil, err
	}
	if contiguous {
		return getDataContiguous(val, size)
	}
	return getDataNonContiguous(val, size)
}

type dataGetter func(reflect.Value, int) ([]byte, error)

func buildTensor(b *fb.Builder, val interface{}, get dataGetter) (fb.UOffsetT, error) {
	v := reflect.ValueOf(val)
	shape, num, size, dt, err := ShapeType(v)
	if err != nil {
		return 0, err
	}
	if dt == graphpipefb.TypeString {
		strs := make([]string, num)
		extractStrs(v, strs)
		return BuildStringTensorRaw(b, strs, shape), nil
	}

	data, err := get(v, int(num*size))
	if err != nil {
		return 0, err
	}
	return BuildDataTensorRaw(b, data, shape, dt), nil
}

func buildTensorRaw(b *fb.Builder, dataFb, stringValFb fb.UOffsetT, shape []int64, dt uint8) fb.UOffsetT {
	ndim := len(shape)
	graphpipefb.TensorStartShapeVector(b, ndim)
	for i := len(shape) - 1; i >= 0; i-- {
		b.PrependInt64(shape[i])
	}
	shapeFb := b.EndVector(ndim)
	graphpipefb.TensorStart(b)
	graphpipefb.TensorAddShape(b, shapeFb)
	graphpipefb.TensorAddType(b, dt)
	if dataFb != 0 {
		graphpipefb.TensorAddData(b, dataFb)
	}
	if stringValFb != 0 {
		graphpipefb.TensorAddStringVal(b, stringValFb)
	}
	return graphpipefb.TensorEnd(b)
}
