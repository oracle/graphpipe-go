package graphpipe

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	bolt "github.com/coreos/bbolt"

	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

func TestSimpleGetResultsString(t *testing.T) {
	testGetResults(t, 1024, 4096, graphpipefb.TypeString, false)
}

func TestSimpleGetResultsFloat32(t *testing.T) {
	testGetResults(t, 1024, 4096, graphpipefb.TypeFloat32, false)
}

func BenchmarkGetResultsString(b *testing.B) {
	benchGetResults(b, 1024, 4096, graphpipefb.TypeString, false)
}

func BenchmarkGetResultsFloat(b *testing.B) {
	benchGetResults(b, 1024, 4096, graphpipefb.TypeFloat32, false)
}

func BenchmarkGetResultsStringCache(b *testing.B) {
	benchGetResults(b, 1024, 4096, graphpipefb.TypeString, true)
	time.Sleep(1)
}

func BenchmarkGetResultsFloatCache(b *testing.B) {
	benchGetResults(b, 1024, 4096, graphpipefb.TypeFloat32, true)
	time.Sleep(1)
}

func makeTensor(numRows int, dataLen int, dt uint8) *NativeTensor {

	shape := make([]int64, 1)
	shape[0] = int64(numRows)
	tp := &NativeTensor{}
	tp.Type = uint8(dt)
	tp.Shape = shape
	if tp.Type == graphpipefb.TypeString {
		tmp := make([]string, numRows)
		size := dataLen / numRows
		for i := 0; i < numRows; i++ {
			s := make([]byte, size)
			rand.Read(s)
			tmp[i] = string(s)
		}
		tp.StringVals = tmp
	} else {
		tp.Data = make([]byte, numRows*4)
		rand.Read(tp.Data)
	}
	return tp
}

func makeRequestRaw(tp *NativeTensor) *graphpipefb.InferRequest {
	builder := fb.NewBuilder(1024)
	inStrs := make([]fb.UOffsetT, 2)
	outStrs := make([]fb.UOffsetT, 2)
	for i := 0; i < 2; i++ {
		inStr := builder.CreateString(fmt.Sprintf("some/input/name:%d", i))
		outStr := builder.CreateString(fmt.Sprintf("some/output/name:%d", i))
		inStrs[i] = inStr
		outStrs[i] = outStr
	}
	graphpipefb.InferRequestStartInputNamesVector(builder, 2)
	for _, offset := range inStrs {
		builder.PrependUOffsetT(offset)
	}
	inputNames := builder.EndVector(2)
	graphpipefb.InferRequestStartOutputNamesVector(builder, 2)
	for _, offset := range outStrs {
		builder.PrependUOffsetT(offset)
	}
	outputNames := builder.EndVector(2)

	inputOffsets := make([]fb.UOffsetT, 2)
	for i := 0; i < 2; i++ {
		inputOffsets[i] = tp.Build(builder)
	}

	graphpipefb.InferRequestStartInputTensorsVector(builder, 2)
	for _, offset := range inputOffsets {
		builder.PrependUOffsetT(offset)
	}
	inputTensors := builder.EndVector(2)

	graphpipefb.InferRequestStart(builder)
	graphpipefb.InferRequestAddInputNames(builder, inputNames)
	graphpipefb.InferRequestAddOutputNames(builder, outputNames)
	graphpipefb.InferRequestAddInputTensors(builder, inputTensors)
	inferRequestOffset := graphpipefb.InferRequestEnd(builder)
	buf := Serialize(builder, inferRequestOffset)
	req := graphpipefb.GetRootAsInferRequest(buf, 0)
	return req
}

func makeRequest(numRows int, dataLen int, dt uint8) *graphpipefb.InferRequest {
	tp := makeTensor(numRows, dataLen, dt)
	return makeRequestRaw(tp)
}

func benchGetResults(b *testing.B, numRows int, dataLen int, dt uint8, cache bool) {
	tp := makeTensor(numRows, dataLen, dt)
	c := &appContext{}
	if cache {
		dir, _ := ioutil.TempDir("", "")
		defer os.RemoveAll(dir)
		dbPath := filepath.Join(dir, "test.db")
		var err error
		c.db, err = bolt.Open(dbPath, 0600, &bolt.Options{Timeout: 1 * time.Second})
		if err != nil {
			b.Fatal(err)
		}
		defer c.db.Close()
	}

	c.apply = func(*RequestContext, string, map[string]*NativeTensor, []string) ([]*NativeTensor, error) {
		return []*NativeTensor{tp, tp}, nil
	}

	req := makeRequest(numRows, dataLen, dt)

	rc := &RequestContext{builder: fb.NewBuilder(1024)}
	// fill the cache
	getResults(c, rc, req)
	for n := 0; n < b.N; n++ {
		getResults(c, rc, req)
	}
}

func testGetResults(t *testing.T, numRows int, dataLen int, dt uint8, cache bool) {
	tp := makeTensor(numRows, dataLen, dt)
	c := &appContext{}
	if cache {
		dir, _ := ioutil.TempDir("", "")
		defer os.RemoveAll(dir)
		dbPath := filepath.Join(dir, "test.db")
		var err error
		c.db, err = bolt.Open(dbPath, 0600, &bolt.Options{Timeout: 1 * time.Second})
		if err != nil {
			t.Fatal(err)
		}
		defer c.db.Close()
	}

	c.apply = func(*RequestContext, string, map[string]*NativeTensor, []string) ([]*NativeTensor, error) {
		return []*NativeTensor{tp, tp}, nil
	}

	req := makeRequest(numRows, dataLen, dt)

	rc := &RequestContext{builder: fb.NewBuilder(1024)}
	getResults(c, rc, req)
}

func TestCachedGetResultsInterleavedFloat32(t *testing.T) {
	dt := uint8(graphpipefb.TypeFloat32)
	c := &appContext{}
	numRows := 10
	dataLen := 1024

	tp1 := makeTensor(numRows, dataLen, dt)
	tp2 := makeTensor(numRows, dataLen, dt)
	tp3 := makeTensor(numRows, dataLen, dt)

	cache := true
	if cache {
		dir, _ := ioutil.TempDir("", "")
		defer os.RemoveAll(dir)
		dbPath := filepath.Join(dir, "test.db")
		var err error
		c.db, err = bolt.Open(dbPath, 0600, &bolt.Options{Timeout: 1 * time.Second})
		if err != nil {
			t.Fatal(err)
		}
		defer c.db.Close()
	}

	c.apply = func(b *RequestContext, c string, inputs map[string]*NativeTensor, d []string) ([]*NativeTensor, error) {
		rval := []*NativeTensor{}
		for _, value := range inputs {
			rval = append(rval, value)
		}
		return rval, nil
	}

	req := makeRequestRaw(tp2)
	rc := &RequestContext{builder: fb.NewBuilder(1024)}
	results, _ := getResults(c, rc, req)

	if !bytes.Equal(results[0].Data, tp2.Data) {
		t.Fatalf("Results are not the same as input \na: %v\nb: %v", results[0].Shape, tp2.Shape)
	}

	results, _ = getResults(c, rc, req)

	if !bytes.Equal(results[0].Data, tp2.Data) {
		t.Fatalf("Results are not the same as input \na: %v\nb: %v", results[0].Shape, tp2.Shape)
	}

	// Create a new tensor that includes cached data in the middle part of the tensor
	tp1.Data = append(tp1.Data, tp2.Data...)
	tp1.Data = append(tp1.Data, tp3.Data...)
	tp1.Shape[0] = 30

	req = makeRequestRaw(tp1)
	rc = &RequestContext{builder: fb.NewBuilder(1024)}

	results, _ = getResults(c, rc, req)

	if !bytes.Equal(results[0].Data, tp1.Data) {
		t.Fatalf("Results are not the same as input \na: %v\nb: %v", results[0].Shape, tp2.Shape)
	}
}
