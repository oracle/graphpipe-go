package graphpipe

import (
	"bytes"
	"crypto/sha512"
	"encoding/binary"
	"fmt"
	"sort"
	"unsafe"

	"github.com/Sirupsen/logrus"
	bolt "github.com/coreos/bbolt"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

const (
	emptyKey string = ".empty"
	tsKey    string = ".typeshape"
	twoGigs         = 2 * 1024 * 1024 * 1024
)

// Nt is a NativeTensor holding struct.
type Nt struct {
	tensor    *NativeTensor
	name      []byte
	dlen      int
	rows      int
	typeShape []byte
}

func newNt(tensor *NativeTensor, name string, chunks int) *Nt {
	typeShape := make([]byte, (len(tensor.Shape)+1)*8)
	binary.LittleEndian.PutUint64(typeShape[0:8], uint64(tensor.Type))

	var dlen int
	if tensor.Type == graphpipefb.TypeString {
		dlen = len(tensor.StringVals)
	} else {
		dlen = len(tensor.Data)
	}
	dlen /= chunks
	// skip the first dimenion
	dims := len(tensor.Shape)
	rows := 0
	if dims > 0 {
		rows = int(tensor.Shape[0])
		rows /= chunks
		binary.LittleEndian.PutUint64(typeShape[(1*8):(1+1)*8], uint64(rows))
	}
	for i := 1; i < dims; i++ {
		o := i + 1
		size := int(tensor.Shape[i])
		binary.LittleEndian.PutUint64(typeShape[(o*8):(o+1)*8], uint64(size))
	}
	return &Nt{
		tensor,
		[]byte(name),
		dlen,
		rows,
		typeShape,
	}
}

func (t *Nt) data(index int) []byte {
	if t.tensor.Type == graphpipefb.TypeString {
		// encode strings into TensorContent style
		strs := make([][]byte, len(t.tensor.StringVals))
		for i := 0; i < len(t.tensor.StringVals); i++ {
			strs[i] = []byte(t.tensor.StringVals[i])
		}
		return encodeStrs(strs)
	}
	rval := t.tensor.Data[index*t.dlen : (index+1)*t.dlen]
	return rval
}

func (t *Nt) tensorFromIndexes(indexes []int) *NativeTensor {
	dt := t.tensor.Type
	shape := make([]int64, len(t.tensor.Shape))
	for i := 0; i < len(t.tensor.Shape); i++ {
		shape[i] = t.tensor.Shape[i]
	}
	if len(shape) > 0 {
		shape[0] = int64(len(indexes))
	}
	nt := &NativeTensor{}

	if dt == graphpipefb.TypeString {
		stringVals := make([]string, len(indexes)*t.dlen)

		for i, index := range indexes {
			for j := 0; j < t.dlen; j++ {
				stringVals[i*t.dlen+j] = t.tensor.StringVals[index*t.dlen+j]
			}
		}
		nt.InitWithStringVals(stringVals, shape)
	} else {
		data := []byte{}
		for _, i := range indexes {
			data = append(data, t.data(i)...)
		}

		nt.InitWithData(data, shape, dt)
	}
	return nt
}

func getKey(c *appContext, inputs []*Nt, index int) []byte {
	numInputs := len(inputs)
	if numInputs == 0 {
		return []byte(emptyKey)
	}
	h := sha512.New()
	for i := 0; i < numInputs; i++ {
		h.Write(inputs[i].name)
		h.Write(inputs[i].typeShape[0:8])
		// skip the batch dimension
		if len(inputs[i].typeShape) > 16 {
			h.Write(inputs[i].typeShape[16:])
		}
		h.Write(inputs[i].data(index))
	}
	return h.Sum(nil)
}

func dataLen(typeShape []byte) int {
	dims := len(typeShape) / 8
	elements := int64(1)
	for i := 1; i < dims; i++ {
		elements *= int64(binary.LittleEndian.Uint64(typeShape[(i * 8) : (i+1)*8]))
	}
	var size int
	dt := binary.LittleEndian.Uint64(typeShape[0:8])
	switch dt {
	case graphpipefb.TypeUint8, graphpipefb.TypeInt8:
		size = 1
	case graphpipefb.TypeUint16, graphpipefb.TypeInt16, graphpipefb.TypeFloat16:
		size = 2
	case graphpipefb.TypeUint32, graphpipefb.TypeInt32, graphpipefb.TypeFloat32:
		size = 4
	case graphpipefb.TypeUint64, graphpipefb.TypeInt64, graphpipefb.TypeFloat64:
		size = 8
	case graphpipefb.TypeString:
		fallthrough
	default:
		return -1
	}
	return int(elements) * size
}

func getCache(c *appContext, keys [][]byte, outputs []string) ([][][]byte, [][]byte, []bool, []bool, error) {
	numOutputs := len(outputs)
	data := make([][][]byte, numOutputs)
	typeShape := make([][]byte, numOutputs)
	incompleteOutputs := make([]bool, numOutputs)
	numChunks := len(keys)
	incompleteChunks := make([]bool, numChunks)
	if err := c.db.View(func(tx *bolt.Tx) error {
		for i := range outputs {
			data[i] = make([][]byte, numChunks)
			// b is only valid for the length of the transaction
			bucket := tx.Bucket([]byte(outputs[i]))
			var b []byte
			if bucket != nil {
				b = bucket.Get([]byte(tsKey))
			}
			if bucket == nil || b == nil {
				// if we haven't set the shape then there should be
				// no keys so skip retrieval
				incompleteOutputs[i] = true
				for j := 0; j < numChunks; j++ {
					incompleteChunks[j] = true
				}
			} else {
				typeShape[i] = make([]byte, len(b))
				copy(typeShape[i], b)
				dlen := dataLen(typeShape[i])
				if dlen != -1 {
					// make data contiguous
					content := make([]byte, dlen*numChunks)
					for j := 0; j < numChunks; j++ {
						data[i][j] = content[j*dlen : (j+1)*dlen]
						b := bucket.Get(keys[j])
						if b == nil {
							incompleteOutputs[i] = true
							incompleteChunks[j] = true
						} else {
							copy(data[i][j], b)
						}
					}
				} else {
					for j := 0; j < numChunks; j++ {
						b := bucket.Get(keys[j])
						if b == nil {
							incompleteOutputs[i] = true
							incompleteChunks[j] = true
						} else {
							data[i][j] = make([]byte, len(b))
							copy(data[i][j], b)
						}
					}
				}
			}
		}
		return nil
	}); err != nil {
		logrus.Errorf("Failed to get item from cache: %v", err)
		return nil, nil, nil, nil, err
	}
	return data, typeShape, incompleteChunks, incompleteOutputs, nil
}

func setCache(c *appContext, keys [][]byte, outputs []string, data [][][]byte, typeShape [][]byte, missing []int) error {
	if err := c.db.Update(func(tx *bolt.Tx) error {
		for i := range outputs {
			bucket := tx.Bucket([]byte(outputs[i]))
			if bucket == nil {
				var err error
				bucket, err = tx.CreateBucketIfNotExists([]byte(outputs[i]))
				if err != nil {
					return err
				}
			}
			b := bucket.Get([]byte(tsKey))
			if b == nil {
				if err := bucket.Put([]byte(tsKey), typeShape[i]); err != nil {
					return err
				}
			}
			for j := range missing {
				row := missing[j]
				if err := bucket.Put(keys[row], data[i][row]); err != nil {
					return err
				}
			}
		}
		return nil
	}); err != nil {
		if err == bolt.ErrDatabaseNotOpen {
			logrus.Debugf("Ingoring put cache error: %v", err)
			return nil
		}
		logrus.Errorf("Failed to put item in cache: %v", err)
		return err
	}
	return nil
}

func rows(inputs []*NativeTensor) int64 {
	rows := int64(0)
	for _, input := range inputs {
		if len(input.Shape) < 1 {
			return 1
		}
		if rows == 0 {
			rows = input.Shape[0]
		}
		if rows != input.Shape[0] {
			return 1
		}
	}
	return rows
}

func decodeStrs(buf []byte, num int64) ([]string, error) {
	reader := bytes.NewReader(buf)
	lens := make([]uint64, num)
	strs := make([]string, num)
	for i := int64(0); i < num; i++ {
		val, err := binary.ReadUvarint(reader)
		if err != nil {
			return nil, err
		}
		lens[i] = val
	}
	for i := int64(0); i < num; i++ {
		tmp := make([]byte, lens[i])
		if lens[i] == 0 {
			continue
		}
		n, err := reader.Read(tmp)
		if err != nil || uint64(n) != lens[i] {
			return nil, err
		}
		strs[i] = string(tmp)
	}
	return strs, nil
}

func encodeStrs(strs [][]byte) []byte {
	num := len(strs)
	buf := make([]byte, binary.MaxVarintLen64*num)
	n := 0
	for i := 0; i < num; i++ {
		n += binary.PutUvarint(buf[n:], uint64(len(strs[i])))
	}
	buf = buf[:n]
	for i := 0; i < num; i++ {
		buf = append(buf, strs[i]...)
	}
	return buf
}

func ntFromData(typeShape []byte, data [][]byte) (*NativeTensor, error) {
	tp := &NativeTensor{}
	// The first 8 bytes holds the type
	tp.Type = uint8(binary.LittleEndian.Uint64(typeShape[0:8]))
	dims := (len(typeShape) - 1) / 8
	tp.Shape = make([]int64, dims)
	for i := 0; i < dims; i++ {
		o := i + 1
		size := int64(binary.LittleEndian.Uint64(typeShape[(o * 8) : (o+1)*8]))
		tp.Shape[i] = size
	}
	// update the dimension with the number of chunks
	elementsPerChunk := tp.Shape[0]
	tp.Shape[0] *= int64(len(data))
	if tp.Type == graphpipefb.TypeString {
		for i := 0; i < len(data); i++ {
			strs, err := decodeStrs(data[i], elementsPerChunk)
			if err != nil {
				return nil, err
			}
			tp.StringVals = append(tp.StringVals, strs...)
		}
	} else {
		// data is contiguous so just grab the whole thing
		dataSize := len(data) * len(data[0])
		if dataSize > twoGigs {
			return nil, fmt.Errorf("Proto is larger than 2 Gigabytes")
		}
		if dataSize > 0 {
			ptr := unsafe.Pointer(&data[0][0])
			tp.Data = (*(*[twoGigs]byte)(ptr))[:dataSize]
		}
	}
	return tp, nil
}

func mergeResultsWithCacheData(results []*NativeTensor, applyIndexes []int, typeShape [][]byte, missing []int, numChunks int, data [][][]byte) [][][]byte {
	numApply := len(applyIndexes)
	numMissing := len(missing)
	for i := 0; i < numApply; i++ {
		ix := applyIndexes[i]
		nt := newNt(results[i], "", numMissing)
		if typeShape[ix] == nil {
			typeShape[ix] = nt.typeShape
			dlen := dataLen(typeShape[ix])
			if dlen != -1 {
				// make data contiguous
				content := make([]byte, dlen*numChunks)
				for j := 0; j < numChunks; j++ {
					data[ix][j] = content[j*dlen : (j+1)*dlen]
				}
			}
		}
		for j := 0; j < numMissing; j++ {
			if data[ix][missing[j]] == nil {
				data[ix][missing[j]] = nt.data(j)
			} else {
				copy(data[ix][missing[j]], nt.data(j))
			}
		}
	}
	return data
}

func getInputTensors(req *graphpipefb.InferRequest) ([]*NativeTensor, error) {
	inputTensors := make([]*NativeTensor, req.InputTensorsLength())

	for i := 0; i < req.InputTensorsLength(); i++ {
		tensor := &graphpipefb.Tensor{}

		if !req.InputTensors(tensor, i) {
			err := fmt.Errorf("Bad input tensor")
			return nil, err
		}

		nt := TensorToNativeTensor(tensor)
		inputTensors[i] = nt
	}
	return inputTensors, nil
}

func getResultsCached(c *appContext, requestContext *RequestContext, req *graphpipefb.InferRequest) ([]*NativeTensor, error) {
	inputTensors, err := getInputTensors(req)
	if err != nil {
		return nil, err
	}

	numChunks := 1
	if len(inputTensors) > 0 {
		numChunks = int(rows(inputTensors))
		// no input, so just get values as a single chunk
		if numChunks == 0 {
			numChunks = 1
		}
		logrus.Debugf("Request divides into %d chunks", numChunks)
	}

	inputs, err := getInputs(c, req, numChunks)
	if err != nil {
		return nil, err
	}

	keys := make([][]byte, numChunks)
	for i := 0; i < numChunks; i++ {
		keys[i] = getKey(c, inputs, i)
	}

	outputNames := getOutputNames(req)
	if len(outputNames) == 0 {
		outputNames = append(outputNames, c.defaultOutputs...)
	}
	for i := range outputNames {
		n := 0
		if outputNames[i] == "" {
			outputNames[i] = c.defaultOutputs[n]
			n++
		}
	}

	data, typeShape, incompleteChunks, incompleteOutputs, err := getCache(c, keys, outputNames)
	if err != nil {
		logrus.Errorf("Failed to get cached data: %v", err)
		return nil, err
	}

	missing := []int{}
	for i := 0; i < numChunks; i++ {
		if incompleteChunks[i] {
			missing = append(missing, i)
		}
	}

	numOutputs := len(outputNames)
	applyIndexes := []int{}
	for i := 0; i < numOutputs; i++ {
		if incompleteOutputs[i] {
			applyIndexes = append(applyIndexes, i)
		}
	}

	numMissing := len(missing)
	numApply := len(applyIndexes)
	logrus.Debugf("%d rows must be calculated", numMissing)

	if numMissing == 0 {
		logrus.Infof("Skipping apply because everything is cached")
	} else if numApply == 0 {
		logrus.Infof("Skipping apply because no outputs requested")
	} else {
		applyInputs := map[string]*NativeTensor{}
		for i := 0; i < len(inputs); i++ {
			applyInputs[string(inputs[i].name)] = inputs[i].tensorFromIndexes(missing)
		}
		results, err := c.apply(requestContext, string(req.Config()), applyInputs, outputNames)
		if err != nil {
			logrus.Errorf("Apply failed: %v", err)
			return nil, err
		}
		data = mergeResultsWithCacheData(results, applyIndexes, typeShape, missing, numChunks, data)
		// set cache async so we can complete the request
		go func() {
			if err := setCache(c, keys, outputNames, data, typeShape, missing); err != nil {
				logrus.Errorf("Failed to set cache: %v", err)
			}
		}()
	}

	outputNts := make([]*NativeTensor, numOutputs)
	for i := 0; i < numOutputs; i++ {
		outputNts[i], err = ntFromData(typeShape[i], data[i])
		if err != nil {
			logrus.Errorf("Failed to create native tensor: %v", err)
		}
	}
	return outputNts, nil
}

type byName []*Nt

func (a byName) Len() int           { return len(a) }
func (a byName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byName) Less(i, j int) bool { return bytes.Compare(a[i].name, a[j].name) < 0 }

func getInputs(c *appContext, req *graphpipefb.InferRequest, numChunks int) ([]*Nt, error) {
	inputs := make([]*Nt, req.InputTensorsLength())
	for i := 0; i < req.InputTensorsLength(); i++ {
		tensor := &graphpipefb.Tensor{}
		if !req.InputTensors(tensor, i) {
			return nil, fmt.Errorf("Could not init tensor")
		}
		nt := TensorToNativeTensor(tensor)
		name := ""
		if i < req.InputNamesLength() {
			name = string(req.InputNames(i))
		}
		if name == "" {
			name = c.defaultInputs[i]
		}
		inputs[i] = newNt(nt, name, numChunks)
	}
	sort.Sort(byName(inputs))
	return inputs, nil
}
