package graphpipe

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"runtime/debug"
	"strconv"
	"strings"

	"github.com/Sirupsen/logrus"

	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

// Applier is the base signature for server actions.
type Applier func(*RequestContext, string, map[string]*NativeTensor, []string) ([]*NativeTensor, error)

// SimpleApplier is the signature for a server action that converts
// between native types and graphpipe objects.
type SimpleApplier func(interface{}) (interface{}, error)
type simpleContext struct {
	names      *graphpipefb.Tensor
	types      *graphpipefb.Tensor
	inShapes   [][]int64
	outShapes  [][]int64
	numInputs  int
	numOutputs int
	shapes     map[string]*graphpipefb.Tensor
	wrapped    interface{}
	rawOpts    *ServeRawOptions
}

func (c *simpleContext) getHandler(w http.ResponseWriter, r *http.Request, body []byte) error {
	js, err := json.MarshalIndent(c.rawOpts.Meta, "", "    ")
	if err == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(js)
	}
	return err
}

// BuildSimpleApply is the factory for producing SimpleAppliers
func BuildSimpleApply(apply interface{}, inShapes, outShapes [][]int64) *ServeRawOptions {
	s := &simpleContext{
		inShapes:  inShapes,
		outShapes: outShapes,
		wrapped:   apply,
	}
	opts := &ServeRawOptions{}

	s.shapes = map[string]*graphpipefb.Tensor{}
	typesTensor, inputTensors, outputTensors := getMethodTypeShapes(apply)
	s.types = typesTensor
	s.numInputs = len(inputTensors)
	s.numOutputs = len(outputTensors)
	for i := range inputTensors {
		name := "input" + strconv.Itoa(i)
		if len(inShapes) > i {
			s.shapes[name], _ = nativeToTensor(inShapes[i])
		} else {
			s.shapes[name] = inputTensors[i]
		}
		opts.DefaultInputs = append(opts.DefaultInputs, name)
	}
	for i := range outputTensors {
		name := "output" + strconv.Itoa(i)
		if len(outShapes) > i {
			s.shapes[name], _ = nativeToTensor(outShapes[i])
		} else {
			s.shapes[name] = outputTensors[i]
		}
		opts.DefaultOutputs = append(opts.DefaultOutputs, name)
	}
	opts.Apply = s.apply
	opts.GetHandler = s.getHandler

	meta := &NativeMetadataResponse{}
	meta.Name = "SimpleModel"
	meta.Description = "A graphpipe server using the simple interface with automatic native type conversion."

	for i := 0; i < len(inputTensors); i++ {
		name := "input" + strconv.Itoa(i)
		t := TensorToNativeTensor(inputTensors[i])
		io := NativeIOMetadata{}
		io.Name = name
		io.Type = t.Type
		io.Shape = t.Shape
		meta.Inputs = append(meta.Inputs, io)
	}

	for i := 0; i < len(outputTensors); i++ {
		name := "output" + strconv.Itoa(i)
		t := TensorToNativeTensor(outputTensors[i])
		io := NativeIOMetadata{}
		io.Name = name
		io.Type = t.Type
		io.Shape = t.Shape
		meta.Outputs = append(meta.Outputs, io)
	}
	opts.Meta = meta
	s.rawOpts = opts
	return opts
}

func (c *simpleContext) apply(requestContext *RequestContext, config string, inputs map[string]*NativeTensor, outputNames []string) ([]*NativeTensor, error) {
	inVals := make([]interface{}, len(inputs)+2) // add 2 for requestContext and config
	inVals[0] = requestContext
	inVals[1] = config

	if len(inputs) > 0 {
		if len(inputs) != c.numInputs {
			return nil, fmt.Errorf("%d inputs sent but %d are required", len(inputs), c.numInputs)
		}
		for k := range inputs {
			trimmed := strings.TrimPrefix(k, "input")
			if trimmed == k {
				return nil, fmt.Errorf("Unexpected input for '%s' was sent", k)
			}
			i, err := strconv.Atoi(trimmed)
			if err != nil {
				return nil, fmt.Errorf("Unexpected input for '%s' was sent", k)
			}
			if len(c.inShapes) > i {
				if len(c.inShapes[i]) != 0 && len(c.inShapes[i]) != len(inputs[k].Shape) {
					return nil, fmt.Errorf("%s doesn't have shape %v", k, c.inShapes[i])
				}
				for j := range c.inShapes[i] {
					if c.inShapes[i][j] != -1 && c.inShapes[i][j] != inputs[k].Shape[j] {
						return nil, fmt.Errorf("%s doesn't have shape %v", k, c.inShapes[i])
					}
				}
			}
			inVals[i+2], err = NativeTensorToNative(inputs[k])
			if err != nil {
				return nil, err
			}
		}
	}

	results, err := safeCallApply(inVals, c.wrapped)
	if err != nil {
		return nil, err
	}

	outputs := []*NativeTensor{}
	for _, result := range results {
		output := &NativeTensor{}
		output.InitSimple(result)
		outputs = append(outputs, output)
	}
	return outputs, err
}

func safeCallApply(inVals []interface{}, apply interface{}) (results []interface{}, err error) {
	defer func() {
		if r := recover(); r != nil {
			for _, line := range strings.Split(string(debug.Stack()), "\n") {
				logrus.Errorf(line)
			}
			err = fmt.Errorf("Failed to call apply: %v", r)
		}
	}()
	fn := reflect.ValueOf(apply)
	mtyp := fn.Type()

	inputs := make([]reflect.Value, len(inVals))
	nin := mtyp.NumIn()
	if nin != len(inVals) {
		return nil, fmt.Errorf("wrong number of inputs supplied")
	}
	for i := range inVals {
		typ := mtyp.In(i)
		if i == 1 && reflect.ValueOf(inVals[i]).Type().Kind() == reflect.String && typ.Kind() != reflect.String {
			// If unmarshal is passed an interface{}, it must wrap a pointer or else
			// it unmarshals into a map, so we use new here
			p := reflect.New(typ).Interface()
			if err := json.Unmarshal([]byte(inVals[i].(string)), &p); err != nil {
				return nil, err
			}
			// The function doesn't necessarily take a pointer, so unwrap it
			inVals[i] = reflect.ValueOf(p).Elem().Interface()
		}
		inputs[i] = reflect.ValueOf(inVals[i])
	}

	vals := fn.Call(inputs)
	results = []interface{}{}
	for _, val := range vals {
		if val.Type().Implements(errorInterface) {
			if err, ok := val.Interface().(error); ok {
				return nil, err
			}
		} else {
			results = append(results, val.Interface())
		}
	}
	return results, err
}
