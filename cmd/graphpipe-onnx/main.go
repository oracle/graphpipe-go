package main

import (
	//  "bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/Sirupsen/logrus"
	"github.com/spf13/cobra"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"runtime/pprof"
	"strings"
	"sync"
	"time"
	"unsafe"
	//graphpipe "github.com/oracle/graphpipe-go"
	graphpipe "github.com/oracle/graphpipe-go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
	//	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

// #cgo CXXFLAGS: -I /pytorch/build/third_party/onnx/ -I /pytorch/third_party/onnx -I /pytorch/third_party/protobuf/src/  -I /pytorch/ -I /pytorch/build -I /pytorch/build/third_party/gloo -DONNX_NAMESPACE=onnx_c2 -D_GNU_SOURCE -D_REENTRANT -D CAFFE2_FORCE_FALLBACK_CUDA_MPI -D CAFFE2_PERF_WITH_AVX -D CAFFE2_PERF_WITH_AVX2 -D CAFFE2_USE_EIGEN_FOR_BLAS -D CAFFE2_USE_GFLAGS -D CAFFE2_USE_GOOGLE_GLOG -D EIGEN_MPL2_ONLY -D NDEBUG -isystem /usr/local/cuda/include -std=c++11
// #cgo LDFLAGS: -ldl /pytorch/build/lib/libcaffe2_gpu.so  /pytorch/build/lib/libcaffe2.so  /usr/local/cuda/lib64/libcurand.so -lprotobuf -lglog -lgflags  -L/usr/local/cuda/lib64 -lcudart
// #include <stdlib.h>
// #include <c2_api.h>
import "C"

var (
	ver string
	sha string
)

func version() string {
	if ver == "" {
		ver = "dev"
		sha = "unknown"
	}
	return fmt.Sprintf("version %s (built from sha %s)", ver, sha)
}

type options struct {
	model       string
	listen      string
	stateDir    string
	verbose     bool
	version     bool
	cache       bool
	cuda        bool
	valueInputs string
	initNet     string
	predictNet  string
	profile     string
	engineCount int
}

func loadFile(uri string) ([]byte, error) {
	logrus.Infof("Loading file ", uri)
	if strings.HasPrefix(uri, "http://") ||
		strings.HasPrefix(uri, "https://") {
		var transport = &http.Transport{
			Dial: (&net.Dialer{
				Timeout: 10 * time.Second,
			}).Dial,
			TLSHandshakeTimeout: 10 * time.Second,
		}
		var client = &http.Client{
			Timeout:   time.Second * 60,
			Transport: transport,
		}
		response, err := client.Get(uri)
		if err != nil {
			logrus.Errorf("Failed to get '%s': %v", uri, err)
			return nil, err
		}
		if response.StatusCode != 200 {
			logrus.Fatalf("Failed to load file: %s, status code: %v", uri, response.StatusCode)
		}
		return ioutil.ReadAll(response.Body)
	}
	return ioutil.ReadFile(uri)
}

func main() {
	var cmdExitCode int
	var opts options
	cmd := cobra.Command{
		Use:   "graphpipe-caffe2",
		Short: "graphpipe-caffe2 - serving up delectable caffe2 and onnx models",
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			if opts.verbose {
				logrus.SetLevel(logrus.DebugLevel)
			} else {
				logrus.SetLevel(logrus.InfoLevel)
			}
		},
		Run: func(cmd *cobra.Command, args []string) {
			if opts.version {
				fmt.Printf("%s\n", version())
				return
			}
			if len(args) != 0 {
				cmdExitCode = 1
				cmd.Usage()
				return
			}
			if opts.model == "" {
				if opts.initNet == "" && opts.predictNet == "" {
					logrus.Errorf("either --model/env(\"GP_MODEL\") must be set or --init_net/env(GP_INIT_NET) AND --predict_net/env(GP_PREDICT_NET) must be set.")
					cmdExitCode = 1
					return
				}
			}
			if opts.valueInputs == "" {
				logrus.Errorf("--value_inputs or env(\"GP_VALUE_INPUTS\") must be set")
				cmdExitCode = 1
				return
			}

			if opts.profile != "" {
				f, err := os.Create(opts.profile)
				if err != nil {
					logrus.Fatal(err)
				}
				pprof.StartCPUProfile(f)
				c := make(chan os.Signal, 1)
				signal.Notify(c, os.Interrupt)
				go func() {
					for _ = range c {
						pprof.StopCPUProfile()
						os.Exit(0)
					}
				}()
			}

			logrus.Infof("Starting graphpipe-caffe2 %s", version())

			if err := serve(opts); err != nil {
				logrus.Errorf("Failed to serve: %v", err)
				cmdExitCode = 1
			}
			if opts.profile != "" {
			}
		},
	}
	f := cmd.Flags()
	f.StringVarP(&opts.stateDir, "dir", "d", "~/.tureen", "dir for local state")
	f.StringVarP(&opts.listen, "listen", "l", "127.0.0.1:9000", "listen string")
	f.BoolVarP(&opts.verbose, "verbose", "v", false, "verbose output")
	f.BoolVarP(&opts.version, "version", "V", false, "show version")
	f.StringVarP(&opts.model, "model", "m", "", "model to load")
	f.StringVarP(&opts.initNet, "init_net", "", "", "init_net file to load")
	f.StringVarP(&opts.predictNet, "predict_net", "", "", "predict_net file to load")
	f.StringVarP(&opts.valueInputs, "value_inputs", "", "", "inputs")
	f.BoolVarP(&opts.cache, "cache", "", false, "cache results")
	f.BoolVarP(&opts.cuda, "cuda", "", false, "Use Cuda")
	f.StringVarP(&opts.profile, "profile", "", "", "profile and write profiling output to this file")
	f.IntVarP(&opts.engineCount, "engine_count", "", 1, "Engine Count")

	opts.stateDir = strings.Replace(opts.stateDir, "~", os.Getenv("HOME"), -1)

	if opts.model == "" {
		opts.model = os.Getenv("GP_MODEL")
	}
	if opts.initNet == "" {
		opts.initNet = os.Getenv("GP_INIT_NET")
	}
	if opts.predictNet == "" {
		opts.predictNet = os.Getenv("GP_PREDICT_NET")
	}
	if opts.valueInputs == "" {
		opts.valueInputs = os.Getenv("GP_VALUE_INPUTS")
	}

	if os.Getenv("GP_CACHE") != "" {
		val := strings.ToLower(os.Getenv("GP_CACHE"))
		if val == "1" || val == "true" {
			opts.cache = true
		}
	}
	cmd.Execute()
	os.Exit(cmdExitCode)
}

type c2Context struct {
	Mutex       sync.Mutex
	EngineCount int
	CEngineCtxs []*C.c2_engine_ctx
	InputDims   map[string][]int64
	OutputDims  map[string][]int64
	Inputs      []string
	Outputs     []string
	modelHash   []byte
	meta        *graphpipe.NativeMetadataResponse
}

func readModel(uri string) ([]byte, error) {
	if strings.HasPrefix(uri, "http://") ||
		strings.HasPrefix(uri, "https://") {
		var transport = &http.Transport{
			Dial: (&net.Dialer{
				Timeout: 5 * time.Second,
			}).Dial,
			TLSHandshakeTimeout: 5 * time.Second,
		}
		var client = &http.Client{
			Timeout:   time.Second * 10,
			Transport: transport,
		}
		response, err := client.Get(uri)
		if err != nil {
			logrus.Errorf("Failed to get '%s': %v", uri, err)
			return nil, err
		}
		return ioutil.ReadAll(response.Body)
	}
	return ioutil.ReadFile(uri)
}

func serve(opts options) error {
	c2c := &c2Context{}

	useCuda := 0
	if opts.cuda {
		useCuda = 1
	}

	c2c.EngineCount = opts.engineCount
	for i := 0; i < c2c.EngineCount; i++ {
		engine_ctx := C.c2_engine_create(C.int(useCuda))
		c2c.CEngineCtxs = append(c2c.CEngineCtxs, engine_ctx)
	}

	valueInputData := make(map[string]interface{})

	err := json.Unmarshal([]byte(opts.valueInputs), &valueInputData)
	if err != nil {
		panic(err)
	}

	for k, v := range valueInputData {
		if reflect.ValueOf(v).Len() == 2 {
			dims := []int64{}
			dtype := reflect.ValueOf(v).Index(0).Interface().(float64)
			shape := reflect.ValueOf(v).Index(1).Elem()
			for i := 0; i < shape.Len(); i++ {
				d := int64(shape.Index(i).Interface().(float64))
				dims = append(dims, d)
			}
			for _, engine_ctx := range c2c.CEngineCtxs {
				C.c2_engine_register_input(engine_ctx, C.CString(k), (*C.int64_t)(&dims[0]), (C.int(len(dims))), C.int(dtype))
			}
		}
	}

	argv := (C.malloc(C.size_t(len(os.Args)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
	tmp := (*[1<<30 - 1]*C.char)(argv)
	for i, s := range os.Args {
		tmp[i] = C.CString(s)
	}

	if opts.model != "" {
		modelData, err := readModel(opts.model)
		if err != nil {
			logrus.Errorln("Could not read file ", opts.model)
			return err
		}

		for _, engine_ctx := range c2c.CEngineCtxs {
			status := C.c2_engine_initialize_onnx(engine_ctx, (*C.char)(unsafe.Pointer((&modelData[0]))), C.size_t(len(modelData)))
			if status != 0 {
				logrus.Fatalf("Could not initialize engine\n")
			}
		}

		h := sha256.New()
		h.Write(modelData)
		c2c.modelHash = h.Sum(nil)
		logrus.Infof("Model hash is '%x'", c2c.modelHash)
	} else {
		initData, err := readModel(opts.initNet)
		if err != nil {
			logrus.Errorln("Could not read init file ", opts.initNet)
			return err
		}

		predData, err := readModel(opts.predictNet)
		if err != nil {
			logrus.Errorln("Could not read predict file ", opts.predictNet)
			return err
		}

		for _, engine_ctx := range c2c.CEngineCtxs {
			status := C.c2_engine_initialize_caffe2(engine_ctx, (*C.char)(unsafe.Pointer((&initData[0]))), C.size_t(len(initData)), (*C.char)(unsafe.Pointer((&predData[0]))), C.size_t(len(predData)))
			if status != 0 {
				logrus.Fatalf("Could not initialize engine\n")
			}
		}

		h := sha256.New()
		h.Write(initData)
		h.Write(predData)
		c2c.modelHash = h.Sum(nil)
		logrus.Infof("Model hash is '%x'", c2c.modelHash)
	}

	inputCount := C.c2_engine_get_input_count(c2c.CEngineCtxs[0])
	c2c.Inputs = make([]string, inputCount)
	c2c.InputDims = make(map[string][]int64)
	c2c.OutputDims = make(map[string][]int64)
	meta := &graphpipe.NativeMetadataResponse{}

	for i := 0; i < int(inputCount); i++ {
		cname := C.c2_engine_get_input_name(c2c.CEngineCtxs[0], C.int(i))
		name := C.GoString(cname)
		c2c.Inputs[i] = name
		dims := make([]int64, 32)
		count := C.c2_engine_get_dimensions(c2c.CEngineCtxs[0], cname, (*C.int64_t)(unsafe.Pointer(&dims[0])))
		c2c.InputDims[name] = dims[:count]
		dtype := C.c2_engine_get_dtype(c2c.CEngineCtxs[0], cname)

		logrus.Debugf("Adding input '%s'", name)

		io := graphpipe.NativeIOMetadata{}
		io.Name = name
		io.Shape = dims[:count]
		io.Type = ctype2gptype[dtype]
		meta.Inputs = append(meta.Inputs, io)

	}

	outputCount := C.c2_engine_get_output_count(c2c.CEngineCtxs[0])
	c2c.Outputs = make([]string, outputCount)
	for i := 0; i < int(outputCount); i++ {
		cname := C.c2_engine_get_output_name(c2c.CEngineCtxs[0], C.int(i))
		name := C.GoString(cname)
		c2c.Outputs[i] = name
		dims := make([]int64, 32)
		count := C.c2_engine_get_dimensions(c2c.CEngineCtxs[0], cname, (*C.int64_t)(unsafe.Pointer(&dims[0])))
		c2c.OutputDims[name] = dims[:count]
		dtype := C.c2_engine_get_dtype(c2c.CEngineCtxs[0], cname)

		logrus.Debugf("Adding output '%s'", name)
		io := graphpipe.NativeIOMetadata{}
		io.Name = name
		io.Shape = dims[:count]
		io.Type = ctype2gptype[dtype]
		meta.Outputs = append(meta.Outputs, io)
	}

	meta.Name = opts.model
	meta.Description = "Implementation of tensorflow model server using graphpipe.  Use a graphpipe client to make requests to this server."
	meta.Server = "graphpipe-tf"
	meta.Version = version()

	c2c.meta = meta

	cachePath := ""
	if opts.cache {
		cachePath = filepath.Join(opts.stateDir, fmt.Sprintf("%x.db", c2c.modelHash))
	}

	serveOpts := &graphpipe.ServeRawOptions{
		Listen:    opts.listen,
		CacheFile: cachePath,
		Meta:      c2c.meta,
		//		DefaultInputs:  dIn,
		//		DefaultOutputs: dOut,
		Apply:      c2c.apply,
		GetHandler: c2c.getHandler,
	}
	if err := graphpipe.ServeRaw(serveOpts); err != nil {
		return err
	}
	return nil
}

func (c2c *c2Context) getHandler(w http.ResponseWriter, r *http.Request, body []byte) error {
	js, err := json.MarshalIndent(c2c.meta, "", "    ")
	if err == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(js)
	}
	return err
}

var gptype2ctype = []int{
	C.TensorProto_DataType_UNDEFINED, // Type_Null = 0,
	C.TensorProto_DataType_UINT8,     // Type_Uint8 = 1,
	C.TensorProto_DataType_INT8,      // Type_Int8 = 2,
	C.TensorProto_DataType_UINT16,    // Type_Uint16 = 3,
	C.TensorProto_DataType_INT16,     // Type_Int16 = 4,
	C.TensorProto_DataType_UNDEFINED, // Type_Uint32 = 5,
	C.TensorProto_DataType_INT32,     // Type_Int32 = 6,
	C.TensorProto_DataType_UNDEFINED, // Type_Uint64 = 7,
	C.TensorProto_DataType_INT64,     // Type_Int64 = 8,
	C.TensorProto_DataType_FLOAT16,   // Type_Float16 = 9,
	C.TensorProto_DataType_FLOAT,     // Type_Float32 = 10,
	C.TensorProto_DataType_DOUBLE,    // Type_Float64 = 11,
	C.TensorProto_DataType_STRING,    // Type_String = 12,
}

var ctype2gptype = []uint8{
	graphpipefb.TypeNull,    // TensorProto_DataType_UNDEFINED = 0,
	graphpipefb.TypeFloat32, // TensorProto_DataType_FLOAT = 1,
	graphpipefb.TypeInt32,   // TensorProto_DataType_INT32 = 2,
	graphpipefb.TypeInt8,    // TensorProto_DataType_BYTE = 3,
	graphpipefb.TypeString,  // TensorProto_DataType_STRING = 4,
	graphpipefb.TypeNull,    // TensorProto_DataType_BOOL = 5,
	graphpipefb.TypeInt8,    // TensorProto_DataType_UINT8 = 6,
	graphpipefb.TypeInt8,    // TensorProto_DataType_INT8 = 7,
	graphpipefb.TypeUint16,  // TensorProto_DataType_UINT16 = 8,
	graphpipefb.TypeNull,    // TensorProto_DataType_INT16 = 9,
	graphpipefb.TypeInt64,   // TensorProto_DataType_INT64 = 10,
	graphpipefb.TypeFloat16, // TensorProto_DataType_FLOAT16 = 12,
	graphpipefb.TypeFloat64, // TensorProto_DataType_DOUBLE = 13
}

func (c2c *c2Context) apply(requestContext *graphpipe.RequestContext, config string, inputs map[string]*graphpipe.NativeTensor, outputNames []string) ([]*graphpipe.NativeTensor, error) {
	var engine_ctx *C.c2_engine_ctx
	if c2c.EngineCount == 1 {
		c2c.Mutex.Lock()
		engine_ctx = c2c.CEngineCtxs[0]
		requestContext.CleanupFunc = func() {
			c2c.Mutex.Unlock()
		}
	} else {
		for {
			c2c.Mutex.Lock()
			if len(c2c.CEngineCtxs) > 0 {
				engine_ctx = c2c.CEngineCtxs[0]
				c2c.CEngineCtxs = c2c.CEngineCtxs[1:]
				c2c.Mutex.Unlock()
				break
			}
			c2c.Mutex.Unlock()
			time.Sleep(time.Second / 2000)
		}

		requestContext.CleanupFunc = func() {
			c2c.Mutex.Lock()
			c2c.CEngineCtxs = append(c2c.CEngineCtxs, engine_ctx)
			c2c.Mutex.Unlock()
		}
	}

	outputNts := make([]*graphpipe.NativeTensor, len(outputNames))
	for name, input := range inputs {

		cname := C.CString(name)
		defer C.free(unsafe.Pointer(cname))

		if gptype2ctype[input.Type] != int(C.c2_engine_get_dtype(engine_ctx, cname)) {
			logrus.Errorf("Input type mismatch.  Got %d expected %d", gptype2ctype[input.Type], C.c2_engine_get_dtype(engine_ctx, cname))
			return nil, nil
		}

		size := len(input.Data)
		C.c2_set_input_batch(engine_ctx, cname, unsafe.Pointer(&input.Data[0]), C.int(size))
	}

	C.c2_execute_batch(engine_ctx)
	for i, name := range outputNames {
		cname := C.CString(name)
		defer C.free(unsafe.Pointer(cname))
		idx := C.c2_engine_get_output_index(engine_ctx, cname)
		size := C.c2_engine_get_output_size(engine_ctx, idx)
		itemsize := int64(C.c2_engine_get_itemsize(engine_ctx, cname))
		dtype := int(C.c2_engine_get_dtype(engine_ctx, cname))
		buf := make([]byte, size)
		actual := C.c2_engine_get_output(engine_ctx, idx, unsafe.Pointer(&buf[0]))
		if actual != size {
			logrus.Errorf("Returned size mismatch: %s != %s", size, actual)
			return nil, nil
		}

		dims := c2c.OutputDims[name]

		shape := make([]int64, len(dims))

		rowSize := int64(1)
		for j, d := range dims {
			rowSize *= int64(d)
			shape[j] = int64(d)
		}

		gptype := -1
		for gp, dt := range gptype2ctype {
			if dt == dtype {
				gptype = gp
			}
		}

		if gptype < 0 {
			logrus.Errorf("Unhandled type %d", dtype)
			return nil, nil
		}

		shape[0] = int64(size) / (rowSize * itemsize)
		logrus.Println(size)
		logrus.Println(rowSize)
		logrus.Println(dims)

		nt := &graphpipe.NativeTensor{}
		nt.InitWithData(buf, shape, uint8(gptype))
		outputNts[i] = nt
	}
	return outputNts, nil
}
