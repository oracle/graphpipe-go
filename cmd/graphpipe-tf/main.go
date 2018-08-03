package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/spf13/cobra"

	graphpipe "github.com/oracle/graphpipe-go"
	tfproto "github.com/oracle/graphpipe-go/cmd/graphpipe-tf/internal/github.com/tensorflow/tensorflow/tensorflow/go/core/framework"
	cproto "github.com/oracle/graphpipe-go/cmd/graphpipe-tf/internal/github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

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
	verbose  bool
	version  bool
	cache    bool
	stateDir string
	listen   string
	model    string
	inputs   string
	shape    string
	outputs  string
}

func main() {
	var opts options
	var cmdExitCode int

	cmd := cobra.Command{
		Use:   "graphpipe-tf",
		Short: "graphpipe-tf - serving up ml models",
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
				logrus.Errorf("--model or env(\"GP_MODEL\") must be set")
				cmdExitCode = 1
				return
			}

			logrus.Infof("Starting graphpipe-tf %s", version())

			if err := serve(opts); err != nil {
				logrus.Errorf("Failed to serve: %v", err)
				cmdExitCode = 1
			}
		},
	}
	f := cmd.Flags()
	f.StringVarP(&opts.stateDir, "dir", "d", "~/.graphpipe-tf", "dir for local cache state")
	f.StringVarP(&opts.listen, "listen", "l", "127.0.0.1:9000", "listen string")
	f.StringVarP(&opts.model, "model", "m", "", "tensorflow model to load (accepts local files and unauthenticated http/https urls)")
	f.StringVarP(&opts.inputs, "inputs", "i", "", "comma seprated default inputs")
	f.StringVarP(&opts.outputs, "outputs", "o", "", "comma separated default outputs")
	f.BoolVarP(&opts.cache, "cache", "c", false, "cache results")
	f = cmd.PersistentFlags()
	f.BoolVarP(&opts.verbose, "verbose", "v", false, "verbose output")
	f.BoolVarP(&opts.version, "version", "V", false, "show version")

	opts.stateDir = strings.Replace(opts.stateDir, "~", os.Getenv("HOME"), -1)
	if opts.model == "" {
		opts.model = os.Getenv("GP_MODEL")
	}
	opts.model = strings.Replace(opts.model, "~", os.Getenv("HOME"), -1)
	if opts.outputs == "" {
		opts.outputs = os.Getenv("GP_OUTPUTS")
	}
	if opts.inputs == "" {
		opts.inputs = os.Getenv("GP_INPUTS")
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

type tfContext struct {
	modelHash []byte
	graphDef  tfproto.GraphDef

	model *tf.SavedModel

	meta    *graphpipe.NativeMetadataResponse
	outputs map[string]tf.Output
	names   []string
	types   []byte
	shapes  [][]int64
}

func getSessionOpts() (*tf.SessionOptions, error) {
	config := cproto.ConfigProto{}
	config.GpuOptions = &cproto.GPUOptions{}
	config.GpuOptions.AllowGrowth = true

	data, err := config.Marshal()
	if err != nil {
		return nil, err
	}
	return &tf.SessionOptions{Config: data}, nil
}

func initializeMetadata(opts options, c *tfContext) *graphpipe.NativeMetadataResponse {
	c.outputs = map[string]tf.Output{}
	ops := c.model.Graph.Operations()
	opMap := map[string]tf.Operation{}
	for _, op := range ops {
		opMap[op.Name()] = op
	}
	for _, node := range c.graphDef.Node {
		op := opMap[node.Name]
		num := op.NumOutputs()
		for i := 0; i < num; i++ {
			name := node.Name + ":" + strconv.Itoa(i)
			c.names = append(c.names, name)
			output := op.Output(i)
			c.outputs[name] = output
			t := toFlatDtype(output.DataType())
			if t == graphpipefb.TypeNull {
				logrus.Debugf("Unknown type for '%s': %v", name, output.DataType())
			}
			c.types = append(c.types, t)
			s := output.Shape()
			shape, err := s.ToSlice()
			if err != nil {
				logrus.Debugf("Unknown shape for '%s'", name)
			}
			c.shapes = append(c.shapes, shape)
		}
	}
	meta := &graphpipe.NativeMetadataResponse{}
	meta.Name = opts.model
	meta.Description = "Implementation of tensorflow model server using graphpipe.  Use a graphpipe client to make requests to this server."
	meta.Server = "graphpipe-tf"
	meta.Version = version()

	for i := range c.names {
		// don't allow unknown types to be used for input and output metadata
		if c.types[i] == graphpipefb.TypeNull {
			continue
		}
		io := graphpipe.NativeIOMetadata{}
		logrus.Debugf("Adding input/output '%s'", c.names[i])
		io.Name = c.names[i]
		io.Shape = c.shapes[i]
		io.Type = c.types[i]
		meta.Inputs = append(meta.Inputs, io)
		meta.Outputs = append(meta.Outputs, io)
	}
	return meta
}

func loadModel(uri string) (*tf.SavedModel, []byte, error) {
	sessionOpts, err := getSessionOpts()
	if err != nil {
		logrus.Errorf("Could not create session opts: %v", err)
		return nil, nil, err
	}
	if info, err := os.Stat(uri); err == nil && info.IsDir() {
		savedModel, err := tf.LoadSavedModel(uri, []string{"serve"}, sessionOpts)
		if err != nil {
			logrus.Errorf("Failed to load '%s': %v", uri, err)
			return nil, nil, err
		}
		buf := &bytes.Buffer{}
		savedModel.Graph.WriteTo(buf)
		return savedModel, buf.Bytes(), nil
	}
	b, err := readModel(uri)
	if err != nil {
		logrus.Errorf("Could not read '%s': %v", uri, err)
		return nil, nil, err
	}

	graph := tf.NewGraph()
	if err := graph.Import(b, ""); err != nil {
		logrus.Errorf("Could not import graph: %v", err)
		return nil, nil, err
	}

	s, err := tf.NewSession(graph, sessionOpts)
	if err != nil {
		logrus.Errorf("Could not create session: %v", err)
		return nil, nil, err
	}

	return &tf.SavedModel{Session: s, Graph: graph}, b, nil
}

func readModel(uri string) ([]byte, error) {
	if strings.HasPrefix(uri, "http://") ||
		strings.HasPrefix(uri, "https://") {
		var transport = &http.Transport{
			Dial: (&net.Dialer{
				Timeout: 5 * time.Second,
			}).Dial,
			Proxy:               http.ProxyFromEnvironment,
			TLSHandshakeTimeout: 5 * time.Second,
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
		return ioutil.ReadAll(response.Body)
	}
	return ioutil.ReadFile(uri)
}

func graphContainsNode(g tfproto.GraphDef, name string) bool {
	name = strings.Split(name, ":")[0]
	for _, n := range g.Node {
		if n.Name == name {
			return true
		}
	}
	return false
}

func serve(opts options) error {
	if err := os.MkdirAll(opts.stateDir, 0700); err != nil {
		logrus.Errorf("Could not make state dir '%s': %v", opts.stateDir, err)
		return err
	}

	c := &tfContext{}

	var serialized []byte
	var err error
	c.model, serialized, err = loadModel(opts.model)
	if err != nil {
		logrus.Errorf("Failed to load '%s': %v", opts.model, err)
		return err
	}

	if err := c.graphDef.Unmarshal(serialized); err != nil {
		logrus.Errorf("Could not load graph_def: %v", err)
		return err
	}

	// reserialize the graph with stable marshal so the hash is the same
	buf, _ := c.graphDef.Marshal()
	h := sha256.New()
	h.Write(buf)
	c.modelHash = h.Sum(nil)
	logrus.Infof("Model hash is '%x'", c.modelHash)

	cachePath := ""
	if opts.cache {
		cachePath = filepath.Join(opts.stateDir, fmt.Sprintf("%x.db", c.modelHash))
	}

	c.meta = initializeMetadata(opts, c)

	first := c.graphDef.Node[0].Name + ":0"
	last := c.graphDef.Node[len(c.graphDef.Node)-1].Name + ":0"

	dIn := strings.Split(opts.inputs, ",")
	for i := range dIn {
		if dIn[i] == "" {
			dIn[i] = first
		}
		if !strings.Contains(dIn[i], ":") {
			dIn[i] += ":0"
		}
	}

	dOut := strings.Split(opts.outputs, ",")
	for i := range dOut {
		if dOut[i] == "" {
			dOut[i] = last
		}
		if !strings.Contains(dOut[i], ":") {
			dOut[i] += ":0"
		}
	}

	missingIO := false

	for _, inp := range dIn {
		if !graphContainsNode(c.graphDef, inp) {
			logrus.Errorf("Couldn't find input in graph: %s", inp)
			missingIO = true
		}
	}

	for _, outp := range dOut {
		if !graphContainsNode(c.graphDef, outp) {
			logrus.Errorf("Couldn't find output in graph: %s", outp)
			missingIO = true
		}
	}

	if missingIO {
		return fmt.Errorf("Could not find some inputs and/or outputs.  Aborting.")
	}

	logrus.Infof("Using default inputs %s", dIn)
	logrus.Infof("Using default outputs %s", dOut)

	serveOpts := &graphpipe.ServeRawOptions{
		Listen:         opts.listen,
		CacheFile:      cachePath,
		Meta:           c.meta,
		DefaultInputs:  dIn,
		DefaultOutputs: dOut,
		Apply:          c.apply,
		GetHandler:     c.getHandler,
	}

	return graphpipe.ServeRaw(serveOpts)
}

var conv2flat = []byte{
	graphpipefb.TypeNull,    // DT_INVALID = 0;
	graphpipefb.TypeFloat32, // DT_FLOAT = 1;
	graphpipefb.TypeFloat64, // DT_DOUBLE = 2;
	graphpipefb.TypeInt32,   // DT_INT32 = 3;
	graphpipefb.TypeUint8,   // DT_UINT8 = 4;
	graphpipefb.TypeInt16,   // DT_INT16 = 5;
	graphpipefb.TypeInt8,    // DT_INT8 = 6;
	graphpipefb.TypeString,  // DT_STRING = 7;
	graphpipefb.TypeNull,    // DT_COMPLEX64 = 8;  // Single-precision complex
	graphpipefb.TypeInt64,   // DT_INT64 = 9;
	graphpipefb.TypeNull,    // DT_BOOL = 10;
	graphpipefb.TypeNull,    // DT_QINT8 = 11;     // Quantized int8
	graphpipefb.TypeNull,    // DT_QUINT8 = 12;    // Quantized uint8
	graphpipefb.TypeNull,    // DT_QINT32 = 13;    // Quantized int32
	graphpipefb.TypeNull,    // DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
	graphpipefb.TypeNull,    // DT_QINT16 = 15;    // Quantized int16
	graphpipefb.TypeNull,    // DT_QUINT16 = 16;   // Quantized uint16
	graphpipefb.TypeUint16,  // DT_UINT16 = 17;
	graphpipefb.TypeNull,    // DT_COMPLEX128 = 18;  // Double-precision complex
	graphpipefb.TypeFloat16, // DT_HALF = 19;
	graphpipefb.TypeNull,    // DT_RESOURCE = 20;
	graphpipefb.TypeNull,    // DT_VARIANT = 21;  // Arbitrary C++ data types
	graphpipefb.TypeUint32,  // DT_UINT32 = 22;
	graphpipefb.TypeUint64,  // DT_UINT64 = 23;
}

func toFlatDtype(dt tf.DataType) byte {
	if int(dt) >= len(conv2flat) {
		return graphpipefb.TypeNull
	}
	return conv2flat[dt]
}

func ntFromTensor(tensor *tf.Tensor) *graphpipe.NativeTensor {
	nt := &graphpipe.NativeTensor{}
	nt.Type = conv2flat[tensor.DataType()]
	for _, size := range tensor.Shape() {
		nt.Shape = append(nt.Shape, int64(size))
	}
	// NOTE(vish): this could support multidimensional strings if we
	//             had a way to flatten the tensor
	if nt.Type == graphpipefb.TypeString {
		for _, s := range tensor.Value().([]string) {
			nt.StringVals = append(nt.StringVals, s)
		}
	} else {
		buf := bytes.Buffer{}
		tensor.WriteContentsTo(&buf)
		nt.Data = buf.Bytes()
	}
	return nt
}

func getOutputRequests(c *tfContext, outputNames []string) ([]tf.Output, error) {
	outputRequests := []tf.Output{}
	for _, name := range outputNames {
		out, ok := c.outputs[name]
		if !ok {
			msg := "Could not find output '%s'"
			logrus.Errorf(msg, name)
			return nil, fmt.Errorf(msg, name)
		}
		outputRequests = append(outputRequests, out)
	}
	return outputRequests, nil
}

var gptype2tftype = []tf.DataType{
	tf.DataType(tfproto.DataType_DT_INVALID), // Type_Null = 0,
	tf.DataType(tfproto.DataType_DT_UINT8),   // Type_Uint8 = 1,
	tf.DataType(tfproto.DataType_DT_INT8),    // Type_Int8 = 2,
	tf.DataType(tfproto.DataType_DT_UINT16),  // Type_Uint16 = 3,
	tf.DataType(tfproto.DataType_DT_INT16),   // Type_Int16 = 4,
	tf.DataType(tfproto.DataType_DT_UINT32),  // Type_Uint32 = 5,
	tf.DataType(tfproto.DataType_DT_INT32),   // Type_Int32 = 6,
	tf.DataType(tfproto.DataType_DT_UINT64),  // Type_Uint64 = 7,
	tf.DataType(tfproto.DataType_DT_INT64),   // Type_Int64 = 8,
	tf.DataType(tfproto.DataType_DT_HALF),    // Type_Float16 = 9,
	tf.DataType(tfproto.DataType_DT_FLOAT),   // Type_Float32 = 10,
	tf.DataType(tfproto.DataType_DT_DOUBLE),  // Type_Float64 = 11,
	tf.DataType(tfproto.DataType_DT_STRING),  // Type_String = 12,
}

func tensorFromNT(nt *graphpipe.NativeTensor) (*tf.Tensor, error) {
	if nt.Type == graphpipefb.TypeString {
		return tf.NewTensor(nt.StringVals)
	}
	dtype := gptype2tftype[nt.Type]
	return tf.ReadTensor(dtype, nt.Shape, bytes.NewReader(nt.Data))
}

func getInputMap(c *tfContext, inputs map[string]*graphpipe.NativeTensor) (map[tf.Output]*tf.Tensor, error) {
	inputMap := map[tf.Output]*tf.Tensor{}
	for name, input := range inputs {
		inputTensor, err := tensorFromNT(input)
		if err != nil {
			logrus.Errorf("Failed to create tensor: %v", err)
			return nil, err
		}
		output := tf.Output{}
		var ok bool
		if !strings.Contains(name, ":") {
			name += ":0"
		}
		output, ok = c.outputs[name]
		if !ok {
			msg := "Could not find input '%s'"
			logrus.Errorf(msg, name)
			return nil, fmt.Errorf(msg, name)
		}
		inputMap[output] = inputTensor
	}
	return inputMap, nil
}

func (tfc *tfContext) apply(requestContext *graphpipe.RequestContext, config string, inputs map[string]*graphpipe.NativeTensor, outputNames []string) ([]*graphpipe.NativeTensor, error) {
	outputIndexes := []int{}
	outputTps := make([]*graphpipe.NativeTensor, len(outputNames))

	for i := range outputNames {
		outputIndexes = append(outputIndexes, i)
	}
	keys := make([]string, 0, len(inputs))
	for k := range inputs {
		keys = append(keys, k)
	}
	outputRequests, err := getOutputRequests(tfc, outputNames)
	if err != nil {
		return nil, err
	}
	inputMap, err := getInputMap(tfc, inputs)
	if err != nil {
		return nil, err
	}
	tensors, err := tfc.model.Session.Run(
		inputMap,
		outputRequests,
		nil,
	)
	if err != nil {
		logrus.Errorf("Failed to run session: %v", err)
		return nil, err
	}
	for i := range outputIndexes {
		nt := ntFromTensor(tensors[i])
		outputTps[outputIndexes[i]] = nt
	}
	return outputTps, nil
}

func (tfc *tfContext) getHandler(w http.ResponseWriter, r *http.Request, body []byte) error {
	js, err := json.MarshalIndent(tfc.meta, "", "    ")
	if err == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(js)
	}
	return err
}
