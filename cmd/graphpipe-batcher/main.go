package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/Sirupsen/logrus"
	"github.com/spf13/cobra"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	graphpipe "github.com/oracle/graphpipe-go"
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
	verbose   bool
	version   bool
	cache     bool
	cacheDir  string
	listen    string
	inputs    string
	outputs   string
	targetURL string
	batchSize int
	timeout   int
	workers   int
}

func main() {
	var opts options
	var cmdExitCode int

	cmd := cobra.Command{
		Use:   "graphpipe-batcher",
		Short: "graphpipe-batcher - batching up ml requests",
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
			if len(opts.targetURL) == 0 {
				cmdExitCode = 1
				logrus.Infof("--target-url must be specified")
				cmd.Usage()
				return
			}
			if len(opts.inputs) == 0 {
				cmdExitCode = 1
				logrus.Infof("--inputs must be specified")
				cmd.Usage()
				return
			}
			if len(opts.outputs) == 0 {
				cmdExitCode = 1
				logrus.Infof("--outputs must be specified")
				cmd.Usage()
				return
			}

			logrus.Infof("Starting graphpipe-batcher %s", version())

			if err := serve(opts); err != nil {
				logrus.Errorf("Failed to serve: %v", err)
				cmdExitCode = 1
			}
		},
	}
	f := cmd.Flags()
	f.StringVarP(&opts.cacheDir, "cache-dir", "d", "~/.graphpipe", "directory for local cache state")
	f.StringVarP(&opts.listen, "listen", "l", "127.0.0.1:10000", "listen string")
	f.BoolVarP(&opts.cache, "cache", "c", false, "enable results caching")
	f.StringVarP(&opts.targetURL, "target-url", "", "", "upstream graphpipe server")
	f.IntVarP(&opts.batchSize, "batch-size", "", 10, "batch size")
	f.StringVarP(&opts.inputs, "inputs", "i", "", "comma seprated default inputs")
	f.StringVarP(&opts.outputs, "outputs", "o", "", "comma separated default outputs")
	f.IntVarP(&opts.timeout, "timeout", "", 250, "timeout, in ms")
	f.IntVarP(&opts.workers, "workers", "", 1, "number of workers")
	f = cmd.PersistentFlags()
	f.BoolVarP(&opts.verbose, "verbose", "v", false, "verbose output")
	f.BoolVarP(&opts.version, "version", "V", false, "show version")

	opts.cacheDir = strings.Replace(opts.cacheDir, "~", os.Getenv("HOME"), -1)

	cmd.Execute()
	os.Exit(cmdExitCode)
}

type ioData struct {
	Inputs        map[string]*graphpipe.NativeTensor
	OutputNames   []string
	OutputTensors []*graphpipe.NativeTensor
	Error         error
	ReturnChannel chan *ioData
}

type bContext struct {
	q    chan (chan *ioData)
	meta *graphpipe.NativeMetadataResponse
}

func concatCopyPreAllocate(slices [][]byte) []byte {
	var totalLen int
	for _, s := range slices {
		totalLen += len(s)
	}
	tmp := make([]byte, totalLen)
	var i int
	for _, s := range slices {
		i += copy(tmp[i:], s)
	}
	return tmp
}

func checkDataShapes(firstIo, io *ioData) error {
	rows := -1
	for name, tensor := range io.Inputs {
		firstTensor, ok := firstIo.Inputs[name]
		if !ok {
			return errors.New("Could not find input: " + name)
		}
		if len(firstTensor.Shape) != len(tensor.Shape) {
			return errors.New("Shape length mismatch in batch")
		}
		for j := 1; j < len(firstTensor.Shape); j++ {
			if firstTensor.Shape[j] != tensor.Shape[j] {
				return errors.New("Shape mismatch in batch")
			}

		}
		if len(tensor.Shape) <= 0 || tensor.Shape[0] <= 0 {
			return errors.New("Invalid shape in batch ")
		}
		if rows < 0 {
			rows = int(tensor.Shape[0])
		} else if int(tensor.Shape[0]) != rows {
			return errors.New("All inputs must have same row count")
		}
	}
	return nil
}

func serve(opts options) error {
	ctx := &bContext{}

	meta := &graphpipe.NativeMetadataResponse{}
	meta.Name = opts.targetURL
	meta.Description = "Graphpipe request batching process"
	meta.Server = "graphpipe-batcher"
	meta.Version = version()
	ctx.meta = meta
	ctx.q = make(chan (chan *ioData), 1000)
	cachePath := ""
	if opts.cache {
		cachePath = filepath.Join(opts.cacheDir, "batcher-cache.db")
	}

	dIn := strings.Split(opts.inputs, ",")
	dOut := strings.Split(opts.outputs, ",")

	for _, name := range dIn {
		io := graphpipe.NativeIOMetadata{}
		io.Name = name
		io.Description = "Input to batch"
		meta.Inputs = append(meta.Inputs, io)
	}

	for _, name := range dOut {
		io := graphpipe.NativeIOMetadata{}
		io.Name = name
		io.Description = "Output Batch"
		meta.Outputs = append(meta.Outputs, io)
	}

	serveOpts := &graphpipe.ServeRawOptions{
		Listen:         opts.listen,
		CacheFile:      cachePath,
		Meta:           ctx.meta,
		DefaultInputs:  dIn,
		DefaultOutputs: dOut,
		Apply:          ctx.apply,
		GetHandler:     ctx.getHandler,
	}

	for i := 0; i < opts.workers; i++ {
		go func() {
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

			data := []*ioData{}

			for {
				timedOut := false
				select {
				case c := <-ctx.q:
					data = append(data, <-c)
					break
				case <-time.After(time.Duration(opts.timeout) * time.Millisecond):
					timedOut = true
					break
				}
				batchErrors := []int{}
				if len(data) == 0 {
					continue
				}

				firstIo := data[0]
				batchErrors = []int{}
				for i, io := range data {
					err := checkDataShapes(firstIo, io)
					if err != nil {
						io.Error = err
						io.ReturnChannel <- io
						batchErrors = append(batchErrors, i)
					}
				}
				if len(batchErrors) >= 0 {
					fixedData := []*ioData{}
					for i := range data {
						for _, idx := range batchErrors {
							if i == idx {
								break
							}
						}
						fixedData = append(fixedData, data[i])
					}
					data = fixedData
				}
				totalRows := int64(0)
				for _, io := range data {
					for _, tensor := range io.Inputs {
						totalRows += tensor.Shape[0]
						break
					}
				}
				if totalRows >= int64(opts.batchSize) || timedOut {
					inputs := []*graphpipe.NativeTensor{}
					inputNames := []string{}
					outputNames := []string{}
					rowCounts := []int64{}

					for _, io := range data {
						for name := range io.Inputs {
							inputNames = append(inputNames, name)
						}
						for j := range io.OutputNames {
							outputNames = append(outputNames, io.OutputNames[j])
						}
						break
					}
					allData := make([][][]byte, len(inputNames))
					for _, io := range data {
						rc := int64(-1)
						for i, name := range inputNames {
							tensor := io.Inputs[name]
							allData[i] = append(allData[i], tensor.Data)
							rc = tensor.Shape[0]
						}
						rowCounts = append(rowCounts, rc)
					}
					for i, d := range allData {
						nt := graphpipe.NativeTensor{}
						name := inputNames[i]
						shape := []int64{}
						for _, s := range data[0].Inputs[name].Shape {
							shape = append(shape, s)
						}
						shape[0] = int64(totalRows)
						inputNames = append(inputNames, name)
						nt.InitWithData(concatCopyPreAllocate(d), shape, data[0].Inputs[name].Type)
						inputs = append(inputs, &nt)
					}
					//ship it!
					tensors, err := graphpipe.MultiRemoteRaw(client, opts.targetURL, "", inputs, inputNames, outputNames)
					if err != nil {
						for _, io := range data {
							io.Error = err
							io.ReturnChannel <- io
						}

					} else {
						offset := int64(0)
						for i, io := range data {
							outputs := []*graphpipe.NativeTensor{}
							curRows := rowCounts[i]
							for _, t := range tensors {
								shape := []int64{}
								rowSize := int64(1)
								for j := 1; j < len(t.Shape); j++ {
									rowSize *= t.Shape[j]
								}
								itemCount := int64(1)
								for j := 0; j < len(t.Shape); j++ {
									itemCount *= t.Shape[j]
									shape = append(shape, t.Shape[j])
								}
								shape[0] = curRows
								itemSize := len(t.Data) / int(itemCount)
								start := int(offset) * int(itemSize) * int(rowSize)
								end := start + int(rowSize)*int(curRows)*int(itemSize)
								nt := graphpipe.NativeTensor{}
								nt.InitWithData(t.Data[start:end], shape, t.Type)
								outputs = append(outputs, &nt)
							}
							io.OutputTensors = outputs

							io.ReturnChannel <- io
							offset += curRows
						}
					}
					data = []*ioData{}
				}
			}
		}()
	}

	return graphpipe.ServeRaw(serveOpts)
}

func (ctx *bContext) apply(requestContext *graphpipe.RequestContext, config string, inputs map[string]*graphpipe.NativeTensor, outputNames []string) ([]*graphpipe.NativeTensor, error) {
	io := ioData{}
	io.Inputs = inputs
	io.OutputNames = outputNames

	io.ReturnChannel = make(chan *ioData, 1)
	c := make(chan *ioData, 1)
	c <- &io

	ctx.q <- c

	response := <-io.ReturnChannel
	return response.OutputTensors, response.Error
}

func (ctx *bContext) getHandler(w http.ResponseWriter, r *http.Request, body []byte) error {
	js, err := json.MarshalIndent(ctx.meta, "", "    ")
	if err == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(js)
	}
	return err
}
