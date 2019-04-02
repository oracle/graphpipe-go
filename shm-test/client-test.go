package main 

import (
	"fmt"
	graphpipe "github.com/oracle/graphpipe-go"
	"time"
	"github.com/kshedden/gonpy"
	"flag"
	"encoding/binary"
	"math"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
	"net/http"
)


func getImages(filename string, count int) [][][][]float32 {
	req := make([][][][]float32, count)
	for i := range req {
		req[i] = make([][][]float32, 224)
		for j := range req[i] {
			req[i][j] = make([][]float32, 224)
			for k := range req[i][j] {
				req[i][j][k] = make([]float32, 3)
			}
		}
	}
	
	r, _ := gonpy.NewFileReader(filename)
	data, err := r.GetFloat32()
	if err != nil {
		panic(err)
	}

	i := 0
	for x := 0; x < 224; x++ {
		for y := 0; y < 224; y++ {
			for c := 0; c < 3; c++ {
				for l := 0; l < count; l++ {
					req[l][x][y][c] = data[i]
				}
				i++
			}
		}
	}
	return req
}

// Gets `batchSize` copies of the data in `filename` (which should be a float32
// npy file with a preprocessed image for vgg16) in one []byte.
func getImagesFlat(filename string, batchSize int) []byte {
	r, _ := gonpy.NewFileReader(filename)
	data, err := r.GetFloat32()
	if err != nil {
		panic(err)
	}

	bytes := make([]byte, 4 * len(data) * batchSize)
	// First populate the first 4 * len(data) bytes (first image in batch).
	for pos, f := range data  {
		i := math.Float32bits(f)
		binary.LittleEndian.PutUint32(bytes[4*pos:], i)
	}
	
	for i := 1; i < batchSize; i++ {
		copy(bytes[4*len(data)*i:], bytes[:4*len(data)])
	}
	
	return bytes
}

func main() {
	var socket = flag.String("socket", "", "socket to use")
	var imgFile = flag.String("imageFile", "dog.npy", 
		"image file (npy, preprocessed for vgg16)")
	var numRequests = flag.Int("numRequests", 100, "num requests to send")
	var batchSize = flag.Int("batchSize", 256, "batch size")
	var uri = flag.String("uri", "http://127.0.0.1:9000", "server address")
	var mode = flag.String("mode", "http", 
		"which mode (http, shm, http-raw, shm-raw)")
	flag.Parse()

	fmt.Printf("Mode is %s\n", *mode)
	var fn func()
	if *mode == "http" {
		request := getImages(*imgFile, *batchSize)
		fn = func() {
			graphpipe.Remote(*uri, request)
		}
	} else if *mode == "http-raw" {
		nt := getNativeTensors(*imgFile, *batchSize)
		client := graphpipe.HttpClient{
			NetHttpClient: http.DefaultClient,
			Uri: *uri,
		}	
		fn = func() {
			graphpipe.MultiRemoteRaw(client, "", nt, nil, nil)
		}
	} else if *mode == "shm-raw" {
		client := makeShmClient(*socket, *batchSize)
		nt := getNativeTensors(*imgFile, *batchSize)
		fn = func() {
			graphpipe.MultiRemoteRaw(client, "", nt, nil, nil)
		}
	} else {
		panic("Unknown mode " + *mode)
	}
	
	timeRequests(*mode, *numRequests, fn)
}

func getNativeTensors(imgFile string, batchSize int) []*graphpipe .NativeTensor {
	data := getImagesFlat(imgFile, batchSize)

	nt := graphpipe.NativeTensor{}
	shape := []int64{int64(batchSize), 224, 224, 3}
	nt.InitWithData(data, shape, graphpipefb.TypeFloat32)
	inputs := []*graphpipe.NativeTensor{}
	inputs = append(inputs, &nt)

	return inputs
}

func timeRequests(mode string, numRequests int, reqFunc func()) {
	fmt.Printf("Making %d %s requests\n", numRequests, mode)
	start := time.Now()
	for i := 0; i < numRequests; i++ {
		reqFunc()
	}
	duration := time.Since(start)
	fmt.Printf("Took %v (%v ms average)\n", duration,
		int(duration)/numRequests/1.0E6)
}

func makeShmClient(socket string, batchSize int) graphpipe.Client {
	var shmSize = 224 * 224 * 3 * 4 * batchSize + 1000
	return graphpipe.CreateShmClient(socket, shmSize)
}