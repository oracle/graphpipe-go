/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/

package graphpipe

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/Sirupsen/logrus"
	bolt "github.com/coreos/bbolt"
	fb "github.com/google/flatbuffers/go"
	flatbuffers "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

// Error is our wrapper around the error interface.
type Error interface {
	error
	Status() int
}

// StatusError is our wrapper around an http error interface.
type StatusError struct {
	Code int
	Err  error
}

// Error returns the error message.
func (se StatusError) Error() string {
	return se.Err.Error()
}

// Status returns an http status code.
func (se StatusError) Status() int {
	return se.Code
}

var clientCount = int64(0)

func setupLifecycleRoutes(c *appContext) {
	http.Handle("/control/is_ready", appHandler{c, isReadyHandler})
	http.Handle("/control/is_alive", appHandler{c, isAliveHandler})
	http.Handle("/control/shutdown", appHandler{c, shutdownHandler})
	http.Handle("/control/client_count", appHandler{c, clientCountHandler})
}

// ListenAndServe is like robocop but for servers (listens on a
// host:port and handles requests).
func ListenAndServe(addr string, handler http.Handler) error {
	server := &http.Server{Addr: addr, Handler: handler}
	if addr == "" {
		addr = ":http"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return server.Serve(&counterListener{ln.(*net.TCPListener)})
}

type counterListener struct {
	*net.TCPListener
}

// Accept is implementing the TCPListener interface, here to count
// connections for performance testing purposes.
func (l *counterListener) Accept() (net.Conn, error) {
	tc, err := l.AcceptTCP()
	if err != nil {
		return nil, err
	}
	tc.SetKeepAlive(true)
	tc.SetKeepAlivePeriod(3 * time.Minute)
	atomic.AddInt64(&clientCount, 1)
	return &counterListenerConn{Conn: tc}, nil
}

type counterListenerConn struct {
	net.Conn
}

// Close closes our connection and decrements our counter.
func (l *counterListenerConn) Close() error {
	err := l.Conn.Close()
	atomic.AddInt64(&clientCount, -1)
	return err
}

// Serve offers multiple inputs and outputs and converts tensors
// into native datatypes based on the shapes passed in to this function
// plus any additional shapes implied by your apply function.
// If cache is true, will attempt to cache using cache.db as cacheFile
func Serve(listen string, cache bool, apply interface{}, inShapes, outShapes [][]int64) error {
	opts := BuildSimpleApply(apply, inShapes, outShapes)
	opts.Listen = listen
	if cache {
		opts.CacheFile = "cache.db"
	}
	return ServeRaw(opts)
}

// GetHandlerFunc is an indirection to return the handler.
type GetHandlerFunc func(http.ResponseWriter, *http.Request, []byte) error

// ServeRawOptions is just a call parameter struct.
type ServeRawOptions struct {
	Listen         string
	CacheFile      string
	Meta           *NativeMetadataResponse
	DefaultInputs  []string
	DefaultOutputs []string
	Apply          Applier
	GetHandler     GetHandlerFunc
}

func parseURL(listen string) (string, string, error) {
	listenURL := listen
	if _, err := strconv.Atoi(listenURL[:1]); err == nil {
		listenURL = "http://" + listenURL
		logrus.Infof("Converted listen param from %s to %s", listen, listenURL)
	}

	u, err := url.Parse(listenURL)
	if err != nil {
		return "", "", err
	}
	return u.Scheme, u.Host, nil
}

// ServeRaw starts the model server. The listen url can be specified
// with the listen parameter. If cacheFile is not "" then caches will be stored
// using it. context will be passed back to the handler
func ServeRaw(opts *ServeRawOptions) error {
	var err error
	c := &appContext{
		meta:           opts.Meta,
		apply:          opts.Apply,
		getHandler:     opts.GetHandler,
		defaultInputs:  opts.DefaultInputs,
		defaultOutputs: opts.DefaultOutputs,
		isReady:        1,
		isAlive:        1,
	}

	if opts.CacheFile != "" {
		c.db, err = bolt.Open(opts.CacheFile, 0600, &bolt.Options{Timeout: 1 * time.Second})
		if err != nil {
			logrus.Errorf("Could not open db at '%s': %v", opts.CacheFile, err)
			return err
		}
		defer c.db.Close()
	}

	scheme, ipPort, err := parseURL(opts.Listen)
	if err != nil {
		return err
	}

	host, _, err := net.SplitHostPort(ipPort)
	if err != nil {
		logrus.Errorf("Could not find port in %s", ipPort)
		return err
	}

	if net.ParseIP(host) == nil {
		logrus.Errorf("Can only bind ips for serving.  Found ip %s", host)
	}

	logrus.Infof("Attempting to serve with protocol %s on %s", scheme, ipPort)
	switch scheme {
	case "http":
		setupLifecycleRoutes(c)
		http.Handle("/", appHandler{c, Handler})
		logrus.Infof("Listening with %s on '%s'", scheme, ipPort)
		err = ListenAndServe(ipPort, nil)
		if err != nil {
			logrus.Errorf("Error trying to ListenAndServe: %v", err)
			return err
		}
	case "grpc+http":
		listen, err := net.Listen("tcp", ipPort)
		if err != nil {
			logrus.Errorf("Error trying to Listen for GRPC: %v", err)
		}

		logrus.Infof("Listening with %s on '%s'", scheme, ipPort)
		ser := grpc.NewServer(grpc.CustomCodec(flatbuffers.FlatbuffersCodec{}))
		graphpipefb.RegisterGraphpipeServiceServer(ser, c)
		if err := ser.Serve(listen); err != nil {
			logrus.Fatalf("Failed to serve: %v", err)
			return err
		}
	default:
		logrus.Errorf("Couldn't serve protocol %s", scheme)
	}

	return nil
}

type appContext struct {
	meta           *NativeMetadataResponse
	apply          Applier
	getHandler     GetHandlerFunc
	defaultInputs  []string
	defaultOutputs []string
	cacheFile      string
	db             *bolt.DB
	isReady        int64
	isAlive        int64
}

type appHandler struct {
	*appContext
	H func(*appContext, http.ResponseWriter, *http.Request) error
}

// RequestContext attaches our flatbuffers to the request.
type RequestContext struct {
	hasDied     int32
	CleanupFunc func()
	builder     *fb.Builder
}

// IsAlive tells you if it isn't dead.
func (ctx *RequestContext) IsAlive() bool {
	return ctx.hasDied == 0
}

// SetDead makes sure it isn't alive.
func (ctx *RequestContext) SetDead() {
	atomic.StoreInt32(&ctx.hasDied, 1)
}

// BuildTensor does the heavy lifting to make sure we have a flatbuffer.
func (ctx *RequestContext) BuildTensor(val interface{}) (fb.UOffsetT, error) {
	return BuildTensorSafe(ctx.builder, val)
}

// ServeHTTP is the handler interface for responding to requests.
func (ah appHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	err := ah.H(ah.appContext, w, r)
	if err != nil {
		switch e := err.(type) {
		case Error:
			// We can retrieve the status here and write out a specific
			// HTTP status code.
			logrus.Errorf("HTTP %d - %s", e.Status(), e)
			http.Error(w, e.Error(), e.Status())
		default:
			// Any error types we don't specifically look out for default
			// to serving a HTTP 500
			http.Error(w, http.StatusText(http.StatusInternalServerError),
				http.StatusInternalServerError)
		}
	}
	duration := time.Now().Sub(startTime)
	logrus.Infof("Request for %s took %s", r.URL.Path, duration)
}

func safeGetResults(c *appContext, requestContext *RequestContext, inferRequest *graphpipefb.InferRequest) (outputs []*NativeTensor, err error) {
	defer func() {
		if r := recover(); r != nil {
			msg := fmt.Sprintf("panic recovered: %s", r)
			logrus.Errorf(msg)
			err = errors.New(msg)
		}
	}()

	if c.db == nil {
		outputs, err = getResults(c, requestContext, inferRequest)
	} else {
		outputs, err = getResultsCached(c, requestContext, inferRequest)
	}
	return outputs, err
}

func (c *appContext) _Infer(inferRequest *graphpipefb.InferRequest, requestContext *RequestContext) (*flatbuffers.Builder, error) {
	var outputs []*NativeTensor
	var err error
	outputs, err = safeGetResults(c, requestContext, inferRequest)

	if requestContext.CleanupFunc != nil {
		defer requestContext.CleanupFunc()
	}

	b := requestContext.builder

	var errOffset flatbuffers.UOffsetT
	if err != nil {
		errStr := b.CreateString(err.Error())
		graphpipefb.ErrorStart(b)
		graphpipefb.ErrorAddCode(b, 1)
		graphpipefb.ErrorAddMessage(b, errStr)
		offset := graphpipefb.ErrorEnd(b)
		graphpipefb.InferResponseStartErrorsVector(b, 1)
		b.PrependUOffsetT(offset)
		errOffset = b.EndVector(1)
	}

	outputOffsets := make([]fb.UOffsetT, len(outputs))
	for i := 0; i < len(outputs); i++ {
		outputOffsets[i] = outputs[i].Build(b)
	}

	graphpipefb.InferResponseStartOutputTensorsVector(b, len(outputOffsets))
	for i := len(outputOffsets) - 1; i >= 0; i-- {
		offset := outputOffsets[i]
		b.PrependUOffsetT(offset)
	}

	tensors := b.EndVector(len(outputOffsets))
	graphpipefb.InferResponseStart(b)
	graphpipefb.InferResponseAddOutputTensors(b, tensors)

	if errOffset != 0 {
		graphpipefb.InferResponseAddErrors(b, errOffset)
	}

	offset := graphpipefb.InferResponseEnd(b)
	b.Finish(offset)
	return b, nil
}

func (c *appContext) Infer(context context.Context, inferRequest *graphpipefb.InferRequest) (*flatbuffers.Builder, error) {
	requestContext := &RequestContext{
		builder: fb.NewBuilder(1024),
	}

	b, err := c._Infer(inferRequest, requestContext)
	return b, err
}

func (c *appContext) Metadata(context context.Context, in *graphpipefb.MetadataRequest) (*flatbuffers.Builder, error) {
	b := fb.NewBuilder(1024)
	offset := c.meta.Build(b)
	b.Finish(offset)
	return b, nil
}

// Handler handles our http requests.
func Handler(c *appContext, w http.ResponseWriter, r *http.Request) error {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return err
	}

	if r.Method == "GET" {
		if c.getHandler != nil {
			return c.getHandler(w, r, body)
		}
		http.Error(w, "Unhandled GET", http.StatusInternalServerError)
		return nil
	}

	request := graphpipefb.GetRootAsRequest(body, 0)
	if request.ReqType() == graphpipefb.ReqInferRequest {
		requestContext := &RequestContext{
			builder: fb.NewBuilder(1024),
		}
		notify := w.(http.CloseNotifier).CloseNotify()
		done := make(chan bool)

		defer func() {
			done <- true
		}()

		go func() {
			select {
			case <-notify:
			case <-done:
			}
			requestContext.SetDead()
		}()
		inferRequest := &graphpipefb.InferRequest{}
		table := inferRequest.Table()
		request.Req(&table)
		inferRequest.Init(table.Bytes, table.Pos)
		b, err := c._Infer(inferRequest, requestContext)
		if err != nil {
			return err
		}

		tmp := b.FinishedBytes()
		io.Copy(w, bytes.NewReader(tmp))

		return nil
	}

	b := fb.NewBuilder(1024)
	offset := c.meta.Build(b)
	tmp := Serialize(b, offset)
	io.Copy(w, bytes.NewReader(tmp))
	return nil
}

func isReadyHandler(c *appContext, w http.ResponseWriter, r *http.Request) error {
	if c.isReady == 1 {
		fmt.Fprintf(w, "ok\n")
		return nil
	}
	return StatusError{503, errors.New("not ready")}
}

func isAliveHandler(c *appContext, w http.ResponseWriter, r *http.Request) error {
	if c.isAlive == 1 {
		fmt.Fprintf(w, "ok\n")
		return nil
	}
	return StatusError{503, errors.New("not alive")}
}

func shutdownHandler(c *appContext, w http.ResponseWriter, r *http.Request) error {
	atomic.AddInt64(&c.isReady, -1)
	for {
		if clientCount == 1 { // Allow for a connection count of 1, to include current client
			break
		}
		time.Sleep(time.Second / 10)
	}
	time.Sleep(time.Second * 5) // sleep also, to give enough time to leave pool if behind proxy
	atomic.AddInt64(&c.isAlive, -1)
	fmt.Fprintf(w, "shutdown\n")
	return nil
}

func clientCountHandler(c *appContext, w http.ResponseWriter, r *http.Request) error {
	fmt.Fprintf(w, "%d\n", clientCount)
	return nil
}
