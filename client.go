package graphpipe

import (
	"net"
	"net/http"
	fb "github.com/google/flatbuffers/go"
	"bytes"
	"github.com/Sirupsen/logrus"
	"io/ioutil"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"github.com/gen2brain/shm"
)

type Client interface {
	call(*fb.Builder, []byte) ([]byte, error)
	builder() *fb.Builder
}

type ShmClient struct {
	Conn  *net.Conn
	Shm   []byte
}

type HttpClient struct {
	NetHttpClient *http.Client
	Uri           string
}


func (sc ShmClient) builder() *fb.Builder {
	b := fb.NewBuilder(0)
	b.Bytes = sc.Shm
	b.Reset()
	return b
}

func (sc ShmClient) call(builder *fb.Builder, request []byte) ([]byte, error) {
	startPos := builder.Head()
	length := len(request)
	WriteInt(sc.Conn, uint32(startPos))
	WriteInt(sc.Conn, uint32(length))
	respStartPos, err := ReadInt(sc.Conn)
	respSize, err := ReadInt(sc.Conn)
	if err != nil {
		return nil, err
	}
	body := sc.Shm[respStartPos:respStartPos + respSize]
	return body, nil
}


func (hc HttpClient) builder() *fb.Builder {
	return fb.NewBuilder(1024)
}

func (hc HttpClient) call(builder *fb.Builder, request []byte) ([]byte, error) {
	rq, err := http.NewRequest("POST", hc.Uri, bytes.NewReader(request))
	if err != nil {
		logrus.Errorf("Failed to create request: %v", err)
		return nil, err
	}

	// send the request
	rs, err := hc.NetHttpClient.Do(rq)
	if err != nil {
		logrus.Errorf("Failed to send request: %v", err)
		return nil, err
	}
	defer rs.Body.Close()

	body, err := ioutil.ReadAll(rs.Body)
	if err != nil {
		logrus.Errorf("Failed to read body: %v", err)
		return nil, err
	}
	if rs.StatusCode != 200 {
		return nil, fmt.Errorf("Remote failed with %d: %s", rs.StatusCode, string(body))
	}

	return body, nil
}



// Opens the socket, creates the shared memory, communicates the shm id over
// the socket, and installs a signal handler to close the socket and remove
// the shm.
func CreateShmClient(socket string, shmSize int) Client {
	conn, err := net.Dial("unix", socket)
	if err != nil {
		logrus.Fatal("Dial error", err)
	}

	shmId, err := shm.Get(shm.IPC_PRIVATE, shmSize, shm.IPC_CREAT|0777)
	if err != nil || shmId < 0 {
		panic(fmt.Sprintf("Could not shmget %d bytes", shmSize))
	}

	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, os.Interrupt, syscall.SIGTERM, syscall.SIGINT)
	go func(conn *net.Conn, c chan os.Signal) {
		sig := <-c
		log.Printf("Caught signal %s: shutting down", sig)
		shm.Rm(shmId)
		(*conn).Close()
		os.Exit(-1)
	}(&conn, sigc)

	shmBytes, err := shm.At(shmId, 0, 0)
	// Communicate our shm id to the server.
	WriteInt(&conn, uint32(shmId))
	if err != nil {
		panic(err)
	}

	return ShmClient{
		Conn: &conn,
		Shm: shmBytes,
	}
}