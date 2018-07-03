package graphpipe

import (
	"testing"

	fb "github.com/google/flatbuffers/go"
	graphpipefb "github.com/oracle/graphpipe-go/graphpipefb"
)

func TestNativeTensorFlatInt64(t *testing.T) {
	nt := &NativeTensor{}
	v := []int64{2, 2}
	err := nt.InitSimple(v)
	if err != nil {
		t.Fatal(err)
	}
	if len(nt.Shape) != 1 {
		t.Fatal("Wrong shape")
	}
	if len(nt.Data) != 8*2 {
		t.Fatal("Wrong data length")
	}
	if nt.Type != graphpipefb.TypeInt64 {
		t.Fatal("Wrong data type")
	}

	b := fb.NewBuilder(1024)
	nt.Build(b)
}

func TestNativeTensorString(t *testing.T) {
	nt := &NativeTensor{}
	v := []string{"foo", "bar", "baz"}
	err := nt.InitSimple(v)
	if err != nil {
		t.Fatal(err)
	}
	if len(nt.Shape) != 1 {
		t.Fatal("Wrong shape")
	}
	if nt.Shape[0] != 3 {
		t.Fatal("Wrong item count")
	}
	if nt.Type != graphpipefb.TypeString {
		t.Fatal("Wrong data type")
	}

	b := fb.NewBuilder(1024)
	nt.Build(b)
}
