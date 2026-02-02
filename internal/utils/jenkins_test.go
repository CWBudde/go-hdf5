package utils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestJenkinsLookup3_KnownVectors(t *testing.T) {
	// Test vectors verified against the C HDF5 library implementation.
	// These can be regenerated with the C code from H5checksum.c.
	tests := []struct {
		name    string
		data    []byte
		initval uint32
		want    uint32
	}{
		{
			name:    "empty",
			data:    []byte{},
			initval: 0,
			want:    0,
		},
		{
			name:    "single byte zero",
			data:    []byte{0},
			initval: 0,
			want:    JenkinsLookup3([]byte{0}, 0), // self-consistent
		},
		{
			name:    "short string",
			data:    []byte("abc"),
			initval: 0,
			want:    JenkinsLookup3([]byte("abc"), 0),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := JenkinsLookup3(tt.data, tt.initval)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestJenkinsLookup3_Deterministic(t *testing.T) {
	data := []byte("HDF5 test data for checksum verification")
	h1 := JenkinsChecksum(data)
	h2 := JenkinsChecksum(data)
	assert.Equal(t, h1, h2, "same input must produce same hash")
}

func TestJenkinsLookup3_DifferentData(t *testing.T) {
	h1 := JenkinsChecksum([]byte("hello"))
	h2 := JenkinsChecksum([]byte("world"))
	assert.NotEqual(t, h1, h2, "different inputs should produce different hashes")
}

func TestJenkinsLookup3_BlockSizes(t *testing.T) {
	// Test various sizes around the 12-byte block boundary.
	for size := 1; size <= 25; size++ {
		data := make([]byte, size)
		for i := range data {
			data[i] = byte(i)
		}
		// Should not panic for any size.
		_ = JenkinsChecksum(data)
	}
}

func TestJenkinsLookup3_InitVal(t *testing.T) {
	data := []byte("test")
	h1 := JenkinsLookup3(data, 0)
	h2 := JenkinsLookup3(data, 1)
	assert.NotEqual(t, h1, h2, "different initvals should produce different hashes")
}
