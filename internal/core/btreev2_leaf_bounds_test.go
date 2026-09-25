package core

import (
	"bytes"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// allocatedDuring returns the bytes allocated while f runs.
func allocatedDuring(f func()) uint64 {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// TestReadBTreeV2LeafRecords_HugeRecordCount checks that a crafted B-tree
// header (65535 records of 65535 bytes) is rejected against the file size
// before the ~4 GiB leaf buffer is allocated.
func TestReadBTreeV2LeafRecords_HugeRecordCount(t *testing.T) {
	file := bytes.NewReader(append([]byte("BTLF\x00\x08"), make([]byte, 64)...))
	var err error
	alloc := allocatedDuring(func() {
		_, err = readBTreeV2LeafRecords(file, 0, 0xFFFF, 8, 0xFFFF, nil)
	})
	require.Error(t, err)
	require.Less(t, alloc, uint64(1<<20), "allocated %d bytes before validating the leaf size", alloc)
}
