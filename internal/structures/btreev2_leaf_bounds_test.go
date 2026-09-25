package structures

import (
	"bytes"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestReadBTreeV2LeafNode_HugeRecordCount checks that a crafted record
// count is rejected against the file size before the leaf buffer is
// allocated.
func TestReadBTreeV2LeafNode_HugeRecordCount(t *testing.T) {
	file := bytes.NewReader(append([]byte("BTLF\x00\x05"), make([]byte, 64)...))
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	_, _, err := readBTreeV2LeafNode(file, 0, 1<<28, BTreeV2TypeLinkNameIndex, nil)
	runtime.ReadMemStats(&after)
	require.Error(t, err)
	require.Less(t, after.TotalAlloc-before.TotalAlloc, uint64(1<<20))
}
