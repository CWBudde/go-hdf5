package structures

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// TestLoadLocalHeap_HugeDataSegment reproduces the fuzz crasher where a
// corrupted data segment size (~131 TB) caused a fatal out-of-memory error.
func TestLoadLocalHeap_HugeDataSegment(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], "HEAP")
	binary.LittleEndian.PutUint64(buf[8:16], 0x780000000000) // data segment size
	binary.LittleEndian.PutUint64(buf[16:24], 1)
	binary.LittleEndian.PutUint64(buf[24:32], 32)

	sb := &core.Superblock{OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}
	_, err := LoadLocalHeap(bytes.NewReader(buf), 0, sb)
	require.Error(t, err)
	require.Contains(t, err.Error(), "exceed file size")
}

func TestParseIndirectBlock_HugeEntryCount(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], "FHIB")
	_, err := ParseIndirectBlock(bytes.NewReader(buf), 8, 0xFFFF, 0xFFFF, 8, 4, binary.LittleEndian, 0)
	require.Error(t, err)
}

func TestReadDirectBlock_HugeBlockSize(t *testing.T) {
	fh := &FractalHeap{
		Header:     &FractalHeapHeader{HeapOffsetSize: 4},
		reader:     bytes.NewReader(make([]byte, 64)),
		sizeofAddr: 8,
		endianness: binary.LittleEndian,
	}
	_, err := fh.readDirectBlock(8, 1<<45)
	require.Error(t, err)
	_, err = fh.readDirectBlock(8, 2) // smaller than the block header
	require.Error(t, err)
}
