package core

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func testSB() *Superblock {
	return &Superblock{OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}
}

// TestReadRawData_HugeContiguous reproduces the fuzz crasher where a
// corrupted dataspace (~4.5 TB of contiguous data) caused a fatal OOM.
func TestReadRawData_HugeContiguous(t *testing.T) {
	r := bytes.NewReader(make([]byte, 4096))
	layout := &DataLayoutMessage{Class: LayoutContiguous, DataAddress: 0}
	ds := &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{573470932992}}
	dt := &DatatypeMessage{Class: DatatypeFloat, Size: 8}

	_, err := readRawData(r, layout, ds, dt, testSB(), nil)
	require.Error(t, err)

	// Unallocated storage is zero-filled, but still bounded.
	layout.DataAddress = undefinedAddress
	_, err = readRawData(r, layout, ds, dt, testSB(), nil)
	require.Error(t, err)

	// Overflowing dimension product.
	ds.Dimensions = []uint64{math.MaxUint32, math.MaxUint32, math.MaxUint32}
	_, err = readRawData(r, layout, ds, dt, testSB(), nil)
	require.Error(t, err)

	// Zero-sized datatype.
	dt.Size = 0
	ds.Dimensions = []uint64{4}
	_, err = readRawData(r, layout, ds, dt, testSB(), nil)
	require.Error(t, err)
}

func TestReadRawData_CompactTruncated(t *testing.T) {
	layout := &DataLayoutMessage{Class: LayoutCompact, CompactData: make([]byte, 8)}
	ds := &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{1 << 40}}
	dt := &DatatypeMessage{Class: DatatypeFloat, Size: 8}
	_, err := readRawData(bytes.NewReader(nil), layout, ds, dt, testSB(), nil)
	require.Error(t, err)
}

func TestDataspaceTotalElements_Saturates(t *testing.T) {
	ds := &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{1 << 40, 1 << 40}}
	require.Equal(t, uint64(math.MaxUint64), ds.TotalElements())
}

func TestApplyFiltersLimit_DeflateBomb(t *testing.T) {
	var buf bytes.Buffer
	w := zlib.NewWriter(&buf)
	_, err := w.Write(make([]byte, 1<<20)) // 1 MiB of zeros compresses to ~1 KiB
	require.NoError(t, err)
	require.NoError(t, w.Close())

	fp := &FilterPipelineMessage{Filters: []Filter{{ID: FilterDeflate}}}
	_, err = fp.ApplyFiltersLimit(buf.Bytes(), 4096)
	require.Error(t, err)

	out, err := fp.ApplyFiltersLimit(buf.Bytes(), 1<<20)
	require.NoError(t, err)
	require.Len(t, out, 1<<20)
}

func TestLZFDecompressLimit(t *testing.T) {
	// Literal run of 4 bytes, then long back-references repeating them.
	in := []byte{3, 'a', 'b', 'c', 'd'}
	for i := 0; i < 100; i++ {
		in = append(in, 0xE0, 0x03, 0xFF) // long backref, offset 4, len 264
	}
	_, err := lzfDecompressLimit(in, 1024)
	require.Error(t, err)
	out, err := lzfDecompressLimit(in, 1<<20)
	require.NoError(t, err)
	require.Len(t, out, 4+100*264)
}

func TestReadGlobalHeapCollection_HugeSize(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], "GCOL")
	buf[4] = 1
	binary.LittleEndian.PutUint64(buf[8:16], 1<<50)
	_, err := ReadGlobalHeapCollection(bytes.NewReader(buf), 0, 8)
	require.Error(t, err)
}

func TestReadGlobalHeapCollection_HugeObject(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], "GCOL")
	buf[4] = 1
	binary.LittleEndian.PutUint64(buf[8:16], 64)
	binary.LittleEndian.PutUint16(buf[16:18], 1)         // object ID
	binary.LittleEndian.PutUint64(buf[24:32], 1<<63+100) // object size (negative as int)
	_, err := ReadGlobalHeapCollection(bytes.NewReader(buf), 0, 8)
	require.Error(t, err)
}

// buildChunkBTreeNode builds a level-1 chunk B-tree node whose single child
// points back at itself.
func buildSelfReferencingBTree() []byte {
	const ndims = 1
	buf := make([]byte, 256)
	copy(buf[0:4], "TREE")
	buf[4] = 1                                 // chunk B-tree
	buf[5] = 1                                 // level 1 (internal)
	binary.LittleEndian.PutUint16(buf[6:8], 1) // one entry
	binary.LittleEndian.PutUint64(buf[8:16], math.MaxUint64)
	binary.LittleEndian.PutUint64(buf[16:24], math.MaxUint64)
	off := 24
	off += 4 + 4 + ndims*8                           // key 0
	binary.LittleEndian.PutUint64(buf[off:off+8], 0) // child 0 -> itself
	return buf
}

func TestCollectAllChunks_Cycle(t *testing.T) {
	r := bytes.NewReader(buildSelfReferencingBTree())
	node, err := ParseBTreeV1Node(r, 0, 8, 1, []uint64{8})
	require.NoError(t, err)
	_, err = node.CollectAllChunks(r, 8, []uint64{8})
	require.Error(t, err)
}

func TestParseBTreeV1Node_MaxEntries(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], "TREE")
	buf[4] = 1
	binary.LittleEndian.PutUint16(buf[6:8], 0xFFFF)
	_, err := ParseBTreeV1Node(bytes.NewReader(buf), 0, 8, 1, []uint64{8})
	require.Error(t, err)
}

func TestAttributeReadValue_TooManyElements(t *testing.T) {
	attr := &Attribute{
		Datatype:  &DatatypeMessage{Class: DatatypeString, Size: 0},
		Dataspace: &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{1 << 60}},
		Data:      []byte{0},
	}
	_, err := attr.ReadValue()
	require.Error(t, err)
}

func TestReadHeapObject_HugeLength(t *testing.T) {
	buf := make([]byte, 64)
	copy(buf[0:4], "FHDB")
	hdr := &fractalHeapHeaderRaw{HeapOffsetSize: 4}
	_, err := readHeapObject(bytes.NewReader(buf), 0, 0, 1<<40, testSB(), hdr)
	require.Error(t, err)
}
