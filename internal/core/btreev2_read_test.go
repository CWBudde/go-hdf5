package core

import (
	"bytes"
	"encoding/binary"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/cwbudde/go-hdf5/internal/utils"
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

// btreeV2HeaderBytes encodes a v2 B-tree header (8-byte addresses).
func btreeV2HeaderBytes(typ uint8, nodeSize uint32, recSize, depth uint16, root uint64, rootNRec uint16, total uint64) []byte {
	b := []byte("BTHD")
	b = append(b, 0, typ)
	b = binary.LittleEndian.AppendUint32(b, nodeSize)
	b = binary.LittleEndian.AppendUint16(b, recSize)
	b = binary.LittleEndian.AppendUint16(b, depth)
	b = append(b, 100, 40)
	b = binary.LittleEndian.AppendUint64(b, root)
	b = binary.LittleEndian.AppendUint16(b, rootNRec)
	b = binary.LittleEndian.AppendUint64(b, total)
	return binary.LittleEndian.AppendUint32(b, utils.JenkinsChecksum(b))
}

func withChecksum(b []byte) []byte {
	return binary.LittleEndian.AppendUint32(b, utils.JenkinsChecksum(b))
}

// TestReadBTreeV2Records_HugeRecordCount checks that a crafted B-tree header
// (65535 records of 65535 bytes) is rejected against the file size before
// the ~4 GiB leaf buffer is allocated.
func TestReadBTreeV2Records_HugeRecordCount(t *testing.T) {
	hdr := btreeV2HeaderBytes(8, 0xFFFFFFF0, 0xFFFF, 0, 64, 0xFFFF, 0xFFFF)
	file := bytes.NewReader(append(append(hdr, make([]byte, 64-len(hdr))...), append([]byte("BTLF\x00\x08"), make([]byte, 64)...)...))
	var err error
	alloc := allocatedDuring(func() {
		_, _, err = ReadBTreeV2Records(file, 0, testSB())
	})
	require.Error(t, err)
	require.Less(t, alloc, uint64(1<<20), "allocated %d bytes before validating the leaf size", alloc)
}

// depth1Tree builds a depth-1 tree of 4-byte records: an internal root with
// one record (value 100) and two leaves {1,2} and {200}. childAddr2 lets a
// test point the second child elsewhere (e.g. back at the root).
func depth1Tree(childAddr2 uint64) []byte {
	const nodeSize = 64
	file := make([]byte, 256)
	copy(file, btreeV2HeaderBytes(5, nodeSize, 4, 1, 64, 1, 4))

	// Internal root at 64: record, then 2 pointers of addr(8) + nrec(1 byte:
	// the leaf holds at most (64-10)/4 = 13 records).
	root := []byte("BTIN\x00\x05")
	root = binary.LittleEndian.AppendUint32(root, 100)
	root = binary.LittleEndian.AppendUint64(root, 128)
	root = append(root, 2)
	root = binary.LittleEndian.AppendUint64(root, childAddr2)
	root = append(root, 1)
	copy(file[64:], withChecksum(root))

	leaf1 := []byte("BTLF\x00\x05")
	leaf1 = binary.LittleEndian.AppendUint32(leaf1, 1)
	leaf1 = binary.LittleEndian.AppendUint32(leaf1, 2)
	copy(file[128:], withChecksum(leaf1))

	leaf2 := []byte("BTLF\x00\x05")
	leaf2 = binary.LittleEndian.AppendUint32(leaf2, 200)
	copy(file[192:], withChecksum(leaf2))
	return file
}

func TestReadBTreeV2Records_Depth1(t *testing.T) {
	info, recs, err := ReadBTreeV2Records(bytes.NewReader(depth1Tree(192)), 0, testSB())
	require.NoError(t, err)
	require.Equal(t, uint16(1), info.Depth)
	got := make([]uint32, 0, len(recs))
	for _, r := range recs {
		got = append(got, binary.LittleEndian.Uint32(r))
	}
	require.Equal(t, []uint32{1, 2, 100, 200}, got)
}

func TestReadBTreeV2Records_Cycle(t *testing.T) {
	// Second child points back at the internal root: must be rejected, not
	// looped over.
	_, _, err := ReadBTreeV2Records(bytes.NewReader(depth1Tree(64)), 0, testSB())
	require.Error(t, err)
}

func TestReadBTreeV2Records_BadChecksum(t *testing.T) {
	file := depth1Tree(192)
	file[128+6] ^= 0xFF
	_, _, err := ReadBTreeV2Records(bytes.NewReader(file), 0, testSB())
	require.ErrorContains(t, err, "checksum")
}

func TestLimitEncSize(t *testing.T) {
	require.Equal(t, 1, limitEncSize(0))
	require.Equal(t, 1, limitEncSize(255))
	require.Equal(t, 2, limitEncSize(256))
	require.Equal(t, 8, limitEncSize(^uint64(0)))
}
