package core

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/bits"

	"github.com/cwbudde/go-hdf5/internal/utils"
)

// Version 2 B-tree node layout constants (H5B2pkg.h).
const (
	btreeV2NodePrefixSize = 4 + 1 + 1 + 4 // signature + version + type + checksum
	btreeV2MaxDepth       = 32            // far beyond any real tree (depth 32 of 2-record nodes is > 2^32 records)
)

// BTreeV2Info holds the fields of a version 2 B-tree header.
type BTreeV2Info struct {
	Type         uint8
	NodeSize     uint32
	RecordSize   uint16
	Depth        uint16
	RootAddr     uint64
	RootNRec     uint16
	TotalRecords uint64
}

// ReadBTreeV2Info reads and validates (signature, version, checksum) the
// version 2 B-tree header at addr.
func ReadBTreeV2Info(r io.ReaderAt, addr uint64, sb *Superblock) (*BTreeV2Info, error) {
	o := int(sb.OffsetSize)
	size := 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + o + 2 + 8 + 4
	buf, err := utils.ReadAtChecked(r, addr, uint64(size), "B-tree v2 header") //nolint:gosec // G115: small constant
	if err != nil {
		return nil, err
	}
	if string(buf[0:4]) != "BTHD" {
		return nil, fmt.Errorf("invalid B-tree v2 header signature %q at 0x%X", buf[0:4], addr)
	}
	if buf[4] != 0 {
		return nil, fmt.Errorf("unsupported B-tree v2 header version %d", buf[4])
	}
	if stored, want := binary.LittleEndian.Uint32(buf[size-4:]), utils.JenkinsChecksum(buf[:size-4]); stored != want {
		return nil, fmt.Errorf("b-tree v2 header checksum mismatch at 0x%X", addr)
	}
	info := &BTreeV2Info{
		Type:       buf[5],
		NodeSize:   binary.LittleEndian.Uint32(buf[6:10]),
		RecordSize: binary.LittleEndian.Uint16(buf[10:12]),
		Depth:      binary.LittleEndian.Uint16(buf[12:14]),
	}
	pos := 16 // after split and merge percent
	info.RootAddr = readAddress(buf[pos:pos+o], o)
	pos += o
	info.RootNRec = binary.LittleEndian.Uint16(buf[pos : pos+2])
	pos += 2
	info.TotalRecords = binary.LittleEndian.Uint64(buf[pos : pos+8])
	return info, nil
}

// btreeV2LevelInfo mirrors H5B2_node_info_t: per-depth record capacities.
type btreeV2LevelInfo struct {
	maxNRec        uint64
	cumMaxNRec     uint64
	cumMaxNRecSize int
}

// limitEncSize mirrors H5VM_limit_enc_size: the number of bytes needed to
// encode values up to l.
func limitEncSize(l uint64) int {
	lg := 0
	if l > 0 {
		lg = bits.Len64(l) - 1
	}
	return lg/8 + 1
}

// btreeV2Layout holds the derived node geometry of a version 2 B-tree
// (H5B2__hdr_init).
type btreeV2Layout struct {
	info        *BTreeV2Info
	offsetSize  int
	levels      []btreeV2LevelInfo
	maxNRecSize int
}

func newBTreeV2Layout(info *BTreeV2Info, offsetSize int) (*btreeV2Layout, error) {
	if info.RecordSize == 0 {
		return nil, errors.New("b-tree v2 record size is zero")
	}
	if info.Depth > btreeV2MaxDepth {
		return nil, fmt.Errorf("b-tree v2 depth %d exceeds limit %d", info.Depth, btreeV2MaxDepth)
	}
	nodeSize := uint64(info.NodeSize)
	recSize := uint64(info.RecordSize)
	if nodeSize < btreeV2NodePrefixSize+recSize {
		return nil, fmt.Errorf("b-tree v2 node size %d too small for %d-byte records", nodeSize, recSize)
	}
	l := &btreeV2Layout{info: info, offsetSize: offsetSize, levels: make([]btreeV2LevelInfo, int(info.Depth)+1)}
	leafMax := (nodeSize - btreeV2NodePrefixSize) / recSize
	l.levels[0] = btreeV2LevelInfo{maxNRec: leafMax, cumMaxNRec: leafMax}
	l.maxNRecSize = limitEncSize(leafMax)
	for u := 1; u <= int(info.Depth); u++ {
		ptr := uint64(l.pointerSize(u)) //nolint:gosec // G115: small
		if nodeSize < btreeV2NodePrefixSize+ptr+recSize+ptr {
			return nil, fmt.Errorf("b-tree v2 node size %d too small for internal nodes", nodeSize)
		}
		maxNRec := (nodeSize - (btreeV2NodePrefixSize + ptr)) / (recSize + ptr)
		hi, lo := bits.Mul64(maxNRec+1, l.levels[u-1].cumMaxNRec)
		cum, carry := bits.Add64(lo, maxNRec, 0)
		if hi != 0 || carry != 0 {
			cum = ^uint64(0)
		}
		l.levels[u] = btreeV2LevelInfo{maxNRec: maxNRec, cumMaxNRec: cum, cumMaxNRecSize: limitEncSize(cum)}
	}
	return l, nil
}

// pointerSize is H5B2_INT_POINTER_SIZE for an internal node at depth d.
func (l *btreeV2Layout) pointerSize(d int) int {
	s := l.offsetSize + l.maxNRecSize
	if d > 1 {
		s += l.levels[d-1].cumMaxNRecSize
	}
	return s
}

// ReadBTreeV2Records reads the version 2 B-tree whose header is at addr and
// returns its header and all records (raw, RecordSize bytes each) in key
// order. Internal nodes of any depth are traversed; node signatures,
// checksums, record counts and addresses are validated and cycles rejected.
func ReadBTreeV2Records(r io.ReaderAt, addr uint64, sb *Superblock) (*BTreeV2Info, [][]byte, error) {
	info, err := ReadBTreeV2Info(r, addr, sb)
	if err != nil {
		return nil, nil, err
	}
	if info.TotalRecords == 0 || info.RootNRec == 0 {
		return info, nil, nil
	}
	layout, err := newBTreeV2Layout(info, int(sb.OffsetSize))
	if err != nil {
		return nil, nil, err
	}
	w := &btreeV2Walker{r: r, layout: layout, visited: map[uint64]bool{}}
	if err := w.walk(info.RootAddr, uint64(info.RootNRec), int(info.Depth)); err != nil {
		return nil, nil, err
	}
	if uint64(len(w.records)) != info.TotalRecords {
		return nil, nil, fmt.Errorf("b-tree v2 at 0x%X holds %d records, header says %d", addr, len(w.records), info.TotalRecords)
	}
	return info, w.records, nil
}

type btreeV2Walker struct {
	r       io.ReaderAt
	layout  *btreeV2Layout
	visited map[uint64]bool
	records [][]byte
}

func (w *btreeV2Walker) walk(addr, nrec uint64, depth int) error {
	buf, err := w.readNode(addr, nrec, depth)
	if err != nil {
		return err
	}
	recSize := uint64(w.layout.info.RecordSize)
	recAt := func(i uint64) []byte {
		start := 6 + i*recSize
		return append([]byte(nil), buf[start:start+recSize]...)
	}
	if depth == 0 {
		for i := uint64(0); i < nrec; i++ {
			w.records = append(w.records, recAt(i))
		}
		return nil
	}

	o := uint64(w.layout.offsetSize)               //nolint:gosec // G115: small
	nrecSize := uint64(w.layout.maxNRecSize)       //nolint:gosec // G115: small
	ptrSize := uint64(w.layout.pointerSize(depth)) //nolint:gosec // G115: small
	pos := 6 + nrec*recSize
	for i := uint64(0); i <= nrec; i++ {
		child := readAddress(buf[pos:pos+o], int(o)) //nolint:gosec // G115: small
		childNRec := readVarLE(buf[pos+o : pos+o+nrecSize])
		pos += ptrSize // the subtree record count (depth > 1) is not needed
		if err := w.walk(child, childNRec, depth-1); err != nil {
			return err
		}
		if i < nrec {
			w.records = append(w.records, recAt(i))
		}
	}
	return nil
}

// readNode reads and validates the node at addr holding nrec records at the
// given depth (0 = leaf).
func (w *btreeV2Walker) readNode(addr, nrec uint64, depth int) ([]byte, error) {
	info := w.layout.info
	if addr == 0 || addr == ^uint64(0) {
		return nil, fmt.Errorf("invalid B-tree v2 node address 0x%X", addr)
	}
	if w.visited[addr] {
		return nil, fmt.Errorf("b-tree v2 node 0x%X referenced twice (cycle)", addr)
	}
	w.visited[addr] = true
	if nrec > w.layout.levels[depth].maxNRec {
		return nil, fmt.Errorf("b-tree v2 node 0x%X claims %d records, at most %d fit", addr, nrec, w.layout.levels[depth].maxNRec)
	}
	if uint64(len(w.records))+nrec > info.TotalRecords {
		return nil, fmt.Errorf("b-tree v2 holds more records than the %d in its header", info.TotalRecords)
	}
	sig := "BTLF"
	size := btreeV2NodePrefixSize + nrec*uint64(info.RecordSize)
	if depth > 0 {
		sig = "BTIN"
		size += (nrec + 1) * uint64(w.layout.pointerSize(depth)) //nolint:gosec // G115: small
	}
	buf, err := utils.ReadAtChecked(w.r, addr, size, "B-tree v2 node")
	if err != nil {
		return nil, err
	}
	if string(buf[0:4]) != sig {
		return nil, fmt.Errorf("invalid B-tree v2 node signature %q at 0x%X (want %s)", buf[0:4], addr, sig)
	}
	if buf[4] != 0 {
		return nil, fmt.Errorf("unsupported B-tree v2 node version %d at 0x%X", buf[4], addr)
	}
	if buf[5] != info.Type {
		return nil, fmt.Errorf("b-tree v2 node 0x%X has type %d, tree has type %d", addr, buf[5], info.Type)
	}
	if stored, want := binary.LittleEndian.Uint32(buf[size-4:]), utils.JenkinsChecksum(buf[:size-4]); stored != want {
		return nil, fmt.Errorf("b-tree v2 node checksum mismatch at 0x%X", addr)
	}
	return buf, nil
}

// readVarLE decodes a little-endian unsigned integer of len(b) (<= 8) bytes.
func readVarLE(b []byte) uint64 {
	var v uint64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}
