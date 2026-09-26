package structures

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/utils"
)

// maxGroupBTreeDepth bounds the levels of a group B-tree followed on read.
const maxGroupBTreeDepth = 16

// ReadGroupBTreeEntries reads entries from a "TREE" format B-tree (type 0 - group symbol table).
// This is the v1 B-tree format used in v0 and some v1 HDF5 files for indexing group entries.
//
// Leaf nodes (level 0) point to Symbol Table Nodes (SNODs); internal nodes
// point to B-tree nodes one level down. Keys are local heap offsets of link
// names. The function walks the whole tree, follows the child pointers of
// the leaves to the SNODs and collects all their entries in key order.
func ReadGroupBTreeEntries(r io.ReaderAt, address uint64, sb *core.Superblock) ([]BTreeEntry, error) {
	var snodAddresses []uint64
	visited := map[uint64]bool{}
	if err := collectGroupBTreeSNODs(r, address, -1, sb, visited, &snodAddresses); err != nil {
		return nil, err
	}

	// Parse each SNOD to collect entries
	var allEntries []BTreeEntry
	for _, snodAddr := range snodAddresses {
		snodNode, err := ParseSymbolTableNode(r, snodAddr, sb)
		if err != nil {
			// Skip invalid SNODs
			continue
		}

		// Convert SNOD entries to BTreeEntry format
		for _, entry := range snodNode.Entries {
			allEntries = append(allEntries, BTreeEntry{
				LinkNameOffset:  entry.LinkNameOffset,
				ObjectAddress:   entry.ObjectAddress,
				CacheType:       entry.CacheType,
				Reserved:        0,
				CachedBTreeAddr: entry.CachedBTreeAddr,
				CachedHeapAddr:  entry.CachedHeapAddr,
			})
		}
	}

	return allEntries, nil
}

// collectGroupBTreeSNODs appends the SNOD addresses below the group B-tree
// node at address to out. wantLevel is the level the parent expects (-1 for
// the root). Levels must decrease by one per step and nodes may not repeat.
func collectGroupBTreeSNODs(r io.ReaderAt, address uint64, wantLevel int, sb *core.Superblock,
	visited map[uint64]bool, out *[]uint64,
) error {
	if visited[address] {
		return fmt.Errorf("group B-tree node 0x%X referenced twice", address)
	}
	visited[address] = true

	// Node header: signature (4), type (1), level (1), entries used (2),
	// left and right sibling addresses.
	o := int(sb.OffsetSize)
	l := int(sb.LengthSize)
	headerSize := 8 + 2*o
	header, err := utils.ReadAtChecked(r, address, uint64(headerSize), "group B-tree node header") //nolint:gosec // G115: small
	if err != nil {
		return err
	}
	if sig := string(header[0:4]); sig != "TREE" {
		return fmt.Errorf("invalid B-tree signature: %q (expected TREE)", sig)
	}
	if header[4] != 0 {
		return fmt.Errorf("expected group B-tree (type 0), got type %d", header[4])
	}
	level := int(header[5])
	if level > maxGroupBTreeDepth || (wantLevel >= 0 && level != wantLevel) {
		return fmt.Errorf("group B-tree node 0x%X has unexpected level %d", address, level)
	}
	entriesUsed := int(sb.Endianness.Uint16(header[6:8]))
	if entriesUsed == 0 {
		return nil
	}

	// Keys and children interleaved: key[0], child[0], key[1], ...,
	// child[n-1], key[n]. Keys are heap offsets (LengthSize bytes).
	dataSize := uint64(entriesUsed) * uint64(l+o)                                                  //nolint:gosec // G115: small
	data, err := utils.ReadAtChecked(r, address+uint64(headerSize), dataSize, "group B-tree node") //nolint:gosec // G115: small
	if err != nil {
		return err
	}
	for i := 0; i < entriesUsed; i++ {
		pos := i*(l+o) + l
		childAddr := readAddress(data[pos:], o, sb.Endianness)
		if childAddr == 0 || childAddr == 0xFFFFFFFFFFFFFFFF {
			continue
		}
		if level == 0 {
			*out = append(*out, childAddr)
			continue
		}
		if err := collectGroupBTreeSNODs(r, childAddr, level-1, sb, visited, out); err != nil {
			return err
		}
	}
	return nil
}

// readAddress reads a variable-sized address from byte slice using the specified endianness.
//
//nolint:gosec // G602: bounds are checked by clamping size to len(data) before switch
func readAddress(data []byte, size int, endianness binary.ByteOrder) uint64 {
	if size > len(data) {
		size = len(data)
	}

	switch size {
	case 1:
		return uint64(data[0])
	case 2:
		return uint64(endianness.Uint16(data[:2]))
	case 4:
		return uint64(endianness.Uint32(data[:4]))
	case 8:
		return endianness.Uint64(data[:8])
	default:
		// Pad to 8 bytes.
		var buf [8]byte
		copy(buf[:], data[:size])
		return endianness.Uint64(buf[:])
	}
}

// BTreeNodeV1 represents a B-tree version 1 node for group symbol tables.
// This is the "TREE" format used for indexing symbol table nodes.
//
// Format:
// - 4 bytes: Signature ("TREE")
// - 1 byte: Node type (0 = group B-tree)
// - 1 byte: Node level (0 = leaf, 1+ = internal)
// - 2 bytes: Number of entries used
// - offsetSize bytes: Left sibling address (0xFFFFFFFFFFFFFFFF for none)
// - offsetSize bytes: Right sibling address (0xFFFFFFFFFFFFFFFF for none)
// - Then: 2K+1 keys (each offsetSize bytes) alternating with 2K child addresses
//
// For MVP (single node), we simplify:
// - Only leaf nodes (level = 0)
// - Only one child pointer (to symbol table node)
// - Left/right siblings are undefined.
type BTreeNodeV1 struct {
	Signature     [4]byte  // "TREE"
	NodeType      uint8    // 0 = group symbol table
	NodeLevel     uint8    // 0 = leaf
	EntriesUsed   uint16   // Number of entries
	LeftSibling   uint64   // Address of left sibling (UNDEF for none)
	RightSibling  uint64   // Address of right sibling (UNDEF for none)
	Keys          []uint64 // Link name offsets in heap (2K+1 for full node)
	ChildPointers []uint64 // Child node addresses (2K for full node, or symbol table node addresses for leaf)
}

// NewBTreeNodeV1 creates a new B-tree v1 node for group symbol tables.
// For MVP, this is always a leaf node pointing to a single symbol table node.
func NewBTreeNodeV1(nodeType uint8, k uint16) *BTreeNodeV1 {
	return &BTreeNodeV1{
		Signature:     [4]byte{'T', 'R', 'E', 'E'},
		NodeType:      nodeType,
		NodeLevel:     0, // Leaf node for MVP
		EntriesUsed:   0,
		LeftSibling:   0xFFFFFFFFFFFFFFFF,            // Undefined
		RightSibling:  0xFFFFFFFFFFFFFFFF,            // Undefined
		Keys:          make([]uint64, 0, 2*int(k)+1), // 2K+1 keys
		ChildPointers: make([]uint64, 0, 2*int(k)),   // 2K children
	}
}

// AddKey adds a key and child pointer to the B-tree node.
// For leaf nodes in groups, keys are link name offsets in the local heap,
// and child pointers are addresses of symbol table nodes.
func (btn *BTreeNodeV1) AddKey(key, childAddr uint64) error {
	maxKeys := cap(btn.Keys)
	if len(btn.Keys) >= maxKeys {
		return fmt.Errorf("b-tree node is full (%d/%d keys)", len(btn.Keys), maxKeys)
	}

	// For MVP: simple append (no balancing)
	btn.Keys = append(btn.Keys, key)
	btn.ChildPointers = append(btn.ChildPointers, childAddr)
	btn.EntriesUsed++

	return nil
}

// WriteAt writes the B-tree node to w at the specified address.
// offsetSize determines the size of addresses in the file (typically 8).
// K is the B-tree order (default 16, so 2K+1 = 33 keys).
func (btn *BTreeNodeV1) WriteAt(w io.WriterAt, address uint64, offsetSize uint8, k uint16, endianness binary.ByteOrder) error {
	// Calculate sizes
	maxKeys := 2*int(k) + 1
	maxChildren := 2 * int(k)

	// Header size: 4 (sig) + 1 (type) + 1 (level) + 2 (entries) + 2*offsetSize (siblings)
	headerSize := 8 + 2*int(offsetSize)

	// Keys and children are interleaved:
	// Key[0], Child[0], Key[1], Child[1], ..., Key[2K], Child[2K-1], Key[2K]
	// Total: (2K+1) keys * offsetSize + 2K children * offsetSize
	keysSize := maxKeys * int(offsetSize)
	childrenSize := maxChildren * int(offsetSize)

	totalSize := headerSize + keysSize + childrenSize
	buf := make([]byte, totalSize)

	// Write header
	pos := 0
	copy(buf[pos:], btn.Signature[:])
	pos += 4
	buf[pos] = btn.NodeType
	pos++
	buf[pos] = btn.NodeLevel
	pos++
	endianness.PutUint16(buf[pos:], btn.EntriesUsed)
	pos += 2

	// Write left sibling
	writeAddr(buf[pos:], btn.LeftSibling, int(offsetSize), endianness)
	pos += int(offsetSize)

	// Write right sibling
	writeAddr(buf[pos:], btn.RightSibling, int(offsetSize), endianness)
	pos += int(offsetSize)

	// Write keys and children (interleaved)
	// For a leaf B-tree in groups: keys are heap offsets, children are symbol table node addresses
	for i := 0; i < maxKeys; i++ {
		var key uint64
		if i < len(btn.Keys) {
			key = btn.Keys[i]
		}
		// If i >= len(Keys), key is 0 (padding)

		writeAddr(buf[pos:], key, int(offsetSize), endianness)
		pos += int(offsetSize)

		// After each key (except the last), write a child pointer
		if i < maxChildren {
			var child uint64
			if i < len(btn.ChildPointers) {
				child = btn.ChildPointers[i]
			}

			writeAddr(buf[pos:], child, int(offsetSize), endianness)
			pos += int(offsetSize)
		}
	}

	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.WriterAt interface
	_, err := w.WriteAt(buf, int64(address))
	return err
}

// writeAddr writes a variable-sized address to byte slice.
func writeAddr(data []byte, addr uint64, size int, endianness binary.ByteOrder) {
	if size > len(data) {
		size = len(data)
	}

	switch size {
	case 1:
		data[0] = byte(addr)
	case 2:
		endianness.PutUint16(data[:2], uint16(addr)) //nolint:gosec // Safe: address size matches offset size
	case 4:
		endianness.PutUint32(data[:4], uint32(addr)) //nolint:gosec // Safe: address size matches offset size
	case 8:
		endianness.PutUint64(data[:8], addr)
	default:
		// Pad to requested size
		var buf [8]byte
		endianness.PutUint64(buf[:], addr)
		copy(data[:size], buf[:size])
	}
}
