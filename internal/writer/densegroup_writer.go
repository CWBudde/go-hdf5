// Copyright (c) 2025 SciGo HDF5 Library Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

package writer

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/structures"
)

// DenseGroupWriter manages dense group creation.
//
// Dense groups (HDF5 1.8+) use:
//   - Link Info Message: Metadata about link storage
//   - Fractal Heap: Storage for link names and messages
//   - B-tree v2: Index for fast link lookup by name
//
// This coordinator:
//  1. Creates Fractal Heap for link storage
//  2. Creates B-tree v2 for link indexing
//  3. Stores link names and metadata in heap
//  4. Indexes links in B-tree
//  5. Builds Link Info Message with addresses
//  6. Constructs object header with all messages
//
// Reference: H5Gdense.c - H5G_dense_create(), H5G_dense_insert().
type DenseGroupWriter struct {
	name string

	linkInfo *core.LinkInfoMessage

	// Links to add
	links []denseLink
}

// denseLink represents a link to be added to dense group.
type denseLink struct {
	name       string
	targetAddr uint64 // For hard links
	// Future: soft link path, external link file+path
}

// NewDenseGroupWriter creates new dense group writer.
//
// Parameters:
//   - name: Group name (for error messages)
//
// Returns:
//   - DenseGroupWriter ready to accept links
//
// Reference: H5Gdense.c - H5G_dense_create().
func NewDenseGroupWriter(name string) *DenseGroupWriter {
	return &DenseGroupWriter{
		name: name,
		linkInfo: &core.LinkInfoMessage{
			Version: 0,
			Flags:   0, // No creation order tracking for MVP
		},
		links: make([]denseLink, 0),
	}
}

// AddLink adds hard link to dense group.
//
// For MVP: Only hard links supported (targetAddr points to object header)
// Future: Soft links, external links
//
// Parameters:
//   - name: Link name (UTF-8 string)
//   - targetAddr: File address of target object header
//
// Returns:
//   - error if name empty, duplicate, or invalid
//
// Reference: H5Gdense.c - H5G_dense_insert().
func (dgw *DenseGroupWriter) AddLink(name string, targetAddr uint64) error {
	if name == "" {
		return errors.New("link name cannot be empty")
	}

	// Check for duplicates
	for _, link := range dgw.links {
		if link.name == name {
			return fmt.Errorf("duplicate link name: %s", name)
		}
	}

	dgw.links = append(dgw.links, denseLink{
		name:       name,
		targetAddr: targetAddr,
	})

	return nil
}

// WriteToFile writes dense group to file, returns object header address.
//
// This method:
//  1. Writes the links to dense storage (WriteDenseLinkStorage)
//  2. Creates the Link Info Message with heap/B-tree addresses
//  3. Writes the object header with Link Info + other messages
//
// Parameters:
//   - fw: FileWriter for write operations
//   - allocator: Space allocator
//   - sb: Superblock for encoding parameters
//
// Returns:
//   - uint64: File address of group's object header
//   - error: Non-nil if write fails
//
// Reference: H5Gdense.c - H5G_dense_create() + H5G_dense_insert().
func (dgw *DenseGroupWriter) WriteToFile(fw *FileWriter, allocator *Allocator, sb *core.Superblock) (uint64, error) {
	if len(dgw.links) == 0 {
		return 0, errors.New("dense group must have at least one link")
	}

	encoded := make([]EncodedLink, len(dgw.links))
	for i, link := range dgw.links {
		encoded[i] = EncodedLink{Name: link.name, Message: EncodeHardLinkMessage(link.name, link.targetAddr, sb)}
	}
	heapAddr, btreeAddr, err := WriteDenseLinkStorage(fw, allocator, sb, encoded)
	if err != nil {
		return 0, err
	}

	// Link Info Message
	dgw.linkInfo.FractalHeapAddress = heapAddr
	dgw.linkInfo.NameBTreeAddress = btreeAddr
	dgw.linkInfo.CreationOrderBTreeAddress = 0 // No creation order tracking in MVP

	// Object header with Link Info Message
	ohAddr, err := dgw.createObjectHeader(fw, allocator, sb)
	if err != nil {
		return 0, fmt.Errorf("failed to create object header: %w", err)
	}

	return ohAddr, nil
}

// EncodeHardLinkMessage encodes a version 1 Link message for a hard link
// to the object header at targetAddr. The same bytes serve as a compact Link
// message in a group's object header and as a dense storage heap object.
//
// Format (H5O__link_encode):
//
//	version (1) | flags (1) | [link type] [creation order] [charset] |
//	name length (1/2/4/8 bytes, flags bits 0-1) | name | link info
//
// A hard link with an ASCII name needs no optional fields; the link info is
// the target object header address.
func EncodeHardLinkMessage(name string, targetAddr uint64, sb *core.Superblock) []byte {
	nameBytes := []byte(name)
	nameLen := uint64(len(nameBytes))

	var sizeCode byte
	lenBytes := 1
	switch {
	case nameLen > 0xFFFFFFFF:
		sizeCode, lenBytes = 3, 8
	case nameLen > 0xFFFF:
		sizeCode, lenBytes = 2, 4
	case nameLen > 0xFF:
		sizeCode, lenBytes = 1, 2
	}

	buf := make([]byte, 0, 2+lenBytes+len(nameBytes)+int(sb.OffsetSize))
	buf = append(buf, 1, sizeCode)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], nameLen)
	buf = append(buf, lenBuf[:lenBytes]...)
	buf = append(buf, nameBytes...)
	addr := make([]byte, sb.OffsetSize)
	writeUint64(addr, targetAddr, int(sb.OffsetSize), sb.Endianness)
	return append(buf, addr...)
}

// EncodedLink is a link name with its encoded Link message.
type EncodedLink struct {
	Name    string
	Message []byte
}

// NewLinkHeap returns an empty fractal heap for dense link storage: 7-byte
// heap IDs (H5G_DENSE_FHEAP_ID_LEN) and a root direct block that grows as
// needed.
func NewLinkHeap() *structures.WritableFractalHeap {
	heap := structures.NewGrowableFractalHeap(structures.LinkHeapStartBlockSize)
	heap.SetMaxManagedObjectSize(structures.LinkHeapMaxManagedObjectSize)
	return heap
}

// InsertDenseLink stores an encoded Link message in the heap and indexes its
// name in the type 5 name index.
func InsertDenseLink(heap *structures.WritableFractalHeap, btree *structures.WritableBTreeV2, link EncodedLink) error {
	heapID, err := heap.InsertObject(link.Message)
	if err != nil {
		return fmt.Errorf("failed to insert link %s into heap: %w", link.Name, err)
	}
	// Link heap IDs are 7 bytes, zero-extended; the record stores the low 7 bytes.
	if len(heapID) == 0 || len(heapID) > 7 {
		return fmt.Errorf("invalid heap ID length for link %s: %d bytes", link.Name, len(heapID))
	}
	var idBuf [8]byte
	copy(idBuf[:], heapID)
	if err := btree.InsertRecord(link.Name, binary.LittleEndian.Uint64(idBuf[:])); err != nil {
		return fmt.Errorf("failed to insert link %s into B-tree: %w", link.Name, err)
	}
	return nil
}

// WriteDenseLinkStorage writes the fractal heap and the name index (B-tree
// v2 type 5) of a dense group holding links and returns their addresses for
// the group's Link Info message.
//
// Reference: H5Gdense.c - H5G__dense_create(), H5G__dense_insert().
func WriteDenseLinkStorage(fw *FileWriter, allocator *Allocator, sb *core.Superblock, links []EncodedLink) (heapAddr, btreeAddr uint64, err error) {
	heap := NewLinkHeap()
	btree := structures.NewWritableBTreeV2(0) // libhdf5 node size (512), grows as needed
	for _, link := range links {
		if err := InsertDenseLink(heap, btree, link); err != nil {
			return 0, 0, err
		}
	}
	if heapAddr, err = heap.WriteToFile(fw, allocator, sb); err != nil {
		return 0, 0, fmt.Errorf("failed to write fractal heap: %w", err)
	}
	if btreeAddr, err = btree.WriteToFile(fw, allocator, sb); err != nil {
		return 0, 0, fmt.Errorf("failed to write B-tree v2: %w", err)
	}
	return heapAddr, btreeAddr, nil
}

// createObjectHeader creates object header with Link Info Message.
//
// Messages to include:
//   - Link Info Message (type 0x0002)
//   - Group Info Message (type 0x000A) - libhdf5 needs it to add links
//   - Dataspace Message (type 0x0001) - scalar for groups
//   - Datatype Message (type 0x0003) - opaque for groups (optional, skipped in MVP)
//
// Reference: H5Oobj.c - H5O_obj_create().
func (dgw *DenseGroupWriter) createObjectHeader(fw *FileWriter, allocator *Allocator, sb *core.Superblock) (uint64, error) {
	// Encode Link Info Message
	linkInfoData, err := core.EncodeLinkInfoMessage(dgw.linkInfo, sb)
	if err != nil {
		return 0, fmt.Errorf("failed to encode link info message: %w", err)
	}

	// Create dataspace message (scalar for groups)
	dataspaceMsg := createScalarDataspaceMessage()

	// Create object header with messages
	ohw := &core.ObjectHeaderWriter{
		Version: 2,
		Flags:   0,
		Messages: []core.MessageWriter{
			{Type: core.MsgLinkInfo, Data: linkInfoData},
			{Type: core.MsgGroupInfo, Data: core.EncodeGroupInfoMessage(&core.GroupInfoMessage{})},
			{Type: core.MsgDataspace, Data: dataspaceMsg},
		},
	}

	// Calculate object header size (prefix + messages + checksum)
	headerSize := ohw.Size()

	// Allocate space for object header
	headerAddr, err := allocator.Allocate(headerSize)
	if err != nil {
		return 0, fmt.Errorf("failed to allocate object header: %w", err)
	}

	// Write object header
	writtenSize, err := ohw.WriteTo(fw, headerAddr)
	if err != nil {
		return 0, fmt.Errorf("failed to write object header: %w", err)
	}

	if writtenSize != headerSize {
		return 0, fmt.Errorf("header size mismatch: expected %d, wrote %d", headerSize, writtenSize)
	}

	return headerAddr, nil
}

// createScalarDataspaceMessage creates a dataspace message for scalar dataspace.
//
// Groups have scalar dataspace (0 dimensions).
//
// Format:
//   - Version: 1 (1 byte) or 2
//   - Dimensionality: 0 (1 byte)
//   - Flags: 0 (1 byte)
//   - Type: 0 = scalar (1 byte) - optional in v1
//
// Reference: H5Odtype.c - H5O__dspace_encode().
func createScalarDataspaceMessage() []byte {
	// Use version 1 for simplicity
	// Version 1: 1 (ver) + 1 (dims) + 1 (flags) + 5 (reserved) = 8 bytes
	buf := make([]byte, 8)

	buf[0] = 1 // Version 1
	buf[1] = 0 // Dimensionality = 0 (scalar)
	buf[2] = 0 // Flags = 0
	// Bytes 3-7: reserved (zeros)

	return buf
}

// compactUint64Size returns the number of bytes needed to encode a uint64 in compact form.
//
// Compact encoding uses 1-8 bytes based on the value:
//   - Values 0-255: 1 byte
//   - Values 256-65535: 2 bytes
//   - etc.
func compactUint64Size(value uint64) int {
	if value == 0 {
		return 1
	}

	size := 0
	for value > 0 {
		size++
		value >>= 8
	}

	return size
}

// encodeCompactUint64 encodes a uint64 value in compact form (little-endian).
//
// The size is determined by compactUint64Size().
func encodeCompactUint64(buf []byte, value uint64) {
	size := compactUint64Size(value)
	for i := 0; i < size; i++ {
		buf[i] = byte(value >> (8 * i))
	}
}

// writeUint64 writes a uint64 value to buffer with specified size and endianness.
// This is a helper function for encoding fields with variable sizes.
func writeUint64(buf []byte, value uint64, size int, endianness binary.ByteOrder) {
	switch size {
	case 1:
		buf[0] = byte(value)
	case 2:
		endianness.PutUint16(buf, uint16(value)) //nolint:gosec // Safe: size limited to 2 bytes
	case 4:
		endianness.PutUint32(buf, uint32(value)) //nolint:gosec // Safe: size limited to 4 bytes
	case 8:
		endianness.PutUint64(buf, value)
	}
}
