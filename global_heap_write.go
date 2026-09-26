package hdf5

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/cwbudde/go-hdf5/internal/core"
)

// globalHeapWriter manages writing variable-length data to HDF5 global heaps.
// Global heaps store variable-length data (strings, ragged arrays) separately from datasets.
// Datasets store heap IDs (references) that point to the actual data in the heap.
type globalHeapWriter struct {
	fileWriter        *FileWriter
	currentHeap       *globalHeapCollectionBuilder
	minCollectionSize uint64 // Minimum heap collection size (4KB default)

	// dimensionHeap is the collection reserved by reserveDimensionHeap for
	// the DIMENSION_LIST references (nil if none was reserved).
	dimensionHeap *globalHeapCollectionBuilder
}

// globalHeapCollectionBuilder is used to build a global heap collection before writing.
type globalHeapCollectionBuilder struct {
	address   uint64
	size      uint64
	objects   []*globalHeapObjectBuilder
	nextIndex uint16
	usedSpace uint64 // Space used by objects (including headers)
	freeSpace uint64 // Remaining free space
}

// globalHeapObjectBuilder represents an object being added to the heap.
type globalHeapObjectBuilder struct {
	index    uint16
	refCount uint16
	data     []byte
}

// HeapID identifies a global heap object (collection address + object index).
// Format: 8 bytes address + 4 bytes index (stored as 12 bytes in file, padded to 16).
type HeapID struct {
	CollectionAddress uint64
	ObjectIndex       uint16
}

// newGlobalHeapWriter creates a new global heap writer.
func newGlobalHeapWriter(fw *FileWriter) *globalHeapWriter {
	return &globalHeapWriter{
		fileWriter:        fw,
		minCollectionSize: 4096, // 4KB default minimum
	}
}

// WriteToGlobalHeap writes data to the global heap and returns a heap ID.
// This handles creating new heap collections as needed and managing space.
// Empty data (len=0) is allowed - it will be written to heap with size 0.
func (ghw *globalHeapWriter) WriteToGlobalHeap(data []byte) (HeapID, error) {
	// Calculate space needed for this object
	// Object header: 2 (index) + 2 (refcount) + 4 (reserved) + 8 (size) = 16 bytes
	// Data: len(data), aligned to 8-byte boundary
	objectHeaderSize := uint64(16)
	alignedDataSize := alignTo8(uint64(len(data)))
	totalObjectSize := objectHeaderSize + alignedDataSize

	// Check if we need a new heap collection
	needsNewHeap := ghw.currentHeap == nil || !ghw.currentHeap.hasSpace(totalObjectSize)
	//nolint:nestif // Reasonable nesting: flush old heap if exists, then create new
	if needsNewHeap {
		// Flush current heap if it exists
		if ghw.currentHeap != nil {
			if err := ghw.flushCurrentHeap(); err != nil {
				return HeapID{}, fmt.Errorf("flush current heap: %w", err)
			}
		}

		// Create new heap collection
		if err := ghw.createNewHeap(totalObjectSize); err != nil {
			return HeapID{}, fmt.Errorf("create new heap: %w", err)
		}
	}

	// Add object to current heap
	objIndex := ghw.currentHeap.addObject(data)

	// Return heap ID
	return HeapID{
		CollectionAddress: ghw.currentHeap.address,
		ObjectIndex:       objIndex,
	}, nil
}

// reserveDimensionHeap allocates a collection at the current end of file
// that only WriteDimensionReferences writes into. FileWriter reserves it
// right after the root group: libmysofa cannot resolve references into a
// collection beyond 64 KiB, and variable-length data written before Close
// would otherwise fill it and push the references into a later one.
func (ghw *globalHeapWriter) reserveDimensionHeap() error {
	heap, err := ghw.newCollection(0)
	if err != nil {
		return err
	}
	ghw.dimensionHeap = heap
	return nil
}

// WriteDimensionReferences writes one DIMENSION_LIST sequence of object
// references to the reserved collection. Without one, or when it is full,
// the data goes to the collections used for other variable-length data.
// The reserved 4 KiB hold about 170 single-reference sequences, one per
// dimension of every dataset with attached scales; a SOFA file needs about
// 30. Beyond that libmysofa cannot read the file anyway, as it looks up all
// references in one collection, but other readers can.
func (ghw *globalHeapWriter) WriteDimensionReferences(data []byte) (HeapID, error) {
	heap := ghw.dimensionHeap
	if heap == nil || !heap.hasSpace(globalHeapObjectSize(data)) {
		return ghw.WriteToGlobalHeap(data)
	}
	return HeapID{CollectionAddress: heap.address, ObjectIndex: heap.addObject(data)}, nil
}

// collectionBuilderFrom rebuilds a collection read from the file, so that
// objects can be added to it and it can be rewritten in place. It returns
// nil if the objects do not fit the collection as this writer encodes it.
func collectionBuilderFrom(c *core.GlobalHeapCollection) *globalHeapCollectionBuilder {
	heap := &globalHeapCollectionBuilder{
		address:   c.Address,
		size:      c.Size,
		nextIndex: 1,
		usedSpace: 16, // collection header
	}
	for _, o := range c.Objects {
		if o.Index <= 0 || o.Index >= math.MaxUint16 {
			return nil
		}
		index := uint16(o.Index)
		heap.objects = append(heap.objects, &globalHeapObjectBuilder{index: index, refCount: o.NRefs, data: o.Data})
		heap.usedSpace += globalHeapObjectSize(o.Data)
		heap.nextIndex = max(heap.nextIndex, index+1)
	}
	if heap.usedSpace > heap.size {
		return nil
	}
	heap.freeSpace = heap.size - heap.usedSpace
	return heap
}

// globalHeapObjectSize is the size of data stored as a heap object: a
// 16-byte header followed by the data padded to 8 bytes.
func globalHeapObjectSize(data []byte) uint64 {
	return 16 + alignTo8(uint64(len(data)))
}

// createNewHeap creates a new global heap collection with enough space for the object.
func (ghw *globalHeapWriter) createNewHeap(minObjectSize uint64) error {
	heap, err := ghw.newCollection(minObjectSize)
	if err != nil {
		return err
	}
	ghw.currentHeap = heap
	return nil
}

// newCollection allocates a global heap collection with room for an
// object of minObjectSize bytes (header included).
func (ghw *globalHeapWriter) newCollection(minObjectSize uint64) (*globalHeapCollectionBuilder, error) {
	// Calculate heap collection size
	// Header: 4 (signature) + 1 (version) + 3 (reserved) + 8 (size) = 16 bytes
	headerSize := uint64(16)

	// Need space for: header + objects + free space marker (16 bytes)
	neededSize := headerSize + minObjectSize + 16

	// Use minimum collection size or needed size, whichever is larger
	collectionSize := ghw.minCollectionSize
	if neededSize > collectionSize {
		// Round up to next 4KB boundary
		collectionSize = ((neededSize + 4095) / 4096) * 4096
	}

	// Allocate space in file
	heapAddr, err := ghw.fileWriter.writer.Allocate(collectionSize)
	if err != nil {
		return nil, fmt.Errorf("allocate heap space: %w", err)
	}

	return &globalHeapCollectionBuilder{
		address:   heapAddr,
		size:      collectionSize,
		objects:   make([]*globalHeapObjectBuilder, 0),
		nextIndex: 1, // Object indices start at 1 (0 is reserved for free space)
		usedSpace: headerSize,
		freeSpace: collectionSize - headerSize,
	}, nil
}

// flushCurrentHeap writes the current heap collection to disk.
func (ghw *globalHeapWriter) flushCurrentHeap() error {
	return ghw.flushHeap(ghw.currentHeap)
}

// flushHeap writes a heap collection to disk.
func (ghw *globalHeapWriter) flushHeap(heap *globalHeapCollectionBuilder) error {
	if heap == nil {
		return nil
	}

	// Encode heap collection to bytes
	heapData := encodeHeapCollection(heap)

	// Write to file
	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.WriterAt interface
	if _, err := ghw.fileWriter.writer.WriteAt(heapData, int64(heap.address)); err != nil {
		return fmt.Errorf("write heap to file: %w", err)
	}

	return nil
}

// encodeHeapCollection encodes a heap collection to bytes.
// Format follows HDF5 spec section 3.2.6.
func encodeHeapCollection(heap *globalHeapCollectionBuilder) []byte {
	buf := make([]byte, heap.size)

	offset := 0

	// 1. Signature (4 bytes): "GCOL"
	copy(buf[offset:], core.GlobalHeapSignature[:])
	offset += 4

	// 2. Version (1 byte): always 1
	buf[offset] = 1
	offset++

	// 3. Reserved (3 bytes): zeros
	offset += 3

	// 4. Collection size (8 bytes)
	binary.LittleEndian.PutUint64(buf[offset:], heap.size)
	offset += 8

	// Now write objects, each aligned to 8-byte boundary
	// Align start of objects to 8 bytes
	if offset%8 != 0 {
		padding := 8 - (offset % 8)
		offset += padding
	}

	// 5. Write each object
	for _, obj := range heap.objects {
		// Object header (16 bytes):
		// - Object index (2 bytes)
		binary.LittleEndian.PutUint16(buf[offset:], obj.index)
		offset += 2

		// - Reference count (2 bytes)
		binary.LittleEndian.PutUint16(buf[offset:], obj.refCount)
		offset += 2

		// - Reserved (4 bytes): zeros
		offset += 4

		// - Object size (8 bytes)
		binary.LittleEndian.PutUint64(buf[offset:], uint64(len(obj.data)))
		offset += 8

		// Object data
		copy(buf[offset:], obj.data)
		offset += len(obj.data)

		// Align to 8-byte boundary
		if offset%8 != 0 {
			padding := 8 - (offset % 8)
			offset += padding
		}
	}

	// 6. Free space marker (object with index 0)
	// This indicates remaining free space in the collection
	if heap.freeSpace >= 16 {
		// Write free space object header
		binary.LittleEndian.PutUint16(buf[offset:], 0) // Index 0 = free space
		offset += 2

		binary.LittleEndian.PutUint16(buf[offset:], 0) // Reference count 0
		offset += 2

		offset += 4 // Reserved

		// Free space size. Like libhdf5 (H5HG_create), the size of object 0
		// covers the whole free region including this header; a size that
		// excludes the header makes libhdf5 parse a zero-sized object 0 at
		// the tail of the collection and loop forever.
		binary.LittleEndian.PutUint64(buf[offset:], heap.freeSpace)
		offset += 8
	}

	// Rest of buffer is already zeros (free space)
	_ = offset // offset tracking complete
	return buf
}

// Encode encodes a heap ID to 16 bytes (HDF5 vlen reference format).
// Format: 8 bytes address + 4 bytes index + 4 bytes padding.
func (hid HeapID) Encode() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint64(buf[0:8], hid.CollectionAddress)
	binary.LittleEndian.PutUint32(buf[8:12], uint32(hid.ObjectIndex))
	// Bytes 12-15 are padding (already zeros)
	return buf
}

// vlenElementSize is the on-disk size of one variable-length element:
// 4-byte sequence length + 8-byte collection address + 4-byte object index.
const vlenElementSize = 16

// encodeVLenElement encodes one variable-length element as stored in a
// dataset or attribute: sequence length (element count, bytes for strings)
// followed by the global heap ID.
//
// Reference: HDF5 spec IV.A.2.d (variable-length data), H5Tvlen.c - H5T__vlen_disk_write().
func encodeVLenElement(length uint32, hid HeapID) []byte {
	buf := make([]byte, vlenElementSize)
	binary.LittleEndian.PutUint32(buf[0:4], length)
	binary.LittleEndian.PutUint64(buf[4:12], hid.CollectionAddress)
	binary.LittleEndian.PutUint32(buf[12:16], uint32(hid.ObjectIndex))
	return buf
}

// hasSpace checks if the heap has enough space for an object of given size.
func (ghc *globalHeapCollectionBuilder) hasSpace(objectSize uint64) bool {
	return ghc.freeSpace >= objectSize
}

// addObject adds an object to the heap collection and returns its index.
func (ghc *globalHeapCollectionBuilder) addObject(data []byte) uint16 {
	obj := &globalHeapObjectBuilder{
		index:    ghc.nextIndex,
		refCount: 1, // New objects have refcount 1
		data:     data,
	}

	ghc.objects = append(ghc.objects, obj)

	// Update space tracking
	objectHeaderSize := uint64(16)
	alignedDataSize := alignTo8(uint64(len(data)))
	totalSize := objectHeaderSize + alignedDataSize
	ghc.usedSpace += totalSize
	ghc.freeSpace -= totalSize

	objIndex := ghc.nextIndex
	ghc.nextIndex++

	return objIndex
}

// alignTo8 aligns a size to the next 8-byte boundary.
func alignTo8(size uint64) uint64 {
	if size%8 == 0 {
		return size
	}
	return size + (8 - size%8)
}

// Flush writes any pending global heap data to disk.
// Should be called before closing the file.
func (ghw *globalHeapWriter) Flush() error {
	if err := ghw.flushHeap(ghw.dimensionHeap); err != nil {
		return err
	}
	return ghw.flushCurrentHeap()
}
