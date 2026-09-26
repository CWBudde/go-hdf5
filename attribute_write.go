package hdf5

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"unsafe"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/structures"
	"github.com/cwbudde/go-hdf5/internal/writer"
)

// Attribute storage threshold.
const (
	// MaxCompactAttributes is the threshold for transitioning to dense storage.
	// A group keeps up to 8 attributes compact (object header messages);
	// adding the 9th moves them to dense storage (Fractal Heap + B-tree).
	// Datasets keep all their attributes compact (see WriteAttribute).
	MaxCompactAttributes = 8

	// unlimitedCompactAttributes is the compact limit for datasets.
	unlimitedCompactAttributes = math.MaxInt
)

// ErrCreationOrderIndexNotSupported is returned when an attribute would
// have to be added to, changed in or deleted from dense attribute storage
// that indexes attribute creation order (netCDF-C files, h5py with
// track_order=True, at more than 8 attributes): go-hdf5 cannot update that
// index (a v2 B-tree of type 9) yet. Compact attributes of such objects can
// be written.
var ErrCreationOrderIndexNotSupported = errors.New("dense attribute storage with a creation order index is not supported")

// WriteAttribute writes an attribute to a dataset.
//
// Storage: attributes are stored compactly (object header messages),
// whatever their number. Only datasets that already use dense storage
// (Fractal Heap + B-tree v2, e.g. written by libhdf5) get new attributes
// there.
//
// Supported value types:
//   - Scalars: int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64
//   - Arrays: []int32, []float64, etc. (1D arrays only)
//   - Strings: string (fixed-length, converted to byte array)
//
// Parameters:
//   - name: Attribute name (ASCII, no null bytes)
//   - value: Attribute value (Go scalar, slice, or string)
//
// Returns:
//   - error: If attribute cannot be written
//
// Example:
//
//	ds, _ := fw.CreateDataset("/temperature", Float64, []uint64{10})
//	ds.WriteAttribute("units", "Celsius")
//	ds.WriteAttribute("sensor_id", int32(42))
//	ds.WriteAttribute("calibration", []float64{1.0, 0.0})
//
// Limitations:
//   - No variable-length strings
//   - No compound types
//   - Attributes cannot be modified after creation (write-once)
//   - No attribute deletion
func (ds *DatasetWriter) WriteAttribute(name string, value interface{}) error {
	value, err := ds.fileWriter.prepareAttributeValue(dimScaleStringAttribute(name, value))
	if err != nil {
		return fmt.Errorf("attribute %q: %w", name, err)
	}

	// For datasets opened with OpenForWrite (or resized), use the cached
	// object header, which the attribute functions keep up to date.
	if ds.objectHeader != nil {
		return writeAttributeToHeader(ds.fileWriter, ds.address, ds.objectHeader, name, value, unlimitedCompactAttributes)
	}

	// For datasets created in this session, read object header fresh
	return writeAttribute(ds.fileWriter, ds.address, name, value, unlimitedCompactAttributes)
}

// DeleteAttribute removes an attribute by name from the dataset.
//
// This method supports both compact and dense attribute storage:
// - Compact storage (0-7 attributes): Removes message from object header
// - Dense storage (8+ attributes): Removes from B-tree and fractal heap
//
// Parameters:
//   - name: Attribute name to delete
//
// Returns:
//   - error: If attribute not found or deletion fails
//
// Reference: H5Adelete.c - H5A__delete(), H5Adense.c - H5A__dense_remove().
func (ds *DatasetWriter) DeleteAttribute(name string) error {
	// For datasets opened with OpenForWrite (or resized), use the cached
	// object header.
	if ds.objectHeader != nil {
		return deleteAttributeFromHeader(ds.fileWriter, ds.address, ds.objectHeader, name)
	}

	// For datasets created in this session, read object header fresh
	return deleteAttribute(ds.fileWriter, ds.address, name)
}

// RebalanceAttributeBTree manually triggers B-tree rebalancing for this dataset's dense attribute storage.
//
// Use this when:
//   - You know this specific dataset needs rebalancing
//   - More efficient than RebalanceAllBTrees() for targeted optimization
//   - After batch deletions with rebalancing disabled
//
// Performance (for current MVP with single-leaf B-trees):
//   - Instant (< 1ms) - no-op for single-leaf trees
//
// Future (when multi-level B-trees implemented):
//   - Small (<1000 attrs): <10ms
//   - Medium (1000-10000 attrs): 10-100ms
//   - Large (10000+ attrs): 100ms-1s
//
// Returns:
//   - error: if dataset doesn't use dense storage or rebalancing fails
//
// Example:
//
//	fw.DisableRebalancing()
//	for i := 0; i < 1000; i++ {
//	    ds.DeleteAttribute(fmt.Sprintf("temp_%d", i))  // Fast deletions
//	}
//	ds.RebalanceAttributeBTree()  // Rebalance this dataset only
//
// Reference: Similar to per-object rebalancing in HDF5 (hypothetical - not exposed in C API).
func (ds *DatasetWriter) RebalanceAttributeBTree() error {
	sb := ds.fileWriter.file.Superblock()
	reader := ds.fileWriter.writer.Reader()
	oh := ds.objectHeader
	if oh == nil {
		var err error
		if oh, err = core.ReadObjectHeader(reader, ds.address, sb); err != nil {
			return fmt.Errorf("failed to read object header: %w", err)
		}
	}
	attrInfo, dense, err := denseAttributeInfo(oh, sb)
	if err != nil {
		return err
	}
	if !dense {
		// No dense storage - nothing to rebalance
		return nil
	}

	// Load and rebalance B-tree
	btree := structures.NewWritableBTreeV2(0)
	err = btree.LoadFromFile(reader, attrInfo.BTreeNameIndexAddr, sb)
	if err != nil {
		return fmt.Errorf("failed to load B-tree: %w", err)
	}

	err = btree.RebalanceAll()
	if err != nil {
		return fmt.Errorf("failed to rebalance B-tree: %w", err)
	}

	// For MVP: RebalanceAll() is a no-op
	// Future: Write modified tree back to disk

	return nil
}

// writeAttribute is the internal implementation for writing attributes.
//
// Storage strategy:
// - Up to compactLimit attributes: Compact storage (object header messages)
// - More: Dense storage (Fractal Heap + B-tree v2)
//
// Groups use MaxCompactAttributes, datasets unlimitedCompactAttributes:
// netCDF variables rarely have more than 8 attributes, and libmysofa cannot
// read dense attributes other than scalar strings (e.g. DIMENSION_LIST).
// Objects that already use dense storage keep using it.
//
// Automatic transition:
// - When adding attribute compactLimit+1, all attributes are migrated to dense storage
// - Compact attribute messages are removed from object header
// - Attribute Info Message is added to object header
//
// For MVP:
// - Transition is one-way (compact → dense only, no dense → compact)
// - No attribute deletion support
//
// Reference: H5Aint.c - H5A__dense_create().
func writeAttribute(fw *FileWriter, objectAddr uint64, name string, value interface{}, compactLimit int) error {
	oh, err := core.ReadObjectHeader(fw.writer.Reader(), objectAddr, fw.file.Superblock())
	if err != nil {
		return fmt.Errorf("failed to read object header: %w", err)
	}
	return writeAttributeToHeader(fw, objectAddr, oh, name, value, compactLimit)
}

// writeAttributeToHeader writes an attribute of the object at objectAddr
// whose current object header is oh (see writeAttribute). oh is updated in
// place, so a cached header (DatasetWriter) stays current.
func writeAttributeToHeader(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader,
	name string, value interface{}, compactLimit int,
) error {
	sb := fw.file.Superblock()
	attrInfo, dense, err := denseAttributeInfo(oh, sb)
	if err != nil {
		return err
	}
	if dense {
		// Already using dense storage → add to dense
		attrMsg, err := encodeAttribute(name, value, sb)
		if err != nil {
			return err
		}
		return updateDenseAttributes(fw, objectAddr, oh, attrInfo, name, attrMsg)
	}

	// Count existing attributes
	compactCount := 0
	for _, msg := range oh.Messages {
		if msg.Type == core.MsgAttribute {
			compactCount++
		}
	}

	if compactCount < compactLimit || hasCompactAttribute(oh, name, sb) {
		// Still compact, or replacing an existing compact attribute
		return writeCompactAttribute(fw, objectAddr, oh, name, value, sb)
	}

	// Transition needed → migrate to dense
	return transitionToDenseAttributes(fw, objectAddr, oh, name, value, sb)
}

// denseAttributeInfo returns the Attribute Info message of oh and whether
// it points to dense attribute storage (see hasDenseAttributeStorage).
func denseAttributeInfo(oh *core.ObjectHeader, sb *core.Superblock) (*core.AttributeInfoMessage, bool, error) {
	for _, msg := range oh.Messages {
		if msg.Type != core.MsgAttributeInfo {
			continue
		}
		info, err := core.ParseAttributeInfoMessage(msg.Data, sb)
		if err != nil {
			return nil, false, fmt.Errorf("failed to parse attribute info message: %w", err)
		}
		return info, isDefinedAddress(info.FractalHeapAddr), nil
	}
	return nil, false, nil
}

// encodeAttribute encodes the attribute message of an attribute name with
// value.
func encodeAttribute(name string, value any, sb *core.Superblock) ([]byte, error) {
	datatype, dataspace, err := inferDatatypeFromValue(value)
	if err != nil {
		return nil, fmt.Errorf("failed to infer datatype: %w", err)
	}
	data, err := encodeAttributeValue(value)
	if err != nil {
		return nil, fmt.Errorf("failed to encode value: %w", err)
	}
	attrMsg, err := core.EncodeAttributeFromStruct(&core.Attribute{Name: name, Datatype: datatype, Dataspace: dataspace, Data: data}, sb)
	if err != nil {
		return nil, fmt.Errorf("failed to encode attribute: %w", err)
	}
	return attrMsg, nil
}

// hasDenseAttributeStorage reports whether the object header points to
// dense attribute storage. libhdf5 may write an Attribute Info message with
// undefined heap/B-tree addresses (e.g. with libver="latest"); such objects
// still store their attributes compactly.
func hasDenseAttributeStorage(oh *core.ObjectHeader, sb *core.Superblock) bool {
	for _, msg := range oh.Messages {
		if msg.Type != core.MsgAttributeInfo {
			continue
		}
		info, err := core.ParseAttributeInfoMessage(msg.Data, sb)
		if err != nil {
			return true // let the dense path report the problem
		}
		if isDefinedAddress(info.FractalHeapAddr) {
			return true
		}
	}
	return false
}

// isDefinedAddress reports whether addr is neither 0 nor the undefined
// address (all bits set).
func isDefinedAddress(addr uint64) bool {
	return addr != 0 && addr != ^uint64(0)
}

// hasCompactAttribute reports whether the object header holds a compact
// attribute message with the given name.
func hasCompactAttribute(oh *core.ObjectHeader, name string, sb *core.Superblock) bool {
	for _, msg := range oh.Messages {
		if msg.Type != core.MsgAttribute {
			continue
		}
		if attr, err := core.ParseAttributeMessage(msg.Data, sb.Endianness); err == nil && attr.Name == name {
			return true
		}
	}
	return false
}

// writeCompactAttribute writes attribute to object header (compact storage).
// This is the Phase 1 code, extracted into separate function.
func writeCompactAttribute(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader,
	name string, value any, sb *core.Superblock,
) error {
	// 1. Infer datatype and encode attribute
	datatype, dataspace, err := inferDatatypeFromValue(value)
	if err != nil {
		return fmt.Errorf("failed to infer datatype: %w", err)
	}

	data, err := encodeAttributeValue(value)
	if err != nil {
		return fmt.Errorf("failed to encode value: %w", err)
	}

	attr := &core.Attribute{
		Name:      name,
		Datatype:  datatype,
		Dataspace: dataspace,
		Data:      data,
	}

	// 2. Check if attribute exists (for upsert semantics)
	// If exists → modify (replace data)
	// If not exists → create (add new message)
	existingIndex := -1
	for i, msg := range oh.Messages {
		if msg.Type == core.MsgAttribute {
			existingAttr, parseErr := core.ParseAttributeMessage(msg.Data, sb.Endianness)
			if parseErr == nil && existingAttr.Name == name {
				existingIndex = i
				break
			}
		}
	}

	// 3. Encode attribute message
	attrMsg, err := core.EncodeAttributeFromStruct(attr, sb)
	if err != nil {
		return fmt.Errorf("failed to encode attribute message: %w", err)
	}

	// 4. Upsert logic: modify if exists, add if not exists
	err = upsertAttributeMessage(fw, objectAddr, oh, existingIndex, attrMsg, name, value, sb)
	if err != nil {
		return err
	}

	// 5. Write updated header back to disk
	err = core.WriteObjectHeader(fw.writer, objectAddr, oh, sb)
	if err != nil {
		return fmt.Errorf("failed to write object header: %w", err)
	}

	return nil
}

// upsertAttributeMessage handles the upsert logic for attribute messages in compact storage.
// If attribute exists (existingIndex >= 0), it replaces the message data.
// If attribute doesn't exist (existingIndex < 0), it adds a new message.
// If object header is full, it triggers transition to dense storage.
func upsertAttributeMessage(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader,
	existingIndex int, attrMsg []byte, name string, value interface{}, sb *core.Superblock,
) error {
	if existingIndex >= 0 {
		// Attribute exists → Replace (upsert semantics)
		oh.Messages[existingIndex].Data = attrMsg
		return nil
	}

	// Attribute doesn't exist → Add new message
	err := core.AddMessageToObjectHeader(oh, core.MsgAttribute, attrMsg)
	if err != nil {
		// If object header is full, transition to dense storage
		if strings.Contains(err.Error(), "object header full") {
			return transitionToDenseAttributes(fw, objectAddr, oh, name, value, sb)
		}
		return fmt.Errorf("failed to add message to header: %w", err)
	}

	return nil
}

// deleteAttribute is the internal implementation for deleting attributes.
//
// Handles both compact and dense storage:
// - Compact: Removes attribute message from object header
// - Dense: Removes from B-tree and fractal heap
//
// Reference: H5Adelete.c - H5A__delete().
func deleteAttribute(fw *FileWriter, objectAddr uint64, name string) error {
	oh, err := core.ReadObjectHeader(fw.writer.Reader(), objectAddr, fw.file.Superblock())
	if err != nil {
		return fmt.Errorf("failed to read object header: %w", err)
	}
	return deleteAttributeFromHeader(fw, objectAddr, oh, name)
}

// deleteAttributeFromHeader deletes an attribute of the object at
// objectAddr whose current object header is oh, updating oh in place.
func deleteAttributeFromHeader(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader, name string) error {
	sb := fw.file.Superblock()
	// An Attribute Info message without dense storage only tracks
	// creation order.
	attrInfo, dense, err := denseAttributeInfo(oh, sb)
	if err != nil {
		return err
	}
	if dense {
		return updateDenseAttributes(fw, objectAddr, oh, attrInfo, name, nil)
	}
	return deleteCompactAttributeFromHeader(fw, objectAddr, oh, name, sb)
}

// deleteCompactAttributeFromHeader deletes attribute from object header.
//
// Implementation note:
// This uses the existing object header write infrastructure to persist
// the deletion to disk.
//
// Reference: H5Adelete.c - H5A__delete(), H5O.c - H5O_msg_remove().
func deleteCompactAttributeFromHeader(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader,
	name string, sb *core.Superblock,
) error {
	// Find and remove attribute message
	msgIndex := -1
	for i, msg := range oh.Messages {
		if msg.Type == core.MsgAttribute {
			attr, parseErr := core.ParseAttributeMessage(msg.Data, sb.Endianness)
			if parseErr == nil && attr.Name == name {
				msgIndex = i
				break
			}
		}
	}

	if msgIndex == -1 {
		return fmt.Errorf("attribute %q not found", name)
	}

	// Remove message (direct removal - clean approach)
	oh.Messages = append(oh.Messages[:msgIndex], oh.Messages[msgIndex+1:]...)

	// Write back object header to disk
	err := core.WriteObjectHeader(fw.writer, objectAddr, oh, sb)
	if err != nil {
		return fmt.Errorf("failed to write object header after deletion: %w", err)
	}

	return nil
}

// updateDenseAttributes adds or replaces (attrMsg set) or deletes (attrMsg
// nil) the attribute name in the dense storage attrInfo of the object at
// objectAddr, whose current object header is oh. Storage in the layout this
// library writes is edited in place; other storage, e.g. written by the
// HDF5 C library, is read completely and rewritten (see
// rebuildDenseAttributes).
//
// Reference: H5Adense.c - H5A__dense_insert(), H5A__dense_write(),
// H5A__dense_remove().
func updateDenseAttributes(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader,
	attrInfo *core.AttributeInfoMessage, name string, attrMsg []byte,
) error {
	if attrInfo.Flags&core.AttributeInfoIndexCreationOrder != 0 {
		return fmt.Errorf("attribute %q: %w", name, ErrCreationOrderIndexNotSupported)
	}
	editable, err := fw.denseAttributesEditable(attrInfo, len(attrMsg))
	if err != nil {
		return err
	}
	if !editable {
		return fw.rebuildDenseAttributes(objectAddr, oh, attrInfo, name, attrMsg)
	}
	return updateDenseAttributesInPlace(fw, objectAddr, oh, attrInfo, name, attrMsg)
}

// denseAttributesEditable reports whether the dense attribute storage
// attrInfo has the layout this library writes, so that an attribute message
// of msgSize bytes can be added in place: a fractal heap whose root is a
// single direct block, without a free-space manager, huge objects or I/O
// filters, and a name index of depth 0 (as denseLinksEditable).
//
// Storage written by the HDF5 C library does not qualify: its heaps track
// free space in a free-space manager, and the next insert would overwrite
// existing objects.
func (fw *FileWriter) denseAttributesEditable(attrInfo *core.AttributeInfoMessage, msgSize int) (bool, error) {
	sb := fw.file.Superblock()
	r := fw.writer.Reader()
	fh, err := structures.OpenFractalHeap(r, attrInfo.FractalHeapAddr, sb.LengthSize, sb.OffsetSize, sb.Endianness)
	if err != nil {
		return false, fmt.Errorf("open attribute heap: %w", err)
	}
	h := fh.Header
	if h.CurrentRowCount != 0 || h.IOFiltersLen != 0 || h.HugeObjCount != 0 ||
		isDefinedAddress(h.FreeSpaceSectionAddr) ||
		uint64(msgSize) > uint64(h.MaxManagedObjSize) { //nolint:gosec // G115: msgSize is a length
		return false, nil
	}
	if !isDefinedAddress(attrInfo.BTreeNameIndexAddr) {
		return false, nil
	}
	info, err := core.ReadBTreeV2Info(r, attrInfo.BTreeNameIndexAddr, sb)
	if err != nil {
		return false, fmt.Errorf("read attribute name index header: %w", err)
	}
	return info.Type == structures.BTreeV2TypeAttrNameIndex && info.RecordSize == attrNameIndexRecordSize && info.Depth == 0, nil
}

// attrNameIndexRecordSize is the size of an attribute name index (v2 B-tree
// type 8) record: heap ID (8), message flags (1), creation order (4), name
// hash (4).
const attrNameIndexRecordSize = 17

// updateDenseAttributesInPlace is updateDenseAttributes for storage that
// denseAttributesEditable accepts.
func updateDenseAttributesInPlace(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader,
	attrInfo *core.AttributeInfoMessage, name string, attrMsg []byte,
) error {
	sb := fw.file.Superblock()
	heap := structures.NewGrowableFractalHeap(structures.AttributeHeapStartBlockSize) // size comes from the file
	if err := heap.LoadFromFile(fw.writer.Reader(), attrInfo.FractalHeapAddr, sb); err != nil {
		return fmt.Errorf("failed to load fractal heap: %w", err)
	}
	btree := structures.NewWritableBTreeV2(0)
	if err := btree.LoadFromFile(fw.writer.Reader(), attrInfo.BTreeNameIndexAddr, sb); err != nil {
		return fmt.Errorf("failed to load B-tree: %w", err)
	}

	_, exists := btree.SearchRecord(name)
	switch {
	case attrMsg == nil:
		if err := core.DeleteDenseAttribute(heap, btree, name, fw.RebalancingEnabled()); err != nil {
			return fmt.Errorf("failed to delete dense attribute: %w", err)
		}
	case exists:
		if err := core.ModifyDenseAttribute(heap, btree, name, &core.Attribute{Name: name, Data: attrMsg}); err != nil {
			return fmt.Errorf("failed to modify existing dense attribute: %w", err)
		}
	default:
		order, err := claimDenseCreationIndex(fw, objectAddr, oh, attrInfo, sb)
		if err != nil {
			return err
		}
		heapIDBytes, err := heap.InsertObject(attrMsg)
		if err != nil {
			return fmt.Errorf("failed to insert into heap: %w", err)
		}
		if len(heapIDBytes) != 8 {
			return fmt.Errorf("unexpected heap ID length: %d bytes", len(heapIDBytes))
		}
		if err := btree.InsertAttributeRecord(name, binary.LittleEndian.Uint64(heapIDBytes), order); err != nil {
			return fmt.Errorf("failed to insert into B-tree: %w", err)
		}
	}

	// Write the heap and the B-tree back where they were loaded from (they
	// move when they grow; their header addresses stay).
	if err := heap.WriteAtWithAllocator(fw.writer, fw.writer.Allocator(), sb); err != nil {
		return fmt.Errorf("failed to write updated heap: %w", err)
	}
	if err := btree.WriteAtWithAllocator(fw.writer, fw.writer.Allocator(), sb); err != nil {
		return fmt.Errorf("failed to write updated B-tree: %w", err)
	}
	return nil
}

// denseAttributeMessage is an attribute read from dense storage.
type denseAttributeMessage struct {
	name  string
	msg   []byte // encoded attribute message
	order uint16 // creation order (0 when not tracked)
}

// readDenseAttributeMessages reads all attribute messages of the dense
// storage attrInfo with the reader, which handles every layout the HDF5 C
// library writes.
func (fw *FileWriter) readDenseAttributeMessages(attrInfo *core.AttributeInfoMessage) ([]denseAttributeMessage, error) {
	sb := fw.file.Superblock()
	r := fw.writer.Reader()
	fh, err := structures.OpenFractalHeap(r, attrInfo.FractalHeapAddr, sb.LengthSize, sb.OffsetSize, sb.Endianness)
	if err != nil {
		return nil, fmt.Errorf("open attribute heap: %w", err)
	}
	info, records, err := core.ReadBTreeV2Records(r, attrInfo.BTreeNameIndexAddr, sb)
	if err != nil {
		return nil, fmt.Errorf("read attribute name index: %w", err)
	}
	if info.Type != structures.BTreeV2TypeAttrNameIndex || info.RecordSize != attrNameIndexRecordSize {
		return nil, fmt.Errorf("attribute name index has type %d and %d-byte records", info.Type, info.RecordSize)
	}
	attrs := make([]denseAttributeMessage, 0, len(records))
	for _, rec := range records {
		if rec[8] != 0 {
			return nil, fmt.Errorf("dense attribute with message flags 0x%02x (shared) is not supported", rec[8])
		}
		msg, err := fh.ReadObjectSpecCompliant(rec[:8])
		if err != nil {
			return nil, fmt.Errorf("read attribute %x from heap: %w", rec[:8], err)
		}
		attr, err := core.ParseAttributeMessage(msg, sb.Endianness)
		if err != nil {
			return nil, fmt.Errorf("parse dense attribute: %w", err)
		}
		attrs = append(attrs, denseAttributeMessage{
			name:  attr.Name,
			msg:   msg,
			order: uint16(binary.LittleEndian.Uint32(rec[9:13])), //nolint:gosec // G115: 16-bit creation indexes
		})
	}
	return attrs, nil
}

// addDenseAttributeMessages adds attrs to daw with the attribute name added,
// replaced (attrMsg set) or deleted (attrMsg nil).
func addDenseAttributeMessages(daw *writer.DenseAttributeWriter, attrs []denseAttributeMessage, name string, attrMsg []byte) error {
	found := false
	for _, a := range attrs {
		msg := a.msg
		if a.name == name {
			found = true
			if attrMsg == nil {
				continue // deleted
			}
			msg = attrMsg
		}
		if err := daw.AddEncodedAttributeWithCreationOrder(a.name, msg, a.order); err != nil {
			return fmt.Errorf("attribute %q: %w", a.name, err)
		}
	}
	switch {
	case !found && attrMsg == nil:
		return fmt.Errorf("attribute %q not found in dense storage", name)
	case !found:
		if err := daw.AddEncodedAttribute(name, attrMsg); err != nil {
			return fmt.Errorf("attribute %q: %w", name, err)
		}
	}
	return nil
}

// rebuildDenseAttributes writes new dense storage for the object at
// objectAddr holding the attributes of attrInfo, with name added, replaced
// (attrMsg set) or deleted (attrMsg nil), and points the Attribute Info
// message of oh to it, rewriting the header. Attributes keep their bytes
// and creation order; the old storage is left unused.
func (fw *FileWriter) rebuildDenseAttributes(objectAddr uint64, oh *core.ObjectHeader,
	attrInfo *core.AttributeInfoMessage, name string, attrMsg []byte,
) error {
	sb := fw.file.Superblock()
	attrs, err := fw.readDenseAttributeMessages(attrInfo)
	if err != nil {
		return err
	}
	if attrMsg == nil && len(attrs) == 1 && attrs[0].name == name {
		// No attributes remain: drop the dense storage, as libhdf5 does,
		// keeping the Attribute Info message (and the creation order
		// tracked in it) without it.
		attrInfo.FractalHeapAddr = undefinedAddress
		attrInfo.BTreeNameIndexAddr = undefinedAddress
		return writeAttributeInfo(fw, objectAddr, oh, attrInfo)
	}
	daw := writer.NewDenseAttributeWriter(objectAddr)
	if attrInfo.Flags&core.AttributeInfoTrackCreationOrder != 0 {
		daw.TrackCreationOrder(uint16(min(attrInfo.MaxCreationIndex, 0xFFFF)))
	}
	if err := addDenseAttributeMessages(daw, attrs, name, attrMsg); err != nil {
		return err
	}

	newInfo, err := daw.WriteToFile(fw.writer, fw.writer.Allocator(), sb)
	if err != nil {
		return fmt.Errorf("failed to write dense storage: %w", err)
	}
	return writeAttributeInfo(fw, objectAddr, oh, newInfo)
}

// writeAttributeInfo replaces the Attribute Info message of oh, the object
// header of the object at objectAddr, with attrInfo and rewrites the header.
func writeAttributeInfo(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader, attrInfo *core.AttributeInfoMessage) error {
	sb := fw.file.Superblock()
	data, err := core.EncodeAttributeInfoMessage(attrInfo, sb)
	if err != nil {
		return fmt.Errorf("failed to encode attribute info: %w", err)
	}
	for _, msg := range oh.Messages {
		if msg.Type == core.MsgAttributeInfo {
			msg.Data = data
		}
	}
	if err := core.WriteObjectHeader(fw.writer, objectAddr, oh, sb); err != nil {
		return fmt.Errorf("failed to write object header: %w", err)
	}
	return nil
}

// transitionToDenseAttributes migrates all compact attributes to dense storage.
//
// Process:
// 1. Read all compact attributes from object header
// 2. Create DenseAttributeWriter
// 3. Add all existing attributes to dense storage
// 4. Add new attribute to dense storage
// 5. Write dense storage (heap + B-tree)
// 6. Get Attribute Info Message
// 7. Remove all compact attribute messages from object header
// 8. Add Attribute Info Message to object header
// 9. Write updated object header
//
// Reference: H5Aint.c - H5A__dense_create().
//
//nolint:gocyclo,cyclop // Complex but necessary business logic for compact→dense transition
func transitionToDenseAttributes(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader,
	name string, value interface{}, sb *core.Superblock,
) error {
	// 1. Add all existing compact attributes to a DenseAttributeWriter,
	// keeping their creation order.
	daw, tracked, err := denseWriterWithCompactAttributes(objectAddr, oh, sb)
	if err != nil {
		return fmt.Errorf("attribute %q: %w", name, err)
	}

	// 2. Infer datatype and encode new attribute
	datatype, dataspace, err := inferDatatypeFromValue(value)
	if err != nil {
		return fmt.Errorf("failed to infer datatype: %w", err)
	}

	data, err := encodeAttributeValue(value)
	if err != nil {
		return fmt.Errorf("failed to encode value: %w", err)
	}

	newAttr := &core.Attribute{
		Name:      name,
		Datatype:  datatype,
		Dataspace: dataspace,
		Data:      data,
	}

	// 3. Add new attribute
	err = daw.AddAttribute(newAttr, sb)
	if err != nil {
		return fmt.Errorf("failed to add new attribute: %w", err)
	}

	// 6. Remove compact attributes (and any Attribute Info message without
	// dense storage, as libhdf5 writes) from the object header
	var newMessages []*core.HeaderMessage
	for _, msg := range oh.Messages {
		if msg.Type != core.MsgAttribute && msg.Type != core.MsgAttributeInfo {
			newMessages = append(newMessages, msg)
		}
	}
	oh.Messages = newMessages

	// 7. Calculate object header size (without AttrInfo message yet)
	// to determine where dense storage should be allocated
	ohWriter := &core.ObjectHeaderWriter{
		Version:  oh.Version,
		Flags:    oh.Flags,
		Messages: make([]core.MessageWriter, len(oh.Messages)),
	}
	for i, msg := range oh.Messages {
		ohWriter.Messages[i] = core.MessageWriter{
			Type: msg.Type,
			Data: msg.Data,
		}
	}

	// Add temporary AttrInfo message to calculate size
	// Use REAL size (2 + offsetSize*2, + 2 if tracked) even though addresses are unknown
	tempAttrInfo := &core.AttributeInfoMessage{
		Version:            0,
		Flags:              0,
		FractalHeapAddr:    0,
		BTreeNameIndexAddr: 0,
	}
	if tracked {
		tempAttrInfo.Flags = core.AttributeInfoTrackCreationOrder
	}
	tempAttrInfoMsg, err := core.EncodeAttributeInfoMessage(tempAttrInfo, sb)
	if err != nil {
		return fmt.Errorf("failed to encode temp attribute info: %w", err)
	}

	ohWriter.Messages = append(ohWriter.Messages, core.MessageWriter{
		Type: core.MsgAttributeInfo,
		Data: tempAttrInfoMsg,
	})

	objectHeaderSize := ohWriter.Size()
	objectHeaderEnd := objectAddr + objectHeaderSize

	// 8. Update allocator to ensure dense storage allocated AFTER object header
	allocator := fw.writer.Allocator()
	if allocator.EndOfFile() < objectHeaderEnd {
		bytesToAdvance := objectHeaderEnd - allocator.EndOfFile()
		_, err = allocator.Allocate(bytesToAdvance)
		if err != nil {
			return fmt.Errorf("failed to advance allocator past object header: %w", err)
		}
	}

	// 9. Write dense storage - allocator will place it AFTER object header
	attrInfo, err := daw.WriteToFile(fw.writer, allocator, sb)
	if err != nil {
		return fmt.Errorf("failed to write dense storage: %w", err)
	}

	// 10. NOW add AttributeInfo message with REAL addresses to object header
	attrInfoMsg, err := core.EncodeAttributeInfoMessage(attrInfo, sb)
	if err != nil {
		return fmt.Errorf("failed to encode attribute info: %w", err)
	}

	err = core.AddMessageToObjectHeader(oh, core.MsgAttributeInfo, attrInfoMsg)
	if err != nil {
		return fmt.Errorf("failed to add attribute info message: %w", err)
	}

	// 11. Write object header with REAL addresses (ONE TIME!)
	err = core.WriteObjectHeader(fw.writer, objectAddr, oh, sb)
	if err != nil {
		return fmt.Errorf("failed to write object header: %w", err)
	}

	// 13. CRITICAL: Flush buffered writes to disk!
	// Dense storage was just created at new addresses.
	// Subsequent attributes will try to load from these addresses.
	// If data isn't flushed, they'll read uninitialized memory!
	err = fw.writer.Flush()
	if err != nil {
		return fmt.Errorf("failed to flush after transition: %w", err)
	}

	return nil
}

// denseWriterWithCompactAttributes returns a DenseAttributeWriter holding
// the compact attributes of oh and whether oh tracks attribute creation
// order. If it does, the attributes keep their creation indexes and new ones
// continue after the largest index used (or recorded in the Attribute Info
// message). It fails with ErrCreationOrderIndexNotSupported if the object
// indexes creation order.
func denseWriterWithCompactAttributes(objectAddr uint64, oh *core.ObjectHeader, sb *core.Superblock) (*writer.DenseAttributeWriter, bool, error) {
	var attrs []denseAttributeMessage
	next := uint16(0)
	for _, msg := range oh.Messages {
		switch msg.Type {
		case core.MsgAttribute:
			attr, err := core.ParseAttributeMessage(msg.Data, sb.Endianness)
			if err != nil {
				return nil, false, fmt.Errorf("failed to parse existing attribute: %w", err)
			}
			attrs = append(attrs, denseAttributeMessage{name: attr.Name, msg: msg.Data, order: msg.CrtIdx})
		case core.MsgAttributeInfo:
			info, err := core.ParseAttributeInfoMessage(msg.Data, sb)
			if err != nil {
				return nil, false, fmt.Errorf("failed to parse attribute info: %w", err)
			}
			if info.Flags&core.AttributeInfoIndexCreationOrder != 0 {
				return nil, false, ErrCreationOrderIndexNotSupported
			}
			if info.Flags&core.AttributeInfoTrackCreationOrder != 0 {
				next = uint16(info.MaxCreationIndex) //nolint:gosec // G115: 16-bit on disk
			}
		}
	}

	daw := writer.NewDenseAttributeWriter(objectAddr)
	tracked := oh.Flags&core.ObjectHeaderAttrCreationOrderTracked != 0
	if tracked {
		daw.TrackCreationOrder(next) // AddAttributeWithCreationOrder raises it past used indexes
	}
	// The messages move as they are, keeping datatypes this library cannot
	// encode.
	for _, a := range attrs {
		if err := daw.AddEncodedAttributeWithCreationOrder(a.name, a.msg, a.order); err != nil {
			return nil, false, fmt.Errorf("failed to add existing attribute: %w", err)
		}
	}
	return daw, tracked, nil
}

// claimDenseCreationIndex returns the creation order of a new attribute in
// the dense storage described by attrInfo and, if the object tracks it,
// advances the maximum creation index in its Attribute Info message
// (rewriting the object header). Untracked storage records order 0.
func claimDenseCreationIndex(fw *FileWriter, objectAddr uint64, oh *core.ObjectHeader,
	attrInfo *core.AttributeInfoMessage, sb *core.Superblock,
) (uint16, error) {
	if attrInfo.Flags&core.AttributeInfoTrackCreationOrder == 0 {
		return 0, nil
	}
	if attrInfo.MaxCreationIndex >= 0xFFFF {
		return 0, fmt.Errorf("maximum attribute creation index reached")
	}
	order := uint16(attrInfo.MaxCreationIndex)
	attrInfo.MaxCreationIndex++
	data, err := core.EncodeAttributeInfoMessage(attrInfo, sb)
	if err != nil {
		return 0, fmt.Errorf("failed to encode attribute info: %w", err)
	}
	for _, msg := range oh.Messages {
		if msg.Type == core.MsgAttributeInfo {
			msg.Data = data
			if err := core.WriteObjectHeader(fw.writer, objectAddr, oh, sb); err != nil {
				return 0, fmt.Errorf("failed to write object header: %w", err)
			}
			return order, nil
		}
	}
	return 0, fmt.Errorf("attribute info message not found")
}

// inferDatatypeFromValue infers HDF5 datatype and dimensions from a Go value.
// Returns datatype message, dataspace message, and error.
func inferDatatypeFromValue(value interface{}) (*core.DatatypeMessage, *core.DataspaceMessage, error) {
	if ev, ok := value.(*encodedAttributeValue); ok {
		return ev.datatype, ev.dataspace, nil
	}

	v := reflect.ValueOf(value)

	// Handle scalar types
	if !v.IsValid() {
		return nil, nil, fmt.Errorf("value is nil or invalid")
	}

	switch v.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return inferSignedInt(v)
	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return inferUnsignedInt(v)
	case reflect.Float32, reflect.Float64:
		return inferFloat(v)
	case reflect.String:
		return inferString(v)
	case reflect.Slice:
		return inferSlice(v)
	default:
		return nil, nil, fmt.Errorf("unsupported value type: %s", v.Kind())
	}
}

// inferSignedInt infers datatype for signed integers.
func inferSignedInt(v reflect.Value) (*core.DatatypeMessage, *core.DataspaceMessage, error) {
	var size uint32
	switch v.Kind() {
	case reflect.Int8:
		size = 1
	case reflect.Int16:
		size = 2
	case reflect.Int32:
		size = 4
	case reflect.Int64:
		size = 8
	default:
		return nil, nil, fmt.Errorf("not a signed integer type")
	}

	dt := &core.DatatypeMessage{
		Class:         core.DatatypeFixed,
		Size:          size,
		ClassBitField: 0x08, // Bit 3 set for signed integers
	}

	ds := &core.DataspaceMessage{
		Dimensions: []uint64{1}, // Scalar (HDF5 uses [1] for scalars)
		MaxDims:    nil,
	}

	return dt, ds, nil
}

// inferUnsignedInt infers datatype for unsigned integers.
func inferUnsignedInt(v reflect.Value) (*core.DatatypeMessage, *core.DataspaceMessage, error) {
	var size uint32
	switch v.Kind() {
	case reflect.Uint8:
		size = 1
	case reflect.Uint16:
		size = 2
	case reflect.Uint32:
		size = 4
	case reflect.Uint64:
		size = 8
	default:
		return nil, nil, fmt.Errorf("not an unsigned integer type")
	}

	dt := &core.DatatypeMessage{
		Class:         core.DatatypeFixed,
		Size:          size,
		ClassBitField: 0, // Bit 3 clear for unsigned integers
	}

	ds := &core.DataspaceMessage{
		Dimensions: []uint64{1}, // Scalar
		MaxDims:    nil,
	}

	return dt, ds, nil
}

// inferFloat infers datatype for floating point numbers.
func inferFloat(v reflect.Value) (*core.DatatypeMessage, *core.DataspaceMessage, error) {
	var size uint32
	switch v.Kind() {
	case reflect.Float32:
		size = 4
	case reflect.Float64:
		size = 8
	default:
		return nil, nil, fmt.Errorf("not a float type")
	}

	dt := &core.DatatypeMessage{
		Class:         core.DatatypeFloat,
		Size:          size,
		ClassBitField: 0, // Little-endian
	}

	ds := &core.DataspaceMessage{
		Dimensions: []uint64{1}, // Scalar
		MaxDims:    nil,
	}

	return dt, ds, nil
}

// inferString infers datatype for strings.
//
// A Go string is written like a netCDF-C text attribute (NC_CHAR): a
// fixed-length string of exactly len(s) bytes (at least 1), NULLTERM
// padding, ASCII character set, in a scalar dataspace. netCDF-C reads a
// one-element simple dataspace as an NC_STRING array instead, which e.g. the
// SOFA Toolbox rejects. Like netCDF-C, non-ASCII content keeps its UTF-8
// bytes under the ASCII character set: libmysofa does not read attributes
// marked UTF-8 from dense storage.
func inferString(v reflect.Value) (*core.DatatypeMessage, *core.DataspaceMessage, error) {
	str := v.String()
	size := uint32(max(len(str), 1)) //nolint:gosec // Safe: string length fits in uint32

	dt := &core.DatatypeMessage{
		Class:         core.DatatypeString,
		Size:          size,
		ClassBitField: 0, // NULLTERM, ASCII
	}

	ds := &core.DataspaceMessage{Type: core.DataspaceScalar} // H5S_SCALAR

	return dt, ds, nil
}

// inferSlice infers datatype for slices (1D arrays).
func inferSlice(v reflect.Value) (*core.DatatypeMessage, *core.DataspaceMessage, error) {
	if v.Len() == 0 {
		return nil, nil, fmt.Errorf("cannot infer datatype from empty slice")
	}

	elemKind := v.Type().Elem().Kind()
	length := uint64(v.Len()) //nolint:gosec // Safe: slice length conversion

	var dt *core.DatatypeMessage

	switch elemKind {
	case reflect.Int8:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFixed,
			Size:          1,
			ClassBitField: 0x08, // Signed
		}
	case reflect.Uint8:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFixed,
			Size:          1,
			ClassBitField: 0, // Unsigned
		}
	case reflect.Int16:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFixed,
			Size:          2,
			ClassBitField: 0x08, // Signed
		}
	case reflect.Uint16:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFixed,
			Size:          2,
			ClassBitField: 0, // Unsigned
		}
	case reflect.Int32:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFixed,
			Size:          4,
			ClassBitField: 0x08, // Signed
		}
	case reflect.Uint32:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFixed,
			Size:          4,
			ClassBitField: 0, // Unsigned
		}
	case reflect.Int64:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFixed,
			Size:          8,
			ClassBitField: 0x08, // Signed
		}
	case reflect.Uint64:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFixed,
			Size:          8,
			ClassBitField: 0, // Unsigned
		}
	case reflect.Float32:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFloat,
			Size:          4,
			ClassBitField: 0,
		}
	case reflect.Float64:
		dt = &core.DatatypeMessage{
			Class:         core.DatatypeFloat,
			Size:          8,
			ClassBitField: 0,
		}
	default:
		return nil, nil, fmt.Errorf("unsupported slice element type: %s", elemKind)
	}

	ds := &core.DataspaceMessage{
		Dimensions: []uint64{length},
		MaxDims:    nil,
	}

	return dt, ds, nil
}

// encodeAttributeValue encodes a Go value to bytes for attribute storage.
func encodeAttributeValue(value interface{}) ([]byte, error) {
	if ev, ok := value.(*encodedAttributeValue); ok {
		return ev.data, nil
	}

	v := reflect.ValueOf(value)

	switch v.Kind() {
	case reflect.Int8:
		return []byte{byte(v.Int())}, nil
	case reflect.Int16:
		buf := make([]byte, 2)
		binary.LittleEndian.PutUint16(buf, uint16(v.Int())) //nolint:gosec // Safe: validated data type
		return buf, nil
	case reflect.Int32:
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, uint32(v.Int())) //nolint:gosec // Safe: validated data type
		return buf, nil
	case reflect.Int64:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, uint64(v.Int())) //nolint:gosec // Safe: validated data type
		return buf, nil
	case reflect.Uint8:
		return []byte{byte(v.Uint())}, nil
	case reflect.Uint16:
		buf := make([]byte, 2)
		binary.LittleEndian.PutUint16(buf, uint16(v.Uint())) //nolint:gosec // Safe: validated data type
		return buf, nil
	case reflect.Uint32:
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, uint32(v.Uint())) //nolint:gosec // Safe: validated data type
		return buf, nil
	case reflect.Uint64:
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, v.Uint())
		return buf, nil
	case reflect.Float32:
		bits := math.Float32bits(float32(v.Float()))
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, bits)
		return buf, nil
	case reflect.Float64:
		bits := math.Float64bits(v.Float())
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, bits)
		return buf, nil
	case reflect.String:
		// Exactly the string bytes (see inferString); "" is one NUL byte.
		str := v.String()
		buf := make([]byte, max(len(str), 1))
		copy(buf, str)
		return buf, nil
	case reflect.Slice:
		return encodeSliceValue(v)
	default:
		return nil, fmt.Errorf("unsupported value type for encoding: %s", v.Kind())
	}
}

// encodeSliceValue encodes a slice to bytes.
func encodeSliceValue(v reflect.Value) ([]byte, error) {
	elemKind := v.Type().Elem().Kind()
	length := v.Len()

	switch elemKind {
	case reflect.Int8:
		buf := make([]byte, length)
		for i := 0; i < length; i++ {
			buf[i] = byte(v.Index(i).Int())
		}
		return buf, nil
	case reflect.Uint8:
		buf := make([]byte, length)
		for i := 0; i < length; i++ {
			buf[i] = byte(v.Index(i).Uint())
		}
		return buf, nil
	case reflect.Int16:
		buf := make([]byte, length*2)
		for i := 0; i < length; i++ {
			val := v.Index(i).Int()
			binary.LittleEndian.PutUint16(buf[i*2:], uint16(val)) //nolint:gosec // Safe: validated data type
		}
		return buf, nil
	case reflect.Uint16:
		buf := make([]byte, length*2)
		for i := 0; i < length; i++ {
			val := v.Index(i).Uint()
			binary.LittleEndian.PutUint16(buf[i*2:], uint16(val)) //nolint:gosec // Safe: validated data type
		}
		return buf, nil
	case reflect.Int32:
		buf := make([]byte, length*4)
		for i := 0; i < length; i++ {
			val := v.Index(i).Int()
			binary.LittleEndian.PutUint32(buf[i*4:], uint32(val)) //nolint:gosec // Safe: validated data type
		}
		return buf, nil
	case reflect.Uint32:
		buf := make([]byte, length*4)
		for i := 0; i < length; i++ {
			val := v.Index(i).Uint()
			binary.LittleEndian.PutUint32(buf[i*4:], uint32(val)) //nolint:gosec // Safe: validated data type
		}
		return buf, nil
	case reflect.Int64:
		buf := make([]byte, length*8)
		for i := 0; i < length; i++ {
			val := v.Index(i).Int()
			binary.LittleEndian.PutUint64(buf[i*8:], uint64(val)) //nolint:gosec // Safe: validated data type
		}
		return buf, nil
	case reflect.Uint64:
		buf := make([]byte, length*8)
		for i := 0; i < length; i++ {
			val := v.Index(i).Uint()
			binary.LittleEndian.PutUint64(buf[i*8:], val)
		}
		return buf, nil
	case reflect.Float32:
		buf := make([]byte, length*4)
		for i := 0; i < length; i++ {
			val := v.Index(i).Float()
			bits := math.Float32bits(float32(val))
			binary.LittleEndian.PutUint32(buf[i*4:], bits)
		}
		return buf, nil
	case reflect.Float64:
		buf := make([]byte, length*8)
		for i := 0; i < length; i++ {
			val := v.Index(i).Float()
			bits := math.Float64bits(val)
			binary.LittleEndian.PutUint64(buf[i*8:], bits)
		}
		return buf, nil
	default:
		return nil, fmt.Errorf("unsupported slice element type: %s", elemKind)
	}
}

// Suppress unused warnings for now (these will be used when attribute writing is fully implemented).
var (
	_ = (*core.DatatypeMessage)(nil)
	_ = (*core.DataspaceMessage)(nil)
	_ = inferDatatypeFromValue
	_ = encodeAttributeValue
	_ = unsafe.Sizeof(0)
)
