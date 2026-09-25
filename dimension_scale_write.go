package hdf5

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/cwbudde/go-hdf5/internal/core"
)

// ObjectRef is an HDF5 object reference (H5T_STD_REF_OBJ): the file
// address of the referenced object's header.
//
// Attribute values of the following types are written with reference-based
// datatypes by WriteAttribute:
//   - ObjectRef, []ObjectRef: H5T_STD_REF_OBJ scalar / 1D array
//   - [][]ObjectRef: 1D array of variable-length sequences of
//     H5T_STD_REF_OBJ (the layout of the DIMENSION_LIST attribute)
//   - []DimensionReference: 1D array of the compound
//     {"dataset": H5T_STD_REF_OBJ, "dimension": H5T_STD_U32LE} (the layout of the
//     REFERENCE_LIST attribute)
type ObjectRef uint64

// DimensionReference is one element of a REFERENCE_LIST attribute: the
// dataset a dimension scale is attached to, and the dimension index.
type DimensionReference struct {
	Dataset ObjectRef
	Index   int32
}

// Address returns the file address of the dataset's object header.
func (dw *DatasetWriter) Address() uint64 {
	return dw.address
}

// Reference returns an object reference to this dataset.
func (dw *DatasetWriter) Reference() ObjectRef {
	return ObjectRef(dw.address)
}

// Dimension scale attribute names and values defined by the HDF5 Dimension
// Scales specification (H5DS).
const (
	dimScaleClassAttr   = "CLASS"
	dimScaleClassValue  = "DIMENSION_SCALE"
	dimScaleNameAttr    = "NAME"
	dimensionListAttr   = "DIMENSION_LIST"
	referenceListAttr   = "REFERENCE_LIST"
	objectReferenceSize = 8
)

// SetDimensionScale marks the dataset as a dimension scale, like
// H5DSset_scale: it writes CLASS="DIMENSION_SCALE" and, when name is not
// empty, NAME=name.
func (dw *DatasetWriter) SetDimensionScale(name string) error {
	if err := dw.WriteAttribute(dimScaleClassAttr, dimScaleClassValue); err != nil {
		return err
	}
	if name == "" {
		return nil
	}
	return dw.WriteAttribute(dimScaleNameAttr, name)
}

// AttachDimensionScale attaches scale to dimension dimIdx of the dataset,
// like H5DSattach_scale. The dataset's DIMENSION_LIST attribute (a
// variable-length list of scale references per dimension) and the scale's
// REFERENCE_LIST attribute (the {dataset, dimension} back references) are
// maintained by the FileWriter and written when it is closed, so all
// attachments made during a session end up in one attribute per object.
// Attachments already stored in the file (OpenForWrite) are kept.
// Attaching the same scale to the same dimension twice is a no-op.
//
// The scale should be marked with SetDimensionScale. A dataset cannot be
// attached to itself.
func (dw *DatasetWriter) AttachDimensionScale(dimIdx int, scale *DatasetWriter) error {
	if scale == nil {
		return fmt.Errorf("attach dimension scale: scale is nil")
	}
	if scale.fileWriter != dw.fileWriter {
		return fmt.Errorf("attach dimension scale: scale belongs to a different file")
	}
	if scale.address == dw.address {
		return fmt.Errorf("attach dimension scale: cannot attach %s to itself", dw.name)
	}
	if dimIdx < 0 || dimIdx >= len(dw.dims) {
		return fmt.Errorf("attach dimension scale: dimension %d out of range for rank %d", dimIdx, len(dw.dims))
	}

	fw := dw.fileWriter
	if fw.dimScales == nil {
		fw.dimScales = &dimensionScaleState{
			dimLists: make(map[uint64][][]ObjectRef),
			refLists: make(map[uint64][]DimensionReference),
		}
	}
	st := fw.dimScales

	// Start from the attachments already stored in the file (files opened
	// with OpenForWrite), so rewriting the attributes on Close keeps them.
	lists, ok := st.dimLists[dw.address]
	if !ok {
		existing, err := fw.readDimensionList(dw.address)
		if err != nil {
			return fmt.Errorf("attach dimension scale: %s: %w", dw.name, err)
		}
		lists = make([][]ObjectRef, len(dw.dims))
		for i := 0; i < len(lists) && i < len(existing); i++ {
			lists[i] = existing[i]
		}
		st.dimLists[dw.address] = lists
		st.datasets = append(st.datasets, dw)
	}
	refs, ok := st.refLists[scale.address]
	if !ok {
		existing, err := fw.readReferenceList(scale.address)
		if err != nil {
			return fmt.Errorf("attach dimension scale: %s: %w", scale.name, err)
		}
		refs = existing
		st.scales = append(st.scales, scale)
	}

	if !slices.Contains(lists[dimIdx], scale.Reference()) {
		lists[dimIdx] = append(lists[dimIdx], scale.Reference())
	}
	back := DimensionReference{
		Dataset: dw.Reference(),
		Index:   int32(dimIdx), //nolint:gosec // G115: dimIdx < rank (<= 32)
	}
	if !slices.Contains(refs, back) {
		refs = append(refs, back)
	}
	st.refLists[scale.address] = refs
	return nil
}

// errNoAttribute reports that an object has no attribute of the given name.
var errNoAttribute = errors.New("attribute not found")

// readObjectAttribute returns the attribute name of the object at addr, or
// errNoAttribute.
func (fw *FileWriter) readObjectAttribute(addr uint64, name string) (*core.Attribute, error) {
	sb := fw.file.Superblock()
	reader := fw.writer.Reader()
	oh, err := core.ReadObjectHeader(reader, addr, sb)
	if err != nil {
		return nil, fmt.Errorf("read object header: %w", err)
	}
	attrs, err := core.ParseAttributesFromMessages(reader, oh.Messages, sb)
	if err != nil {
		return nil, fmt.Errorf("read attributes: %w", err)
	}
	for _, a := range attrs {
		if a.Name == name {
			return a, nil
		}
	}
	return nil, errNoAttribute
}

// readDimensionList decodes an existing DIMENSION_LIST attribute (a 1D
// array of variable-length sequences of object references).
func (fw *FileWriter) readDimensionList(addr uint64) ([][]ObjectRef, error) {
	attr, err := fw.readObjectAttribute(addr, dimensionListAttr)
	if errors.Is(err, errNoAttribute) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if attr.Datatype.Class != core.DatatypeVarLen {
		return nil, fmt.Errorf("%s is not a variable-length attribute", dimensionListAttr)
	}
	offsetSize := int(fw.file.Superblock().OffsetSize)
	elemSize := 4 + offsetSize + 4
	n := int(attr.Dataspace.TotalElements()) //nolint:gosec // G115: bounded by attribute data below
	if n < 0 || n*elemSize > len(attr.Data) {
		return nil, fmt.Errorf("%s data too short", dimensionListAttr)
	}

	reader := fw.writer.Reader()
	heaps := make(map[uint64]*core.GlobalHeapCollection)
	out := make([][]ObjectRef, n)
	for i := range out {
		elem := attr.Data[i*elemSize : (i+1)*elemSize]
		count := int(binary.LittleEndian.Uint32(elem[0:4]))
		ref, err := core.ParseGlobalHeapReference(elem[4:], offsetSize)
		if err != nil {
			return nil, fmt.Errorf("%s[%d]: %w", dimensionListAttr, i, err)
		}
		if count == 0 || !isDefinedAddress(ref.HeapAddress) {
			continue
		}
		refs, err := readReferenceSequence(reader, heaps, ref, count, offsetSize)
		if err != nil {
			return nil, fmt.Errorf("%s[%d]: %w", dimensionListAttr, i, err)
		}
		out[i] = refs
	}
	return out, nil
}

// readReferenceSequence reads a variable-length sequence of count object
// references from the global heap object ref points to.
func readReferenceSequence(reader io.ReaderAt, heaps map[uint64]*core.GlobalHeapCollection,
	ref *core.GlobalHeapReference, count, offsetSize int,
) ([]ObjectRef, error) {
	heap, ok := heaps[ref.HeapAddress]
	if !ok {
		var err error
		heap, err = core.ReadGlobalHeapCollection(reader, ref.HeapAddress, offsetSize)
		if err != nil {
			return nil, err
		}
		heaps[ref.HeapAddress] = heap
	}
	obj, err := heap.GetObject(ref.ObjectIndex)
	if err != nil {
		return nil, err
	}
	if count*objectReferenceSize > len(obj.Data) {
		return nil, fmt.Errorf("sequence of %d references exceeds heap object", count)
	}
	refs := make([]ObjectRef, count)
	for j := range refs {
		refs[j] = ObjectRef(binary.LittleEndian.Uint64(obj.Data[j*objectReferenceSize:]))
	}
	return refs, nil
}

// readReferenceList decodes an existing REFERENCE_LIST attribute (a 1D
// array of the compound {dataset reference, uint32 dimension}).
func (fw *FileWriter) readReferenceList(addr uint64) ([]DimensionReference, error) {
	attr, err := fw.readObjectAttribute(addr, referenceListAttr)
	if errors.Is(err, errNoAttribute) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if attr.Datatype.Class != core.DatatypeCompound {
		return nil, fmt.Errorf("%s is not a compound attribute", referenceListAttr)
	}
	refOff, dimOff := uint32(0), uint32(8) // libhdf5 ds_list_t layout
	if ct, err := core.ParseCompoundType(attr.Datatype); err == nil {
		for _, m := range ct.Members {
			switch m.Name {
			case "dataset":
				refOff = m.Offset
			case "dimension":
				dimOff = m.Offset
			}
		}
	}
	elemSize := int(attr.Datatype.Size)
	n := int(attr.Dataspace.TotalElements()) //nolint:gosec // G115: bounded by attribute data below
	if elemSize < int(refOff)+objectReferenceSize || elemSize < int(dimOff)+4 || n < 0 || n*elemSize > len(attr.Data) {
		return nil, fmt.Errorf("unsupported %s layout", referenceListAttr)
	}
	out := make([]DimensionReference, n)
	for i := range out {
		elem := attr.Data[i*elemSize : (i+1)*elemSize]
		out[i] = DimensionReference{
			Dataset: ObjectRef(binary.LittleEndian.Uint64(elem[refOff:])),
			Index:   int32(binary.LittleEndian.Uint32(elem[dimOff:])), //nolint:gosec // G115: two's complement
		}
	}
	return out, nil
}

// dimensionScaleState collects dimension scale attachments until Close.
// Slices keep first-use order so output is deterministic.
type dimensionScaleState struct {
	datasets []*DatasetWriter
	dimLists map[uint64][][]ObjectRef
	scales   []*DatasetWriter
	refLists map[uint64][]DimensionReference
}

// writeDimensionScaleAttributes writes the DIMENSION_LIST and REFERENCE_LIST
// attributes collected by AttachDimensionScale.
func (fw *FileWriter) writeDimensionScaleAttributes() error {
	st := fw.dimScales
	if st == nil {
		return nil
	}
	fw.dimScales = nil

	for _, ds := range st.datasets {
		if err := ds.WriteAttribute(dimensionListAttr, st.dimLists[ds.address]); err != nil {
			return fmt.Errorf("write %s on %s: %w", dimensionListAttr, ds.name, err)
		}
	}
	for _, sc := range st.scales {
		if err := sc.WriteAttribute(referenceListAttr, st.refLists[sc.address]); err != nil {
			return fmt.Errorf("write %s on %s: %w", referenceListAttr, sc.name, err)
		}
	}
	return nil
}

// encodedAttributeValue is an attribute value whose datatype, dataspace and
// raw data have already been encoded. inferDatatypeFromValue and
// encodeAttributeValue pass it through unchanged.
type encodedAttributeValue struct {
	datatype  *core.DatatypeMessage
	dataspace *core.DataspaceMessage
	data      []byte
}

// objectReferenceDatatype returns the H5T_STD_REF_OBJ datatype.
func objectReferenceDatatype() *core.DatatypeMessage {
	return &core.DatatypeMessage{
		Class:   core.DatatypeReference,
		Version: 1,
		Size:    objectReferenceSize,
	}
}

// prepareAttributeValue converts reference-based attribute values into an
// encodedAttributeValue. Other values are returned unchanged.
func (fw *FileWriter) prepareAttributeValue(value interface{}) (interface{}, error) {
	switch v := value.(type) {
	case ObjectRef:
		ev := encodeObjectReferences([]ObjectRef{v})
		ev.dataspace = &core.DataspaceMessage{Type: core.DataspaceScalar} // H5S_SCALAR
		return ev, nil
	case []ObjectRef:
		if len(v) == 0 {
			return nil, fmt.Errorf("cannot write empty []ObjectRef attribute")
		}
		return encodeObjectReferences(v), nil
	case [][]ObjectRef:
		return fw.encodeObjectReferenceLists(v)
	case []DimensionReference:
		return encodeDimensionReferences(v)
	default:
		return value, nil
	}
}

func encodeObjectReferences(refs []ObjectRef) *encodedAttributeValue {
	data := make([]byte, len(refs)*objectReferenceSize)
	for i, r := range refs {
		binary.LittleEndian.PutUint64(data[i*objectReferenceSize:], uint64(r))
	}
	return &encodedAttributeValue{
		datatype:  objectReferenceDatatype(),
		dataspace: &core.DataspaceMessage{Dimensions: []uint64{uint64(len(refs))}},
		data:      data,
	}
}

// encodeObjectReferenceLists encodes a 1D array of variable-length sequences
// of object references. Each sequence is stored in the global heap; the
// attribute holds {uint32 length, heap collection address, uint32 index}.
func (fw *FileWriter) encodeObjectReferenceLists(lists [][]ObjectRef) (*encodedAttributeValue, error) {
	if len(lists) == 0 {
		return nil, fmt.Errorf("cannot write empty [][]ObjectRef attribute")
	}
	if fw.globalHeapWriter == nil {
		fw.globalHeapWriter = newGlobalHeapWriter(fw)
	}

	base, err := core.EncodeDatatypeMessage(objectReferenceDatatype())
	if err != nil {
		return nil, fmt.Errorf("encode reference base type: %w", err)
	}

	const elemSize = 4 + 8 + 4 // length + heap address + object index
	data := make([]byte, len(lists)*elemSize)
	for i, refs := range lists {
		elem := data[i*elemSize : (i+1)*elemSize]
		if len(refs) == 0 {
			continue // empty sequence: zero length, null heap ID
		}
		seq := encodeObjectReferences(refs).data
		hid, err := fw.globalHeapWriter.WriteToGlobalHeap(seq)
		if err != nil {
			return nil, fmt.Errorf("write reference list %d to global heap: %w", i, err)
		}
		binary.LittleEndian.PutUint32(elem[0:4], uint32(len(refs))) //nolint:gosec // G115: bounded by heap size
		binary.LittleEndian.PutUint64(elem[4:12], hid.CollectionAddress)
		binary.LittleEndian.PutUint32(elem[12:16], uint32(hid.ObjectIndex))
	}

	return &encodedAttributeValue{
		datatype: &core.DatatypeMessage{
			Class:         core.DatatypeVarLen,
			Version:       1,
			Size:          elemSize,
			ClassBitField: 0, // sequence
			Properties:    base,
		},
		dataspace: &core.DataspaceMessage{Dimensions: []uint64{uint64(len(lists))}},
		data:      data,
	}, nil
}

// encodeDimensionReferences encodes a REFERENCE_LIST style compound array,
// laid out like libhdf5's ds_list_t: {hobj_ref_t dataset; unsigned dimension}
// with 8-byte alignment (16 bytes per element).
func encodeDimensionReferences(refs []DimensionReference) (*encodedAttributeValue, error) {
	if len(refs) == 0 {
		return nil, fmt.Errorf("cannot write empty []DimensionReference attribute")
	}

	const elemSize = 16
	props, err := encodeCompoundV1Members([]compoundMemberV1{
		{name: "dataset", offset: 0, datatype: objectReferenceDatatype()},
		{name: "dimension", offset: 8, datatype: &core.DatatypeMessage{
			// H5T_STD_U32LE, as libhdf5 writes it (see h5ex_ds1.ddl)
			Class: core.DatatypeFixed, Version: 1, Size: 4, ClassBitField: 0, // little-endian uint32
		}},
	})
	if err != nil {
		return nil, err
	}

	data := make([]byte, len(refs)*elemSize)
	for i, r := range refs {
		binary.LittleEndian.PutUint64(data[i*elemSize:], uint64(r.Dataset))
		binary.LittleEndian.PutUint32(data[i*elemSize+8:], uint32(r.Index)) //nolint:gosec // G115: two's complement
	}

	return &encodedAttributeValue{
		datatype: &core.DatatypeMessage{
			Class:         core.DatatypeCompound,
			Version:       1,
			Size:          elemSize,
			ClassBitField: 2, // number of members
			Properties:    props,
		},
		dataspace: &core.DataspaceMessage{Dimensions: []uint64{uint64(len(refs))}},
		data:      data,
	}, nil
}

type compoundMemberV1 struct {
	name     string
	offset   uint32
	datatype *core.DatatypeMessage
}

// encodeCompoundV1Members encodes the member list of a version 1 compound
// datatype (the version libhdf5 writes by default): name padded to a
// multiple of 8, 4-byte offset, 28 bytes of (unused) array information and
// the member datatype.
func encodeCompoundV1Members(members []compoundMemberV1) ([]byte, error) {
	var buf []byte
	for _, m := range members {
		nameLen := (len(m.name) + 1 + 7) / 8 * 8
		name := make([]byte, nameLen)
		copy(name, m.name)
		buf = append(buf, name...)

		var fixed [32]byte
		binary.LittleEndian.PutUint32(fixed[0:4], m.offset)
		// fixed[4]: dimensionality 0, reserved, permutation, reserved, dims: all zero
		buf = append(buf, fixed[:]...)

		dt, err := core.EncodeDatatypeMessage(m.datatype)
		if err != nil {
			return nil, fmt.Errorf("encode compound member %q: %w", m.name, err)
		}
		buf = append(buf, dt...)
	}
	return buf, nil
}
