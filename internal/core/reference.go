package core

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/cwbudde/go-hdf5/internal/utils"
)

// ObjectReference is an HDF5 object reference (H5T_STD_REF_OBJ): the file
// address of the referenced object's header.
type ObjectReference uint64

// Reference types stored in bits 0-3 of a reference datatype's class bit
// field (HDF5 Format Specification IV.A.2.d, class 7).
const (
	referenceTypeObject = 0
)

// IsObjectReference reports whether dt is an object reference
// (H5T_STD_REF_OBJ) with a 4- or 8-byte address.
func (dt *DatatypeMessage) IsObjectReference() bool {
	return dt.Class == DatatypeReference &&
		dt.ClassBitField&0x0F == referenceTypeObject &&
		(dt.Size == 4 || dt.Size == 8)
}

// VarLenBase returns the base type of a variable-length datatype.
func (dt *DatatypeMessage) VarLenBase() (*DatatypeMessage, error) {
	if dt.Class != DatatypeVarLen {
		return nil, fmt.Errorf("datatype class %d is not variable-length", dt.Class)
	}
	return ParseDatatypeMessage(dt.Properties)
}

// decodeObjectReference decodes one object reference of the given size
// (4 or 8 bytes, little-endian as written by libhdf5).
func decodeObjectReference(data []byte, size uint32) (ObjectReference, error) {
	switch {
	case size == 8 && len(data) >= 8:
		return ObjectReference(binary.LittleEndian.Uint64(data)), nil
	case size == 4 && len(data) >= 4:
		return ObjectReference(binary.LittleEndian.Uint32(data)), nil
	}
	return 0, errors.New("insufficient data for object reference")
}

// DecodeObjectReferences decodes n object references of the given size
// from data.
func DecodeObjectReferences(data []byte, n uint64, size uint32) ([]ObjectReference, error) {
	if size != 4 && size != 8 {
		return nil, fmt.Errorf("unsupported object reference size %d", size)
	}
	total, err := utils.SafeMultiply(n, uint64(size))
	if err != nil || total > uint64(len(data)) {
		return nil, fmt.Errorf("reference data size mismatch: %d references of %d bytes, have %d bytes",
			n, size, len(data))
	}
	refs := make([]ObjectReference, n)
	for i := range n {
		refs[i], _ = decodeObjectReference(data[i*uint64(size):], size)
	}
	return refs, nil
}

// readReferenceValue decodes an object reference attribute: an
// ObjectReference for a scalar, []ObjectReference otherwise.
func (a *Attribute) readReferenceValue(totalElements uint64, isScalar bool) (interface{}, error) {
	if !a.Datatype.IsObjectReference() {
		return nil, fmt.Errorf("unsupported reference type %d (size %d): only object references are supported",
			a.Datatype.ClassBitField&0x0F, a.Datatype.Size)
	}
	refs, err := DecodeObjectReferences(a.Data, totalElements, a.Datatype.Size)
	if err != nil {
		return nil, err
	}
	if isScalar {
		return refs[0], nil
	}
	return refs, nil
}

// readVarLenReferences decodes an attribute whose elements are
// variable-length sequences of object references (e.g. the H5DS
// DIMENSION_LIST attribute) into one []ObjectReference per element.
func (a *Attribute) readVarLenReferences(base *DatatypeMessage, totalElements uint64) ([][]ObjectReference, error) {
	if a.reader == nil {
		return nil, errors.New("variable-length attribute requires file reader (not available)")
	}
	offsetSize := a.offsetSize
	//nolint:gosec // G115: offsetSize is 4 or 8
	elemSize := uint64(4 + offsetSize + 4)
	total, err := utils.SafeMultiply(totalElements, elemSize)
	if err != nil || total > uint64(len(a.Data)) {
		return nil, fmt.Errorf("attribute data size mismatch for variable-length references: %d elements, %d bytes",
			totalElements, len(a.Data))
	}

	heaps := make(map[uint64]*GlobalHeapCollection)
	out := make([][]ObjectReference, totalElements)
	for i := range totalElements {
		elem := a.Data[i*elemSize : (i+1)*elemSize]
		count := uint64(binary.LittleEndian.Uint32(elem[0:4]))
		ref, err := ParseGlobalHeapReference(elem[4:], offsetSize)
		if err != nil {
			return nil, fmt.Errorf("element %d: %w", i, err)
		}
		if count == 0 || ref.HeapAddress == 0 || ref.HeapAddress == undefinedAddress {
			out[i] = []ObjectReference{}
			continue
		}
		heap, ok := heaps[ref.HeapAddress]
		if !ok {
			heap, err = ReadGlobalHeapCollection(a.reader, ref.HeapAddress, offsetSize)
			if err != nil {
				return nil, fmt.Errorf("element %d: %w", i, err)
			}
			heaps[ref.HeapAddress] = heap
		}
		obj, err := heap.GetObject(ref.ObjectIndex)
		if err != nil {
			return nil, fmt.Errorf("element %d: %w", i, err)
		}
		refs, err := DecodeObjectReferences(obj.Data, count, base.Size)
		if err != nil {
			return nil, fmt.Errorf("element %d: %w", i, err)
		}
		out[i] = refs
	}
	return out, nil
}

// readCompoundValue decodes a compound attribute: a CompoundValue for a
// scalar, []CompoundValue otherwise.
func (a *Attribute) readCompoundValue(totalElements uint64, isScalar bool) (interface{}, error) {
	ct, err := ParseCompoundType(a.Datatype)
	if err != nil {
		return nil, fmt.Errorf("failed to parse compound attribute type: %w", err)
	}
	offsetSize := a.offsetSize
	if offsetSize == 0 {
		offsetSize = 8
	}
	sb := &Superblock{
		OffsetSize: uint8(offsetSize), //nolint:gosec // G115: 4 or 8
		LengthSize: 8,
		Endianness: binary.LittleEndian,
	}
	values, err := parseCompoundData(a.Data, ct, totalElements, a.reader, sb)
	if err != nil {
		return nil, err
	}
	if isScalar {
		return values[0], nil
	}
	return values, nil
}
