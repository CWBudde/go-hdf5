package core

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// CompoundFieldDef defines a field for compound datatype creation.
// This is used when creating new compound datatypes for writing.
type CompoundFieldDef struct {
	Name   string           // Field name (null-terminated in encoding).
	Offset uint32           // Byte offset within compound structure.
	Type   *DatatypeMessage // Field datatype (can be nested compound).
}

// EncodeCompoundDatatypeV3 encodes a version 3 compound datatype message.
// This is the preferred format for new files (HDF5 1.8+).
//
// Format (version 3, HDF5 File Format Specification IV.A.2.d):
//   - Header (8 bytes):
//   - Byte 0: Class (low 4 bits) | Version (high 4 bits)
//   - Bytes 1-3: Class bit field; bits 0-15 hold the number of members
//   - Bytes 4-7: Size (total compound size in bytes)
//   - For each member:
//   - Name (null-terminated, NOT padded)
//   - Byte offset (little-endian; the field is the minimum number of bytes
//     needed to encode the compound size, e.g. 1 byte for size <= 255)
//   - Member datatype (full datatype message: header + properties)
//
// Parameters:
//   - totalSize: Total size of compound structure in bytes
//   - fields: List of field definitions with names, offsets, and types
//
// Returns:
//   - Encoded datatype message bytes
//   - Error if encoding fails
//
// Reference: HDF5 Format Spec IV.A.2.d, H5Odtype.c (H5O__dtype_encode_helper).
func EncodeCompoundDatatypeV3(totalSize uint32, fields []CompoundFieldDef) ([]byte, error) {
	if len(fields) == 0 {
		return nil, errors.New("compound datatype must have at least one field")
	}

	if totalSize == 0 {
		return nil, errors.New("compound datatype size cannot be 0")
	}

	// The member count lives in bits 0-15 of the class bit field.
	if len(fields) > 0xFFFF {
		return nil, fmt.Errorf("too many fields: %d (max: %d)", len(fields), 0xFFFF)
	}

	offsetSize := compoundMemberOffsetSize(totalSize)

	// Calculate properties size (all member definitions)
	propsSize := 0

	for i, field := range fields {
		if field.Name == "" {
			return nil, fmt.Errorf("field %d: name cannot be empty", i)
		}
		if field.Type == nil {
			return nil, fmt.Errorf("field %d (%s): type cannot be nil", i, field.Name)
		}
		if err := validateInlineDatatype(field.Type); err != nil {
			return nil, fmt.Errorf("field %d (%s): %w", i, field.Name, err)
		}
		if uint64(field.Offset)+uint64(field.Type.Size) > uint64(totalSize) {
			return nil, fmt.Errorf("field %d (%s): offset %d + size %d exceeds compound size %d",
				i, field.Name, field.Offset, field.Type.Size, totalSize)
		}

		propsSize += len(field.Name) + 1        // Name + null terminator
		propsSize += offsetSize                 // Byte offset
		propsSize += 8                          // Member datatype header
		propsSize += len(field.Type.Properties) // Member datatype properties
	}

	// Allocate buffer: header (8 bytes) + properties
	buf := make([]byte, 8+propsSize)
	offset := 0

	// Header: class, version and class bit field (member count).
	version := uint8(3)
	classAndVersion := uint32(DatatypeCompound) | (uint32(version) << 4) | (uint32(len(fields)) << 8) //nolint:gosec // G115: validated above
	binary.LittleEndian.PutUint32(buf[offset:], classAndVersion)
	offset += 4

	// Byte 4-7: Total compound size
	binary.LittleEndian.PutUint32(buf[offset:], totalSize)
	offset += 4

	// Encode each member
	for _, field := range fields {
		// 1. Member name (null-terminated, NOT padded in version 3)
		copy(buf[offset:], field.Name)
		offset += len(field.Name)
		buf[offset] = 0 // Null terminator
		offset++

		// 2. Member byte offset (offsetSize bytes, little-endian)
		for b := 0; b < offsetSize; b++ {
			buf[offset+b] = byte(field.Offset >> (8 * b))
		}
		offset += offsetSize

		// 3. Member datatype: header (8 bytes) + properties
		memberClassAndVersion := uint32(field.Type.Class) | (uint32(field.Type.Version) << 4) | (field.Type.ClassBitField << 8)
		binary.LittleEndian.PutUint32(buf[offset:], memberClassAndVersion)
		offset += 4

		binary.LittleEndian.PutUint32(buf[offset:], field.Type.Size)
		offset += 4

		copy(buf[offset:], field.Type.Properties)
		offset += len(field.Type.Properties)
	}

	// Validate we used exactly the expected space
	if offset != len(buf) {
		return nil, fmt.Errorf("internal error: buffer size mismatch (expected %d, used %d)", len(buf), offset)
	}

	return buf, nil
}

// validateInlineDatatype checks that a member datatype's properties have the
// exact length its class requires, so the encoded member list can be walked
// by a reader (compound members are not length-prefixed).
func validateInlineDatatype(dt *DatatypeMessage) error {
	encoded := make([]byte, 8+len(dt.Properties))
	binary.LittleEndian.PutUint32(encoded[0:4], uint32(dt.Class)|(uint32(dt.Version)<<4)|(dt.ClassBitField<<8))
	binary.LittleEndian.PutUint32(encoded[4:8], dt.Size)
	copy(encoded[8:], dt.Properties)
	n, err := datatypeEncodedLen(encoded)
	if err != nil {
		return fmt.Errorf("invalid member datatype: %w", err)
	}
	if n != len(encoded) {
		return fmt.Errorf("invalid member datatype: class %d expects %d property bytes, got %d",
			dt.Class, n-8, len(dt.Properties))
	}
	return nil
}

// EncodeCompoundDatatypeV1 encodes a version 1 compound datatype message.
// This is the legacy format for HDF5 1.0-1.6 compatibility.
//
// Format (version 1):
//   - Header (8 bytes):
//   - Byte 0-3: Class (4 bits) | Version (4 bits) | NumMembers (16 bits, low)
//   - Byte 4-7: Size (total compound size in bytes)
//   - For each member:
//   - Name (null-terminated, padded to 8-byte boundary)
//   - Offset (4 bytes, uint32)
//   - Array info (28 bytes, always present even for scalars)
//   - Member datatype (recursive, variable length, NO padding between members)
//
// Parameters:
//   - totalSize: Total size of compound structure in bytes
//   - fields: List of field definitions
//
// Returns:
//   - Encoded datatype message bytes
//   - Error if encoding fails
//
// Reference: H5Odtype.c:360-481.
func EncodeCompoundDatatypeV1(totalSize uint32, fields []CompoundFieldDef) ([]byte, error) {
	if len(fields) == 0 {
		return nil, errors.New("compound datatype must have at least one field")
	}

	if totalSize == 0 {
		return nil, errors.New("compound datatype size cannot be 0")
	}

	// Validate field count fits in uint16 (version 1 uses 16 bits in ClassBitField)
	if len(fields) > 0xFFFF {
		return nil, fmt.Errorf("too many fields for version 1: %d (max: 65535)", len(fields))
	}

	// Calculate properties size
	propsSize := 0

	for i, field := range fields {
		if field.Name == "" {
			return nil, fmt.Errorf("field %d: name cannot be empty", i)
		}
		if field.Type == nil {
			return nil, fmt.Errorf("field %d (%s): type cannot be nil", i, field.Name)
		}

		// Name (null-terminated, padded to 8-byte boundary)
		nameLen := len(field.Name)
		paddedNameLen := ((nameLen + 8) / 8) * 8 // Round up to nearest 8 bytes
		propsSize += paddedNameLen

		// Offset (4 bytes)
		propsSize += 4

		// Array info (28 bytes, always present in version 1)
		propsSize += 28

		// Member datatype (inline encoding: header + properties, NO padding)
		propsSize += 8                          // Header (8 bytes always)
		propsSize += len(field.Type.Properties) // Properties (variable)
	}

	// Allocate buffer: header (8 bytes) + properties
	buf := make([]byte, 8+propsSize)
	offset := 0

	// Encode header (8 bytes)
	// Byte 0-3: Class (4 bits) | Version (4 bits) | NumMembers (16 bits low) | Reserved (8 bits)
	version := uint8(1)
	numMembers := uint16(len(fields)) //nolint:gosec // G115: validated above
	classAndVersion := uint32(DatatypeCompound) | (uint32(version) << 4) | (uint32(numMembers) << 8)
	binary.LittleEndian.PutUint32(buf[offset:], classAndVersion)
	offset += 4

	// Byte 4-7: Total compound size
	binary.LittleEndian.PutUint32(buf[offset:], totalSize)
	offset += 4

	// Encode each member
	for _, field := range fields {
		// 1. Member name (null-terminated, padded to 8-byte boundary)
		nameStart := offset
		copy(buf[offset:], field.Name)
		offset += len(field.Name)
		buf[offset] = 0 // Null terminator
		// offset++ is not needed here - will be recalculated below

		// Pad to 8-byte boundary
		nameLen := len(field.Name)
		paddedLen := ((nameLen + 8) / 8) * 8
		offset = nameStart + paddedLen

		// 2. Member byte offset (4 bytes, uint32)
		binary.LittleEndian.PutUint32(buf[offset:], field.Offset)
		offset += 4

		// 3. Array info (28 bytes, always zeros for scalar members in MVP)
		// Format: dimensionality (1) + reserved (3) + permutation (4) + reserved (4) + dimensions (16)
		// We support only scalar members for now, so all zeros
		offset += 28 // Skip, already zeros

		// 4. Member datatype (inline encoding, NO padding)
		// Same as V3: encode header + properties inline

		// Encode member datatype header (8 bytes)
		memberClassAndVersion := uint32(field.Type.Class) | (uint32(field.Type.Version) << 4) | (field.Type.ClassBitField << 8)
		binary.LittleEndian.PutUint32(buf[offset:], memberClassAndVersion)
		offset += 4

		binary.LittleEndian.PutUint32(buf[offset:], field.Type.Size)
		offset += 4

		// Encode member datatype properties
		copy(buf[offset:], field.Type.Properties)
		offset += len(field.Type.Properties)
	}

	// Validate we used exactly the expected space
	if offset != len(buf) {
		return nil, fmt.Errorf("internal error: buffer size mismatch (expected %d, used %d)", len(buf), offset)
	}

	return buf, nil
}

// CreateCompoundTypeFromFields creates a DatatypeMessage for a compound type.
// This is a convenience function for creating compound datatypes with automatic
// offset calculation.
//
// Parameters:
//   - fields: List of field definitions (offsets will be calculated)
//
// Returns:
//   - DatatypeMessage ready for writing
//   - Error if creation fails.
func CreateCompoundTypeFromFields(fields []CompoundFieldDef) (*DatatypeMessage, error) {
	if len(fields) == 0 {
		return nil, errors.New("compound type must have at least one field")
	}

	// Validate offsets are correct and calculate total size
	currentOffset := uint32(0)
	for i, field := range fields {
		if field.Offset != currentOffset {
			return nil, fmt.Errorf("field %d (%s): offset mismatch (expected %d, got %d)", i, field.Name, currentOffset, field.Offset)
		}
		currentOffset += field.Type.Size
	}

	totalSize := currentOffset

	// Encode as version 3 (modern format)
	encoded, err := EncodeCompoundDatatypeV3(totalSize, fields)
	if err != nil {
		return nil, fmt.Errorf("failed to encode compound type: %w", err)
	}

	// Parse back to get DatatypeMessage (includes validation)
	dt, err := ParseDatatypeMessage(encoded)
	if err != nil {
		return nil, fmt.Errorf("failed to parse encoded compound type: %w", err)
	}

	return dt, nil
}

// CreateBasicDatatypeMessage creates a simple datatype message for basic types.
// This is a helper for creating member types in compound datatypes.
//
//   - DatatypeFixed: signed little-endian integer of 1, 2, 4 or 8 bytes;
//     properties are bit offset (uint16) and bit precision (uint16).
//   - DatatypeFloat: little-endian IEEE 754 float of 4 or 8 bytes; properties
//     are bit offset, bit precision, exponent/mantissa location and size and
//     exponent bias.
//   - DatatypeString: fixed-length, null-terminated ASCII string of size
//     bytes; strings have no properties (padding and character set live in
//     the class bit field).
func CreateBasicDatatypeMessage(class DatatypeClass, size uint32) (*DatatypeMessage, error) {
	switch class {
	case DatatypeFixed, DatatypeFloat:
		dt := &DatatypeMessage{Class: class, Size: size}
		if class == DatatypeFixed {
			dt.ClassBitField = 0x08 // Bit 3: signed (two's complement)
		}
		encoded, err := encodeDatatypeNumeric(dt)
		if err != nil {
			return nil, err
		}
		return ParseDatatypeMessage(encoded)

	case DatatypeString:
		if size == 0 {
			return nil, errors.New("fixed-length strings must have size > 0")
		}
		return &DatatypeMessage{
			Class:         DatatypeString,
			Version:       1,
			Size:          size,
			ClassBitField: 0, // Null-terminated, ASCII
			Properties:    []byte{},
		}, nil

	default:
		return nil, fmt.Errorf("unsupported datatype class: %d", class)
	}
}
