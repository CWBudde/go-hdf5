package core

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// CompoundMember represents a single field in a compound datatype.
type CompoundMember struct {
	Name   string           // Field name.
	Offset uint32           // Byte offset within the compound structure.
	Type   *DatatypeMessage // Member datatype (can be any type, including nested compound).
}

// CompoundType represents a parsed compound datatype with all its members.
type CompoundType struct {
	Size    uint32           // Total size of the compound structure in bytes.
	Members []CompoundMember // List of members/fields.
}

// ParseCompoundType parses compound datatype properties to extract member information.
//
// Layout per HDF5 File Format Specification (IV.A.2.d, datatype class 6).
// For all versions the number of members is stored in bits 0-15 of the
// class bit field. Each member is encoded as:
//
//   - Version 1: name (null-terminated, padded to a multiple of 8 bytes),
//     byte offset (4 bytes), dimensionality (1), reserved (3), dimension
//     permutation (4), reserved (4), dimension sizes (4 x 4 bytes), member
//     datatype.
//   - Version 2: name (padded to a multiple of 8 bytes), byte offset
//     (4 bytes), member datatype.
//   - Version 3 (and 4, 5): name (null-terminated, not padded), byte offset
//     (the minimum number of bytes needed to encode the compound size),
//     member datatype.
//
// go-hdf5 releases up to v0.16.1 wrote version 3 compounds with the member
// count as a uint32 at the start of the properties (class bit field 0) and
// 4-byte member offsets. Such messages are still accepted.
func ParseCompoundType(dt *DatatypeMessage) (*CompoundType, error) {
	if dt.Class != DatatypeCompound {
		return nil, errors.New("not a compound datatype")
	}

	members, _, err := walkCompoundMembers(dt.Properties, dt.Version, dt.ClassBitField, dt.Size)
	if err != nil {
		return nil, err
	}

	return &CompoundType{Size: dt.Size, Members: members}, nil
}

// compoundMemberOffsetSize returns the size in bytes of the member offset
// field of a version 3 compound datatype: the minimum number of bytes
// needed to encode the datatype size (H5VM_limit_enc_size in the C library).
func compoundMemberOffsetSize(size uint32) int {
	n := 1
	for s := size >> 8; s > 0; s >>= 8 {
		n++
	}
	return n
}

// walkCompoundMembers decodes the member list of a compound datatype and
// returns the members and the number of property bytes they occupy.
func walkCompoundMembers(properties []byte, version uint8, classBitField, size uint32) ([]CompoundMember, int, error) {
	numMembers := int(classBitField & 0xFFFF)

	switch version {
	case 1, 2:
		return walkCompoundMembersV1V2(properties, version, numMembers)
	case 3, 4, 5:
		// Versions 4 (HDF5 1.12 references) and 5 (HDF5 2.0) encode
		// compound members exactly like version 3.
		if numMembers == 0 && version == 3 {
			return walkCompoundMembersLegacyV3(properties)
		}
		return walkCompoundMembersV3(properties, numMembers, compoundMemberOffsetSize(size), 0, false)
	default:
		return nil, 0, fmt.Errorf("unsupported compound datatype version: %d", version)
	}
}

// walkCompoundMembersLegacyV3 decodes version 3 compounds written by
// go-hdf5 <= v0.16.1: uint32 member count in the properties, 4-byte offsets.
// A spec-conformant compound with zero members has no properties at all.
func walkCompoundMembersLegacyV3(properties []byte) ([]CompoundMember, int, error) {
	if len(properties) < 4 {
		return []CompoundMember{}, 0, nil
	}
	count := binary.LittleEndian.Uint32(properties[0:4])
	// Every member needs at least a name terminator, an offset and a
	// datatype header, so the count is bounded by the available bytes.
	if uint64(count) > uint64(len(properties)) {
		return nil, 0, fmt.Errorf("compound member count %d exceeds properties length %d", count, len(properties))
	}
	return walkCompoundMembersV3(properties, int(count), 4, 4, true)
}

// walkCompoundMembersV3 decodes version 3 members starting at start. In
// legacy mode (go-hdf5 <= v0.16.1 output), string members carry the single
// padding/charset property byte that CreateBasicDatatypeMessage used to emit.
func walkCompoundMembersV3(properties []byte, numMembers, offsetSize, start int, legacy bool) ([]CompoundMember, int, error) {
	members := make([]CompoundMember, 0, numMembers)
	offset := start

	for i := 0; i < numMembers; i++ {
		// Member name (null-terminated, NOT padded in version 3).
		nameEnd := offset
		for nameEnd < len(properties) && properties[nameEnd] != 0 {
			nameEnd++
		}
		if nameEnd >= len(properties) {
			return nil, 0, fmt.Errorf("member %d: name not null-terminated (offset=%d, len=%d)", i, offset, len(properties))
		}
		name := string(properties[offset:nameEnd])
		offset = nameEnd + 1

		// Member byte offset (variable width, little-endian).
		if offset+offsetSize > len(properties) {
			return nil, 0, fmt.Errorf("member %d (%s): offset field truncated (offset=%d, len=%d)", i, name, offset, len(properties))
		}
		var memberOffset uint32
		for b := offsetSize - 1; b >= 0; b-- {
			memberOffset = memberOffset<<8 | uint32(properties[offset+b])
		}
		offset += offsetSize

		var memberType *DatatypeMessage
		var n int
		var err error
		if legacy && offset+9 <= len(properties) && DatatypeClass(properties[offset]&0x0F) == DatatypeString {
			memberType, err = ParseDatatypeMessage(properties[offset : offset+9])
			n = 9
		} else {
			memberType, n, err = parseInlineDatatype(properties[offset:])
		}
		if err != nil {
			return nil, 0, fmt.Errorf("member %d (%s): %w", i, name, err)
		}
		offset += n

		members = append(members, CompoundMember{Name: name, Offset: memberOffset, Type: memberType})
	}

	return members, offset, nil
}

// walkCompoundMembersV1V2 decodes version 1 and 2 members.
func walkCompoundMembersV1V2(properties []byte, version uint8, numMembers int) ([]CompoundMember, int, error) {
	members := make([]CompoundMember, 0, numMembers)
	offset := 0

	for i := 0; i < numMembers; i++ {
		// 1. Member name (null-terminated, padded to multiple of 8 bytes).
		nameStart := offset
		nameEnd := nameStart
		for nameEnd < len(properties) && properties[nameEnd] != 0 {
			nameEnd++
		}
		if nameEnd >= len(properties) {
			return nil, 0, fmt.Errorf("member %d name not null-terminated", i)
		}
		name := string(properties[nameStart:nameEnd])
		offset = nameStart + ((nameEnd-nameStart+8)/8)*8

		// 2. Member byte offset (uint32).
		if offset+4 > len(properties) {
			return nil, 0, fmt.Errorf("member %d: offset field truncated", i)
		}
		memberOffset := binary.LittleEndian.Uint32(properties[offset : offset+4])
		offset += 4

		// 3. Array info (version 1 only, always 28 bytes). Array members
		// are not supported; the base type is reported.
		if version == 1 {
			if offset+28 > len(properties) {
				return nil, 0, fmt.Errorf("member %d: array info truncated", i)
			}
			offset += 28
		}

		// 4. Member datatype (no padding between members).
		memberType, n, err := parseInlineDatatype(properties[offset:])
		if err != nil {
			return nil, 0, fmt.Errorf("member %d (%s): %w", i, name, err)
		}
		offset += n

		members = append(members, CompoundMember{Name: name, Offset: memberOffset, Type: memberType})
	}

	return members, offset, nil
}

// parseInlineDatatype parses a datatype embedded in another datatype
// (compound member, array/vlen/enum base type) and returns it together with
// its exact encoded length.
func parseInlineDatatype(data []byte) (*DatatypeMessage, int, error) {
	n, err := datatypeEncodedLen(data)
	if err != nil {
		return nil, 0, err
	}
	dt, err := ParseDatatypeMessage(data[:n])
	if err != nil {
		return nil, 0, fmt.Errorf("failed to parse datatype: %w", err)
	}
	// ParseDatatypeMessage may compute a different length for some
	// classes; the embedded encoding is exactly n bytes.
	dt.Properties = data[8:n]
	return dt, n, nil
}

// datatypeEncodedLen returns the exact encoded length (header + properties)
// of the datatype message at the start of data.
//
//nolint:gocyclo,cyclop // one case per datatype class
func datatypeEncodedLen(data []byte) (int, error) {
	if len(data) < 8 {
		return 0, errors.New("datatype header truncated")
	}
	classAndVersion := binary.LittleEndian.Uint32(data[0:4])
	class := DatatypeClass(classAndVersion & 0x0F)  //nolint:gosec // G115: 4-bit field
	version := uint8((classAndVersion >> 4) & 0x0F) //nolint:gosec // G115: 4-bit field
	classBitField := (classAndVersion >> 8) & 0x00FFFFFF
	size := binary.LittleEndian.Uint32(data[4:8])
	props := data[8:]

	var propsLen int
	switch class {
	case DatatypeFixed, DatatypeBitfield:
		propsLen = 4
	case DatatypeFloat:
		propsLen = 12
	case DatatypeTime:
		propsLen = 2
	case DatatypeString:
		propsLen = 0
	case DatatypeOpaque:
		propsLen = int(classBitField & 0xFF)
	case DatatypeReference:
		if version >= 4 {
			return 0, errors.New("inline revised reference datatypes are not supported")
		}
		propsLen = 0
	case DatatypeCompound:
		_, n, err := walkCompoundMembers(props, version, classBitField, size)
		if err != nil {
			return 0, err
		}
		propsLen = n
	case DatatypeVarLen:
		n, err := datatypeEncodedLen(props)
		if err != nil {
			return 0, fmt.Errorf("vlen base type: %w", err)
		}
		propsLen = n
	case DatatypeArray:
		n, err := arrayPropsLen(props, version)
		if err != nil {
			return 0, err
		}
		propsLen = n
	case DatatypeEnum:
		n, err := enumPropsLen(props, version, classBitField)
		if err != nil {
			return 0, err
		}
		propsLen = n
	default:
		return 0, fmt.Errorf("cannot determine encoded length of datatype class %d", class)
	}

	if propsLen < 0 || propsLen > len(props) {
		return 0, fmt.Errorf("datatype class %d truncated (need %d property bytes, have %d)", class, propsLen, len(props))
	}
	return 8 + propsLen, nil
}

// arrayPropsLen returns the property length of an array datatype: the
// dimension header followed by the base type.
func arrayPropsLen(props []byte, version uint8) (int, error) {
	if len(props) < 1 {
		return 0, errors.New("array datatype truncated")
	}
	ndims := int(props[0])
	hdr := 1 + 4*ndims // version 3+: ndims + dimension sizes
	if version < 3 {
		hdr = 4 + 8*ndims // ndims + reserved(3) + dimension sizes + permutation
	}
	if hdr > len(props) {
		return 0, errors.New("array datatype truncated")
	}
	n, err := datatypeEncodedLen(props[hdr:])
	if err != nil {
		return 0, fmt.Errorf("array base type: %w", err)
	}
	return hdr + n, nil
}

// enumPropsLen returns the property length of an enumeration datatype: base
// type, member names (padded to 8 bytes before version 3) and values.
func enumPropsLen(props []byte, version uint8, classBitField uint32) (int, error) {
	baseLen, err := datatypeEncodedLen(props)
	if err != nil {
		return 0, fmt.Errorf("enum base type: %w", err)
	}
	baseSize := int(binary.LittleEndian.Uint32(props[4:8]))
	numMembers := int(classBitField & 0xFFFF)
	off := baseLen
	for i := 0; i < numMembers; i++ {
		end := off
		for end < len(props) && props[end] != 0 {
			end++
		}
		if end >= len(props) {
			return 0, fmt.Errorf("enum member %d name not null-terminated", i)
		}
		if version < 3 {
			off += ((end - off + 8) / 8) * 8
		} else {
			off = end + 1
		}
	}
	return off + numMembers*baseSize, nil
}

// String returns human-readable compound type description.
func (ct *CompoundType) String() string {
	result := fmt.Sprintf("compound{size=%d, members=[", ct.Size)
	for i, member := range ct.Members {
		if i > 0 {
			result += ", "
		}
		result += fmt.Sprintf("%s:%s@%d", member.Name, member.Type.String(), member.Offset)
	}
	result += "]}"
	return result
}
