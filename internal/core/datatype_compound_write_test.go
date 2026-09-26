package core

import (
	"encoding/binary"
	"testing"
)

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// TestEncodeCompoundDatatypeV3_Simple tests encoding a simple compound with 3 fields.
func TestEncodeCompoundDatatypeV3_Simple(t *testing.T) {
	// Create a simple compound: { int32, float64, int64 }
	// Offsets: 0, 4, 12 (total size: 20 bytes)
	int32Type, err := CreateBasicDatatypeMessage(DatatypeFixed, 4)
	if err != nil {
		t.Fatalf("Failed to create int32 type: %v", err)
	}

	float64Type, err := CreateBasicDatatypeMessage(DatatypeFloat, 8)
	if err != nil {
		t.Fatalf("Failed to create float64 type: %v", err)
	}

	int64Type, err := CreateBasicDatatypeMessage(DatatypeFixed, 8)
	if err != nil {
		t.Fatalf("Failed to create int64 type: %v", err)
	}

	fields := []CompoundFieldDef{
		{Name: "field1", Offset: 0, Type: int32Type},
		{Name: "field2", Offset: 4, Type: float64Type},
		{Name: "field3", Offset: 12, Type: int64Type},
	}

	encoded, err := EncodeCompoundDatatypeV3(20, fields)
	if err != nil {
		t.Fatalf("EncodeCompoundDatatypeV3 failed: %v", err)
	}

	t.Logf("Encoded length: %d bytes", len(encoded))
	if len(encoded) >= 12 {
		t.Logf("Header: class=%d, size=%d", encoded[0]&0x0F, binary.LittleEndian.Uint32(encoded[4:8]))
		t.Logf("Member count in class bit field: %d", binary.LittleEndian.Uint16(encoded[1:3]))
	}

	// Parse and validate
	dt, err := ParseDatatypeMessage(encoded)
	if err != nil {
		t.Fatalf("ParseDatatypeMessage failed: %v (encoded len=%d, first 32 bytes=%#v)", err, len(encoded), encoded[:minInt(32, len(encoded))])
	}

	t.Logf("Parsed: Class=%d, Version=%d, Size=%d, Props len=%d", dt.Class, dt.Version, dt.Size, len(dt.Properties))

	if dt.Class != DatatypeCompound {
		t.Errorf("Expected class Compound (6), got %d", dt.Class)
	}

	if dt.Version != 3 {
		t.Errorf("Expected version 3, got %d", dt.Version)
	}

	if dt.Size != 20 {
		t.Errorf("Expected size 20, got %d", dt.Size)
	}

	// Parse compound members
	t.Logf("About to parse compound members from %d property bytes", len(dt.Properties))
	compound, err := ParseCompoundType(dt)
	if err != nil {
		// Debug: show where parsing stopped
		t.Logf("Properties (first 60 bytes): %#v", dt.Properties[:minInt(60, len(dt.Properties))])
		t.Fatalf("ParseCompoundType failed: %v", err)
	}

	if len(compound.Members) != 3 {
		t.Fatalf("Expected 3 members, got %d", len(compound.Members))
	}

	// Verify member 1
	if compound.Members[0].Name != "field1" {
		t.Errorf("Member 0 name: expected 'field1', got '%s'", compound.Members[0].Name)
	}
	if compound.Members[0].Offset != 0 {
		t.Errorf("Member 0 offset: expected 0, got %d", compound.Members[0].Offset)
	}
	if compound.Members[0].Type.Size != 4 {
		t.Errorf("Member 0 size: expected 4, got %d", compound.Members[0].Type.Size)
	}

	// Verify member 2
	if compound.Members[1].Name != "field2" {
		t.Errorf("Member 1 name: expected 'field2', got '%s'", compound.Members[1].Name)
	}
	if compound.Members[1].Offset != 4 {
		t.Errorf("Member 1 offset: expected 4, got %d", compound.Members[1].Offset)
	}
	if compound.Members[1].Type.Size != 8 {
		t.Errorf("Member 1 size: expected 8, got %d", compound.Members[1].Type.Size)
	}

	// Verify member 3
	if compound.Members[2].Name != "field3" {
		t.Errorf("Member 2 name: expected 'field3', got '%s'", compound.Members[2].Name)
	}
	if compound.Members[2].Offset != 12 {
		t.Errorf("Member 2 offset: expected 12, got %d", compound.Members[2].Offset)
	}
	if compound.Members[2].Type.Size != 8 {
		t.Errorf("Member 2 size: expected 8, got %d", compound.Members[2].Type.Size)
	}
}

// TestEncodeCompoundDatatypeV3_WithString tests compound with string field.
func TestEncodeCompoundDatatypeV3_WithString(t *testing.T) {
	// Compound: { int32, string[10] }
	fields := []CompoundFieldDef{
		{
			Name:   "id",
			Offset: 0,
			Type: &DatatypeMessage{
				Class:      DatatypeFixed,
				Version:    1,
				Size:       4,
				Properties: []byte{0, 0, 32, 0}, // bit offset 0, precision 32
			},
		},
		{
			Name:   "name",
			Offset: 4,
			Type: &DatatypeMessage{
				Class:   DatatypeString,
				Version: 1,
				Size:    10, // strings carry no properties
			},
		},
	}

	encoded, err := EncodeCompoundDatatypeV3(14, fields)
	if err != nil {
		t.Fatalf("EncodeCompoundDatatypeV3 failed: %v", err)
	}

	// Round-trip validation
	dt, err := ParseDatatypeMessage(encoded)
	if err != nil {
		t.Fatalf("ParseDatatypeMessage failed: %v", err)
	}

	compound, err := ParseCompoundType(dt)
	if err != nil {
		t.Fatalf("ParseCompoundType failed: %v", err)
	}

	if len(compound.Members) != 2 {
		t.Fatalf("Expected 2 members, got %d", len(compound.Members))
	}

	if compound.Members[1].Type.Class != DatatypeString {
		t.Errorf("Expected string type for member 1, got class %d", compound.Members[1].Type.Class)
	}
}

// TestEncodeCompoundDatatypeV3_NestedCompound tests nested compound types.
func TestEncodeCompoundDatatypeV3_NestedCompound(t *testing.T) {
	// Inner compound: { float32, float32 } (8 bytes)
	// Must use CreateBasicDatatypeMessage to populate Properties
	floatType, err := CreateBasicDatatypeMessage(DatatypeFloat, 4)
	if err != nil {
		t.Fatalf("Failed to create float type: %v", err)
	}

	innerFields := []CompoundFieldDef{
		{
			Name:   "x",
			Offset: 0,
			Type:   floatType,
		},
		{
			Name:   "y",
			Offset: 4,
			Type:   floatType,
		},
	}

	innerEncoded, err := EncodeCompoundDatatypeV3(8, innerFields)
	if err != nil {
		t.Fatalf("Failed to encode inner compound: %v", err)
	}

	innerDt, err := ParseDatatypeMessage(innerEncoded)
	if err != nil {
		t.Fatalf("Failed to parse inner compound: %v", err)
	}

	// Outer compound: { int32, Point, int32 } (16 bytes)
	int32Type, err := CreateBasicDatatypeMessage(DatatypeFixed, 4)
	if err != nil {
		t.Fatalf("Failed to create int32 type: %v", err)
	}

	outerFields := []CompoundFieldDef{
		{
			Name:   "id",
			Offset: 0,
			Type:   int32Type,
		},
		{
			Name:   "point",
			Offset: 4,
			Type:   innerDt, // Nested compound!
		},
		{
			Name:   "count",
			Offset: 12,
			Type:   int32Type,
		},
	}

	outerEncoded, err := EncodeCompoundDatatypeV3(16, outerFields)
	if err != nil {
		t.Fatalf("Failed to encode outer compound: %v", err)
	}

	// Parse and validate nested structure
	outerDt, err := ParseDatatypeMessage(outerEncoded)
	if err != nil {
		t.Fatalf("ParseDatatypeMessage failed: %v", err)
	}

	outerCompound, err := ParseCompoundType(outerDt)
	if err != nil {
		t.Fatalf("ParseCompoundType failed: %v", err)
	}

	if len(outerCompound.Members) != 3 {
		t.Fatalf("Expected 3 members in outer compound, got %d", len(outerCompound.Members))
	}

	// Check nested compound member
	nestedMember := outerCompound.Members[1]
	if nestedMember.Name != "point" {
		t.Errorf("Nested member name: expected 'point', got '%s'", nestedMember.Name)
	}
	if nestedMember.Type.Class != DatatypeCompound {
		t.Errorf("Nested member should be compound, got class %d", nestedMember.Type.Class)
	}

	// Parse inner compound
	innerCompound, err := ParseCompoundType(nestedMember.Type)
	if err != nil {
		t.Fatalf("Failed to parse nested compound: %v", err)
	}

	if len(innerCompound.Members) != 2 {
		t.Fatalf("Expected 2 members in nested compound, got %d", len(innerCompound.Members))
	}
}

// TestEncodeCompoundDatatypeV1_Simple tests version 1 encoding.
func TestEncodeCompoundDatatypeV1_Simple(t *testing.T) {
	// Simple compound: { int32, float64 }
	// Must use CreateBasicDatatypeMessage to populate Properties
	int32Type, err := CreateBasicDatatypeMessage(DatatypeFixed, 4)
	if err != nil {
		t.Fatalf("Failed to create int32 type: %v", err)
	}

	float64Type, err := CreateBasicDatatypeMessage(DatatypeFloat, 8)
	if err != nil {
		t.Fatalf("Failed to create float64 type: %v", err)
	}

	fields := []CompoundFieldDef{
		{
			Name:   "field1",
			Offset: 0,
			Type:   int32Type,
		},
		{
			Name:   "field2",
			Offset: 4,
			Type:   float64Type,
		},
	}

	encoded, err := EncodeCompoundDatatypeV1(12, fields)
	if err != nil {
		t.Fatalf("EncodeCompoundDatatypeV1 failed: %v", err)
	}

	// Parse and validate
	dt, err := ParseDatatypeMessage(encoded)
	if err != nil {
		t.Fatalf("ParseDatatypeMessage failed: %v", err)
	}

	if dt.Class != DatatypeCompound {
		t.Errorf("Expected class Compound (6), got %d", dt.Class)
	}

	if dt.Version != 1 {
		t.Errorf("Expected version 1, got %d", dt.Version)
	}

	if dt.Size != 12 {
		t.Errorf("Expected size 12, got %d", dt.Size)
	}

	// Verify member count in ClassBitField (bits 0-15)
	numMembers := uint16(dt.ClassBitField & 0xFFFF)
	if numMembers != 2 {
		t.Errorf("Expected 2 members in ClassBitField, got %d", numMembers)
	}

	// Parse compound members
	compound, err := ParseCompoundType(dt)
	if err != nil {
		t.Fatalf("ParseCompoundType failed: %v", err)
	}

	if len(compound.Members) != 2 {
		t.Fatalf("Expected 2 members, got %d", len(compound.Members))
	}
}

// TestEncodeCompoundDatatypeV1_NamePadding tests version 1 name padding to 8-byte boundary.
func TestEncodeCompoundDatatypeV1_NamePadding(t *testing.T) {
	// Test different name lengths to verify 8-byte padding
	testCases := []struct {
		name           string
		expectedPadLen int
	}{
		{"a", 8},         // 1 char + null = 2, padded to 8
		{"abc", 8},       // 3 chars + null = 4, padded to 8
		{"abcdefg", 8},   // 7 chars + null = 8, already aligned
		{"abcdefgh", 16}, // 8 chars + null = 9, padded to 16
	}

	for _, tc := range testCases {
		int32Type, err := CreateBasicDatatypeMessage(DatatypeFixed, 4)
		if err != nil {
			t.Fatalf("Failed to create int32 type: %v", err)
		}

		fields := []CompoundFieldDef{
			{
				Name:   tc.name,
				Offset: 0,
				Type:   int32Type,
			},
		}

		encoded, err := EncodeCompoundDatatypeV1(4, fields)
		if err != nil {
			t.Fatalf("EncodeCompoundDatatypeV1 failed for name '%s': %v", tc.name, err)
		}

		// Header is 8 bytes, then comes name (padded) + offset (4) + array info (28) + member type
		// Name section starts at byte 8
		nameSection := encoded[8:]

		// Find null terminator
		nullPos := 0
		for i, b := range nameSection {
			if b == 0 {
				nullPos = i
				break
			}
		}

		actualNameLen := nullPos // Name without null terminator
		if actualNameLen != len(tc.name) {
			t.Errorf("Name '%s': expected length %d, got %d", tc.name, len(tc.name), actualNameLen)
		}

		// The next non-zero data should be at offset = tc.expectedPadLen + 4 (after array info and offset)
		// (name + padding + offset field + array info)
		// Round-trip parse to verify
		dt, err := ParseDatatypeMessage(encoded)
		if err != nil {
			t.Fatalf("Parse failed for name '%s': %v", tc.name, err)
		}

		compound, err := ParseCompoundType(dt)
		if err != nil {
			t.Fatalf("ParseCompoundType failed for name '%s': %v", tc.name, err)
		}

		if compound.Members[0].Name != tc.name {
			t.Errorf("Name mismatch: expected '%s', got '%s'", tc.name, compound.Members[0].Name)
		}
	}
}

// TestEncodeCompoundDatatypeV3_ErrorCases tests error handling.
func TestEncodeCompoundDatatypeV3_ErrorCases(t *testing.T) {
	tests := []struct {
		name      string
		totalSize uint32
		fields    []CompoundFieldDef
		wantErr   string
	}{
		{
			name:      "no fields",
			totalSize: 8,
			fields:    []CompoundFieldDef{},
			wantErr:   "at least one field",
		},
		{
			name:      "zero size",
			totalSize: 0,
			fields: []CompoundFieldDef{
				{Name: "field1", Offset: 0, Type: &DatatypeMessage{Class: DatatypeFixed, Size: 4}},
			},
			wantErr: "size cannot be 0",
		},
		{
			name:      "empty field name",
			totalSize: 4,
			fields: []CompoundFieldDef{
				{Name: "", Offset: 0, Type: &DatatypeMessage{Class: DatatypeFixed, Size: 4}},
			},
			wantErr: "name cannot be empty",
		},
		{
			name:      "nil field type",
			totalSize: 4,
			fields: []CompoundFieldDef{
				{Name: "field1", Offset: 0, Type: nil},
			},
			wantErr: "type cannot be nil",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := EncodeCompoundDatatypeV3(tt.totalSize, tt.fields)
			if err == nil {
				t.Fatalf("Expected error containing '%s', got nil", tt.wantErr)
			}
			if err.Error() == "" || tt.wantErr == "" {
				t.Fatalf("Error message empty: %v", err)
			}
			// Check error contains expected substring
			errMsg := err.Error()
			found := false
			for i := 0; i <= len(errMsg)-len(tt.wantErr); i++ {
				if errMsg[i:i+len(tt.wantErr)] == tt.wantErr {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Error message '%s' does not contain '%s'", errMsg, tt.wantErr)
			}
		})
	}
}

// TestCreateCompoundTypeFromFields tests convenience function.
func TestCreateCompoundTypeFromFields(t *testing.T) {
	// Create compound with automatic validation
	fields := []CompoundFieldDef{
		{
			Name:   "field1",
			Offset: 0,
			Type: &DatatypeMessage{
				Class:      DatatypeFixed,
				Version:    1,
				Size:       4,
				Properties: []byte{0, 0, 32, 0}, // bit offset 0, precision 32
			},
		},
		{
			Name:   "field2",
			Offset: 4,
			Type:   mustBasicType(t, DatatypeFloat, 8),
		},
	}

	dt, err := CreateCompoundTypeFromFields(fields)
	if err != nil {
		t.Fatalf("CreateCompoundTypeFromFields failed: %v", err)
	}

	if dt.Class != DatatypeCompound {
		t.Errorf("Expected Compound class, got %d", dt.Class)
	}

	if dt.Size != 12 {
		t.Errorf("Expected size 12 (4+8), got %d", dt.Size)
	}

	// Verify round-trip
	compound, err := ParseCompoundType(dt)
	if err != nil {
		t.Fatalf("ParseCompoundType failed: %v", err)
	}

	if len(compound.Members) != 2 {
		t.Fatalf("Expected 2 members, got %d", len(compound.Members))
	}
}

// TestCompoundDatatypeEncodeDecode_Binary tests binary format correctness
// against the HDF5 File Format Specification (datatype message, class 6,
// version 3): member count in class bit field bits 0-15, unpadded names,
// member offsets as wide as needed to encode the compound size.
func TestCompoundDatatypeEncodeDecode_Binary(t *testing.T) {
	fields := []CompoundFieldDef{
		{Name: "x", Offset: 0, Type: mustBasicType(t, DatatypeFixed, 4)},
		{Name: "y", Offset: 4, Type: mustBasicType(t, DatatypeFloat, 8)},
	}

	encoded, err := EncodeCompoundDatatypeV3(12, fields)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	want := []byte{
		0x36, 0x02, 0x00, 0x00, // class 6, version 3, 2 members
		0x0C, 0x00, 0x00, 0x00, // size 12
		'x', 0, // name
		0x00,                   // offset 0 (1 byte: size 12 <= 255)
		0x10, 0x08, 0x00, 0x00, // int, version 1, signed LE
		0x04, 0x00, 0x00, 0x00, // size 4
		0x00, 0x00, 0x20, 0x00, // bit offset 0, precision 32
		'y', 0, // name
		0x04,                   // offset 4
		0x11, 0x20, 0x3F, 0x00, // float, version 1, LE, implied mantissa, sign bit 63
		0x08, 0x00, 0x00, 0x00, // size 8
		0x00, 0x00, 0x40, 0x00, // bit offset 0, precision 64
		52, 11, 0, 52, // exponent loc/size, mantissa loc/size
		0xFF, 0x03, 0x00, 0x00, // exponent bias 1023
	}
	if len(encoded) != len(want) {
		t.Fatalf("encoded length %d, want %d: %#v", len(encoded), len(want), encoded)
	}
	for i := range want {
		if encoded[i] != want[i] {
			t.Fatalf("byte %d = %#x, want %#x (encoded %#v)", i, encoded[i], want[i], encoded)
		}
	}
}

// TestCompoundMemberOffsetSize checks the v3 member offset width.
func TestCompoundMemberOffsetSize(t *testing.T) {
	cases := map[uint32]int{1: 1, 255: 1, 256: 2, 65535: 2, 65536: 3, 1 << 24: 4, 0xFFFFFFFF: 4}
	for size, want := range cases {
		if got := compoundMemberOffsetSize(size); got != want {
			t.Errorf("compoundMemberOffsetSize(%d) = %d, want %d", size, got, want)
		}
	}
}

// TestEncodeCompoundDatatypeV3_WideOffsets round-trips a compound whose size
// needs 2- and 3-byte member offsets.
func TestEncodeCompoundDatatypeV3_WideOffsets(t *testing.T) {
	for _, total := range []uint32{300, 70000} {
		str, err := CreateBasicDatatypeMessage(DatatypeString, total-8)
		if err != nil {
			t.Fatal(err)
		}
		fields := []CompoundFieldDef{
			{Name: "s", Offset: 0, Type: str},
			{Name: "v", Offset: total - 8, Type: mustBasicType(t, DatatypeFloat, 8)},
		}
		encoded, err := EncodeCompoundDatatypeV3(total, fields)
		if err != nil {
			t.Fatal(err)
		}
		dt, err := ParseDatatypeMessage(encoded)
		if err != nil {
			t.Fatal(err)
		}
		if len(dt.Properties) != len(encoded)-8 {
			t.Fatalf("size %d: properties length %d, want %d", total, len(dt.Properties), len(encoded)-8)
		}
		ct, err := ParseCompoundType(dt)
		if err != nil {
			t.Fatal(err)
		}
		if len(ct.Members) != 2 || ct.Members[1].Offset != total-8 || ct.Members[0].Type.Size != total-8 {
			t.Fatalf("size %d: unexpected members %s", total, ct)
		}
	}
}

// TestParseCompoundType_LegacyGoHDF5V3 parses the version 3 layout written by
// go-hdf5 <= v0.16.1 (uint32 member count in the properties, 4-byte offsets,
// 1-byte string properties).
func TestParseCompoundType_LegacyGoHDF5V3(t *testing.T) {
	props := []byte{
		2, 0, 0, 0, // member count
		'i', 'd', 0,
		0, 0, 0, 0, // offset 0
		0x10, 0, 0, 0, 4, 0, 0, 0, // int32 header
		0, 32, 0, 0, // legacy (wrong) integer properties
		'n', 0,
		4, 0, 0, 0, // offset 4
		0x13, 0, 0, 0, 6, 0, 0, 0, // string[6] header
		0, // legacy string property byte
	}
	msg := append([]byte{0x36, 0, 0, 0, 10, 0, 0, 0}, props...)
	dt, err := ParseDatatypeMessage(msg)
	if err != nil {
		t.Fatal(err)
	}
	ct, err := ParseCompoundType(dt)
	if err != nil {
		t.Fatal(err)
	}
	if len(ct.Members) != 2 || ct.Members[0].Name != "id" || ct.Members[1].Name != "n" ||
		ct.Members[1].Offset != 4 || ct.Members[1].Type.Class != DatatypeString || ct.Members[1].Type.Size != 6 {
		t.Fatalf("unexpected members: %s", ct)
	}
}

// TestParseCompoundType_Version2 parses a version 2 compound (padded names,
// 4-byte offsets, no array info).
func TestParseCompoundType_Version2(t *testing.T) {
	props := []byte{
		'a', 'b', 'c', 0, 0, 0, 0, 0, // name padded to 8
		0, 0, 0, 0, // offset 0
		0x10, 0x08, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0, // int16
		'l', 'o', 'n', 'g', 'n', 'a', 'm', 'e', 0, 0, 0, 0, 0, 0, 0, 0, // 8 chars + NUL, padded to 16
		2, 0, 0, 0, // offset 2
		0x10, 0x08, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0, // int16
	}
	msg := append([]byte{0x26, 2, 0, 0, 4, 0, 0, 0}, props...)
	msg = append(msg, 0xAA, 0xBB) // trailing message padding must be ignored
	dt, err := ParseDatatypeMessage(msg)
	if err != nil {
		t.Fatal(err)
	}
	if len(dt.Properties) != len(props) {
		t.Fatalf("properties length %d, want %d", len(dt.Properties), len(props))
	}
	ct, err := ParseCompoundType(dt)
	if err != nil {
		t.Fatal(err)
	}
	if len(ct.Members) != 2 || ct.Members[1].Name != "longname" || ct.Members[1].Offset != 2 {
		t.Fatalf("unexpected members: %s", ct)
	}
}

// TestEncodeCompoundDatatypeV3_RejectsBadMemberTypes checks member validation.
func TestEncodeCompoundDatatypeV3_RejectsBadMemberTypes(t *testing.T) {
	bad := []CompoundFieldDef{{Name: "x", Offset: 0, Type: &DatatypeMessage{Class: DatatypeFixed, Version: 1, Size: 4}}}
	if _, err := EncodeCompoundDatatypeV3(4, bad); err == nil {
		t.Error("expected error for integer member without properties")
	}
	overflow := []CompoundFieldDef{{Name: "x", Offset: 2, Type: mustBasicType(t, DatatypeFixed, 4)}}
	if _, err := EncodeCompoundDatatypeV3(4, overflow); err == nil {
		t.Error("expected error for member beyond compound size")
	}
}

// TestCreateBasicDatatypeMessage checks the spec layout of basic member types.
func TestCreateBasicDatatypeMessage(t *testing.T) {
	i16 := mustBasicType(t, DatatypeFixed, 2)
	if i16.ClassBitField != 0x08 || binary.LittleEndian.Uint16(i16.Properties[0:2]) != 0 ||
		binary.LittleEndian.Uint16(i16.Properties[2:4]) != 16 {
		t.Errorf("int16: bitfield %#x, properties %v", i16.ClassBitField, i16.Properties)
	}
	f32 := mustBasicType(t, DatatypeFloat, 4)
	if f32.ClassBitField != 0x1F20 || len(f32.Properties) != 12 ||
		binary.LittleEndian.Uint16(f32.Properties[2:4]) != 32 || f32.Properties[4] != 23 ||
		f32.Properties[5] != 8 || f32.Properties[7] != 23 || binary.LittleEndian.Uint32(f32.Properties[8:12]) != 127 {
		t.Errorf("float32: bitfield %#x, properties %v", f32.ClassBitField, f32.Properties)
	}
	s := mustBasicType(t, DatatypeString, 7)
	if len(s.Properties) != 0 || s.Size != 7 {
		t.Errorf("string: %+v", s)
	}
	for _, c := range []struct {
		class DatatypeClass
		size  uint32
	}{{DatatypeFixed, 3}, {DatatypeFloat, 2}, {DatatypeString, 0}, {DatatypeEnum, 4}} {
		if _, err := CreateBasicDatatypeMessage(c.class, c.size); err == nil {
			t.Errorf("class %d size %d: expected error", c.class, c.size)
		}
	}
}

func mustBasicType(t *testing.T, class DatatypeClass, size uint32) *DatatypeMessage {
	t.Helper()
	dt, err := CreateBasicDatatypeMessage(class, size)
	if err != nil {
		t.Fatalf("CreateBasicDatatypeMessage(%d, %d): %v", class, size, err)
	}
	return dt
}
