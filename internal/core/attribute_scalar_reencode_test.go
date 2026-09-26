package core

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestEncodeAttributeFromStructKeepsParsedScalar re-encodes a parsed scalar
// attribute (whose dataspace reads as one element) and checks it is still
// scalar, as when attributes move from compact to dense storage.
func TestEncodeAttributeFromStructKeepsParsedScalar(t *testing.T) {
	msg, err := EncodeAttributeMessage("n",
		&DatatypeMessage{Class: DatatypeFixed, Version: 1, Size: 4},
		&DataspaceMessage{Type: DataspaceScalar}, []byte{7, 0, 0, 0})
	require.NoError(t, err)
	parsed, err := ParseAttributeMessage(msg, binary.LittleEndian)
	require.NoError(t, err)
	require.Equal(t, DataspaceScalar, parsed.Dataspace.Type)

	again, err := EncodeAttributeFromStruct(parsed, nil)
	require.NoError(t, err)
	require.Equal(t, msg, again)

	// A one-element simple dataspace stays simple.
	msg, err = EncodeAttributeMessage("v",
		&DatatypeMessage{Class: DatatypeFixed, Version: 1, Size: 4},
		&DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{1}}, []byte{7, 0, 0, 0})
	require.NoError(t, err)
	parsed, err = ParseAttributeMessage(msg, binary.LittleEndian)
	require.NoError(t, err)
	again, err = EncodeAttributeFromStruct(parsed, nil)
	require.NoError(t, err)
	require.Equal(t, msg, again)
}

// TestScalarAttributeMatchesNetCDFLayout checks that a scalar string
// attribute is encoded like netCDF-C writes it: attribute message v3 with an
// 8-byte string datatype and a 4-byte version 2 scalar dataspace. libmysofa
// reads dense attributes only in exactly this layout.
func TestScalarAttributeMatchesNetCDFLayout(t *testing.T) {
	msg, err := EncodeAttributeMessage("Conventions",
		&DatatypeMessage{Class: DatatypeString, Version: 1, Size: 4},
		&DataspaceMessage{Type: DataspaceScalar}, []byte("SOFA"))
	require.NoError(t, err)

	require.Equal(t, []byte{3, 0}, msg[0:2], "version 3, no flags")
	require.Equal(t, uint16(12), binary.LittleEndian.Uint16(msg[2:4]), "name size")
	require.Equal(t, uint16(8), binary.LittleEndian.Uint16(msg[4:6]), "datatype size")
	require.Equal(t, uint16(4), binary.LittleEndian.Uint16(msg[6:8]), "dataspace size")
	dataspace := msg[9+12+8 : 9+12+8+4]
	require.Equal(t, []byte{2, 0, 0, 0}, dataspace, "version 2 scalar dataspace")
	require.Equal(t, []byte("SOFA"), msg[len(msg)-4:])

	parsed, err := ParseAttributeMessage(msg, binary.LittleEndian)
	require.NoError(t, err)
	require.Equal(t, DataspaceScalar, parsed.Dataspace.Type)
	require.Equal(t, []byte("SOFA"), parsed.Data)
}
