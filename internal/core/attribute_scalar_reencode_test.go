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
