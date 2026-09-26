package core

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestTrackedHeaderRoundTrip checks that a header tracking attribute
// creation order keeps message flags and creation indexes through writing,
// reading, adding attributes and rewriting in place (also when messages
// spill into a continuation chunk).
func TestTrackedHeaderRoundTrip(t *testing.T) {
	sb := &Superblock{Version: 2, OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}
	ohw := &ObjectHeaderWriter{
		Version: 2,
		Messages: []MessageWriter{
			{Type: MsgDataspace, Flags: 0x01, Data: []byte{2, 0, 0, 0}}, // constant
			{Type: MsgAttribute, Data: []byte("first")},
			{Type: MsgAttribute, Data: []byte("second")},
		},
	}
	require.NoError(t, ohw.TrackAttributeCreationOrder(sb))

	rw := &mockReaderWriterAt{data: make([]byte, 16384)}
	const addr = 64
	_, err := ohw.WriteTo(rw, addr)
	require.NoError(t, err)

	oh, err := ReadObjectHeader(rw, addr, sb)
	require.NoError(t, err)
	require.Equal(t, ObjectHeaderAttrCreationOrderTracked, oh.Flags&ObjectHeaderAttrCreationOrderTracked)
	require.Len(t, oh.Messages, 4)
	require.Equal(t, uint8(0x01), oh.Messages[0].Flags)
	require.Equal(t, MsgAttributeInfo, oh.Messages[1].Type, "Attribute Info goes before the attributes")
	info, err := ParseAttributeInfoMessage(oh.Messages[1].Data, sb)
	require.NoError(t, err)
	require.Equal(t, AttributeInfoTrackCreationOrder, info.Flags)
	require.Equal(t, uint64(2), info.MaxCreationIndex)
	require.Equal(t, []uint16{0, 1}, attributeCrtIdxs(oh))

	// Add attributes until they no longer fit chunk #0.
	for range 20 {
		require.NoError(t, AddMessageToObjectHeader(oh, MsgAttribute, make([]byte, 40)))
	}
	require.NoError(t, WriteObjectHeader(rw, addr, oh, sb))

	oh, err = ReadObjectHeader(rw, addr, sb)
	require.NoError(t, err)
	conts := 0
	for _, m := range oh.Messages {
		if m.Type == MsgContinuation {
			conts++
		}
	}
	require.Equal(t, 1, conts, "the added attributes must spill into a continuation chunk")
	want := make([]uint16, 22)
	for i := range want {
		want[i] = uint16(i)
	}
	require.Equal(t, want, attributeCrtIdxs(oh))
	require.Equal(t, uint8(0x01), oh.Messages[0].Flags, "message flags must survive a rewrite")
	for _, m := range oh.Messages {
		if m.Type == MsgAttributeInfo {
			info, err := ParseAttributeInfoMessage(m.Data, sb)
			require.NoError(t, err)
			require.Equal(t, uint64(22), info.MaxCreationIndex)
		}
	}
}

// attributeCrtIdxs returns the creation indexes of the attribute messages.
func attributeCrtIdxs(oh *ObjectHeader) []uint16 {
	var idx []uint16
	for _, m := range oh.Messages {
		if m.Type == MsgAttribute {
			idx = append(idx, m.CrtIdx)
		}
	}
	return idx
}
