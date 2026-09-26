package core

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGroupInfoMessageRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		msg  GroupInfoMessage
		size int
	}{
		{"defaults", GroupInfoMessage{}, 2},
		{"link phase change", GroupInfoMessage{Flags: GroupInfoLinkPhaseChange, MaxCompact: 16, MinDense: 4}, 6},
		{"entry estimates", GroupInfoMessage{Flags: GroupInfoEntryEstimates, EstNumEntries: 10, EstNameLen: 12}, 6},
		{"both", GroupInfoMessage{
			Flags:      GroupInfoLinkPhaseChange | GroupInfoEntryEstimates,
			MaxCompact: 8, MinDense: 6, EstNumEntries: 4, EstNameLen: 8,
		}, 10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := EncodeGroupInfoMessage(&tt.msg)
			require.Len(t, data, tt.size)
			got, err := ParseGroupInfoMessage(data)
			require.NoError(t, err)
			require.Equal(t, tt.msg, *got)
		})
	}
}

func TestGroupInfoMessageDefaultsMatchLibhdf5(t *testing.T) {
	// H5G_CRT_GINFO_DEF: version 0, no flags, no optional fields.
	require.Equal(t, []byte{0, 0}, EncodeGroupInfoMessage(&GroupInfoMessage{}))

	m, err := ParseGroupInfoMessage([]byte{0, 0})
	require.NoError(t, err)
	require.Equal(t, uint16(8), m.MaxCompactLinks())
	require.Equal(t, uint16(6), m.MinDenseLinks())
}

func TestParseGroupInfoMessageErrors(t *testing.T) {
	for _, data := range [][]byte{
		nil,
		{0},
		{1, 0},                             // unknown version
		{0, 0x04},                          // unknown flag
		{0, GroupInfoLinkPhaseChange, 8},   // truncated phase change values
		{0, GroupInfoEntryEstimates, 4, 0}, // truncated estimates
	} {
		_, err := ParseGroupInfoMessage(data)
		require.Error(t, err, "data % x", data)
	}
}
