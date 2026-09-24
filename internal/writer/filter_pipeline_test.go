package writer

import (
	"encoding/binary"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

// mockFilter is a test filter that transforms data in a predictable way.
type mockFilter struct {
	id         FilterID
	name       string
	flags      uint16
	cdValues   []uint32
	shouldFail bool
}

func (m *mockFilter) ID() FilterID {
	return m.id
}

func (m *mockFilter) Name() string {
	return m.name
}

func (m *mockFilter) Apply(data []byte) ([]byte, error) {
	if m.shouldFail {
		return nil, errors.New("mock filter apply failed")
	}
	// Simple transformation: add filter ID to each byte
	result := make([]byte, len(data))
	for i, b := range data {
		result[i] = b + byte(m.id)
	}
	return result, nil
}

func (m *mockFilter) Remove(data []byte) ([]byte, error) {
	if m.shouldFail {
		return nil, errors.New("mock filter remove failed")
	}
	// Reverse transformation: subtract filter ID from each byte
	result := make([]byte, len(data))
	for i, b := range data {
		result[i] = b - byte(m.id)
	}
	return result, nil
}

func (m *mockFilter) Encode() (flags uint16, cdValues []uint32) {
	return m.flags, m.cdValues
}

func TestNewFilterPipeline(t *testing.T) {
	pipeline := NewFilterPipeline()
	require.NotNil(t, pipeline)
	require.True(t, pipeline.IsEmpty())
	require.Equal(t, 0, pipeline.Count())
}

func TestFilterPipeline_AddFilter(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter := &mockFilter{id: FilterGZIP, name: "test-filter"}

	pipeline.AddFilter(filter)

	require.False(t, pipeline.IsEmpty())
	require.Equal(t, 1, pipeline.Count())
}

func TestFilterPipeline_AddMultipleFilters(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter1 := &mockFilter{id: FilterShuffle, name: "shuffle"}
	filter2 := &mockFilter{id: FilterGZIP, name: "gzip"}
	filter3 := &mockFilter{id: FilterFletcher32, name: "fletcher32"}

	pipeline.AddFilter(filter1)
	pipeline.AddFilter(filter2)
	pipeline.AddFilter(filter3)

	require.Equal(t, 3, pipeline.Count())
}

func TestFilterPipeline_AddFilterAtStart(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter1 := &mockFilter{id: FilterGZIP, name: "gzip"}
	filter2 := &mockFilter{id: FilterShuffle, name: "shuffle"}

	pipeline.AddFilter(filter1)
	pipeline.AddFilterAtStart(filter2)

	// Verify order by applying filters
	data := []byte{10, 20, 30}
	result, err := pipeline.Apply(data)
	require.NoError(t, err)

	// Shuffle (ID=2) should be applied first, then GZIP (ID=1)
	// Expected: ((10+2)+1, (20+2)+1, (30+2)+1) = (13, 23, 33)
	require.Equal(t, []byte{13, 23, 33}, result)
}

func TestFilterPipeline_ApplySingleFilter(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter := &mockFilter{id: 5, name: "test"}
	pipeline.AddFilter(filter)

	data := []byte{10, 20, 30}
	result, err := pipeline.Apply(data)

	require.NoError(t, err)
	require.Equal(t, []byte{15, 25, 35}, result)
}

func TestFilterPipeline_ApplyMultipleFilters(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter1 := &mockFilter{id: 2, name: "filter1"}
	filter2 := &mockFilter{id: 3, name: "filter2"}
	pipeline.AddFilter(filter1)
	pipeline.AddFilter(filter2)

	data := []byte{10, 20, 30}
	result, err := pipeline.Apply(data)

	require.NoError(t, err)
	// First filter adds 2, second adds 3
	require.Equal(t, []byte{15, 25, 35}, result)
}

func TestFilterPipeline_RemoveSingleFilter(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter := &mockFilter{id: 5, name: "test"}
	pipeline.AddFilter(filter)

	data := []byte{15, 25, 35}
	result, err := pipeline.Remove(data)

	require.NoError(t, err)
	require.Equal(t, []byte{10, 20, 30}, result)
}

func TestFilterPipeline_RemoveMultipleFilters(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter1 := &mockFilter{id: 2, name: "filter1"}
	filter2 := &mockFilter{id: 3, name: "filter2"}
	pipeline.AddFilter(filter1)
	pipeline.AddFilter(filter2)

	data := []byte{15, 25, 35}
	result, err := pipeline.Remove(data)

	require.NoError(t, err)
	// Filters removed in reverse order: subtract 3, then subtract 2
	require.Equal(t, []byte{10, 20, 30}, result)
}

func TestFilterPipeline_RoundTrip(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter1 := &mockFilter{id: 1, name: "filter1"}
	filter2 := &mockFilter{id: 2, name: "filter2"}
	filter3 := &mockFilter{id: 3, name: "filter3"}
	pipeline.AddFilter(filter1)
	pipeline.AddFilter(filter2)
	pipeline.AddFilter(filter3)

	original := []byte{10, 20, 30, 40, 50}

	// Apply all filters
	filtered, err := pipeline.Apply(original)
	require.NoError(t, err)
	require.NotEqual(t, original, filtered)

	// Remove all filters
	restored, err := pipeline.Remove(filtered)
	require.NoError(t, err)
	require.Equal(t, original, restored)
}

func TestFilterPipeline_ApplyError(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter1 := &mockFilter{id: 1, name: "good-filter"}
	filter2 := &mockFilter{id: 2, name: "bad-filter", shouldFail: true}
	pipeline.AddFilter(filter1)
	pipeline.AddFilter(filter2)

	data := []byte{10, 20, 30}
	_, err := pipeline.Apply(data)

	require.Error(t, err)
	require.Contains(t, err.Error(), "bad-filter")
}

func TestFilterPipeline_RemoveError(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter1 := &mockFilter{id: 1, name: "good-filter"}
	filter2 := &mockFilter{id: 2, name: "bad-filter", shouldFail: true}
	pipeline.AddFilter(filter1)
	pipeline.AddFilter(filter2)

	data := []byte{10, 20, 30}
	_, err := pipeline.Remove(data)

	require.Error(t, err)
	require.Contains(t, err.Error(), "bad-filter")
}

func TestFilterPipeline_EmptyData(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter := &mockFilter{id: 1, name: "test"}
	pipeline.AddFilter(filter)

	data := []byte{}
	result, err := pipeline.Apply(data)
	require.NoError(t, err)
	require.Equal(t, []byte{}, result)

	result, err = pipeline.Remove(data)
	require.NoError(t, err)
	require.Equal(t, []byte{}, result)
}

func TestFilterPipeline_EncodePipelineMessage_Empty(t *testing.T) {
	pipeline := NewFilterPipeline()

	_, err := pipeline.EncodePipelineMessage()
	require.Error(t, err)
	require.Contains(t, err.Error(), "empty filter pipeline")
}

func TestFilterPipeline_EncodePipelineMessage_SingleFilter(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter := &mockFilter{
		id:       FilterGZIP,
		name:     "deflate",
		flags:    0,
		cdValues: []uint32{6}, // Compression level
	}
	pipeline.AddFilter(filter)

	msg, err := pipeline.EncodePipelineMessage()
	require.NoError(t, err)

	// Version 2 header: version (1) + number of filters (1), no reserved bytes.
	require.Equal(t, byte(2), msg[0])
	require.Equal(t, byte(1), msg[1])

	// Predefined filters (ID < 256) have no name-length field and no name.
	offset := 2
	require.Equal(t, uint16(FilterGZIP), binary.LittleEndian.Uint16(msg[offset:]))
	require.Equal(t, uint16(0), binary.LittleEndian.Uint16(msg[offset+2:])) // flags
	require.Equal(t, uint16(1), binary.LittleEndian.Uint16(msg[offset+4:])) // number of CD values
	require.Equal(t, uint32(6), binary.LittleEndian.Uint32(msg[offset+6:]))
	require.Len(t, msg, 2+6+4)
}

func TestFilterPipeline_EncodePipelineMessage_MultipleFilters(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter1 := &mockFilter{
		id:       FilterShuffle,
		name:     "shuffle",
		flags:    0,
		cdValues: []uint32{4}, // Element size
	}
	filter2 := &mockFilter{
		id:       FilterGZIP,
		name:     "deflate",
		flags:    0,
		cdValues: []uint32{9}, // Compression level
	}
	pipeline.AddFilter(filter1)
	pipeline.AddFilter(filter2)

	msg, err := pipeline.EncodePipelineMessage()
	require.NoError(t, err)

	require.Equal(t, byte(2), msg[0]) // Version 2
	require.Equal(t, byte(2), msg[1]) // 2 filters

	// Header (2) + 2 * (ID 2 + flags 2 + nCD 2 + 1 CD value 4)
	require.Len(t, msg, 2+2*10)

	require.Equal(t, uint16(FilterShuffle), binary.LittleEndian.Uint16(msg[2:]))
	require.Equal(t, uint32(4), binary.LittleEndian.Uint32(msg[8:]))
	require.Equal(t, uint16(FilterGZIP), binary.LittleEndian.Uint16(msg[12:]))
	require.Equal(t, uint32(9), binary.LittleEndian.Uint32(msg[18:]))
}

func TestFilterPipeline_EncodePipelineMessage_NoName(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter := &mockFilter{
		id:       FilterFletcher32,
		name:     "", // No name
		flags:    0,
		cdValues: []uint32{}, // No CD values
	}
	pipeline.AddFilter(filter)

	msg, err := pipeline.EncodePipelineMessage()
	require.NoError(t, err)

	require.Equal(t, byte(2), msg[0]) // Version 2
	require.Equal(t, byte(1), msg[1]) // 1 filter
	require.Equal(t, uint16(FilterFletcher32), binary.LittleEndian.Uint16(msg[2:]))
	require.Equal(t, uint16(0), binary.LittleEndian.Uint16(msg[6:])) // no CD values

	// Header (2) + ID (2) + flags (2) + nCD (2)
	require.Len(t, msg, 8)
}

func TestFilterPipeline_EncodePipelineMessage_LongName(t *testing.T) {
	pipeline := NewFilterPipeline()
	filter := &mockFilter{
		id:       FilterID(32000), // third-party filter: name is encoded
		name:     "very-long-filter-name",
		flags:    42,
		cdValues: []uint32{1, 2, 3},
	}
	pipeline.AddFilter(filter)

	msg, err := pipeline.EncodePipelineMessage()
	require.NoError(t, err)

	offset := 2
	require.Equal(t, uint16(32000), binary.LittleEndian.Uint16(msg[offset:]))
	nameLen := binary.LittleEndian.Uint16(msg[offset+2:])
	require.Equal(t, uint16(22), nameLen) // 21 characters + NUL, not padded in version 2
	require.Equal(t, uint16(42), binary.LittleEndian.Uint16(msg[offset+4:]))
	require.Equal(t, uint16(3), binary.LittleEndian.Uint16(msg[offset+6:]))

	name := string(msg[offset+8 : offset+8+21])
	require.Equal(t, "very-long-filter-name", name)

	cdOffset := offset + 8 + 22
	require.Equal(t, uint32(1), binary.LittleEndian.Uint32(msg[cdOffset:]))
	require.Equal(t, uint32(2), binary.LittleEndian.Uint32(msg[cdOffset+4:]))
	require.Equal(t, uint32(3), binary.LittleEndian.Uint32(msg[cdOffset+8:]))
}
