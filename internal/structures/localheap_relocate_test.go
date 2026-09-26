package structures

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestLocalHeapRelocate fills a heap, relocates its data segment and checks
// that old and new strings keep their offsets after a reload.
func TestLocalHeapRelocate(t *testing.T) {
	var buf bytes.Buffer
	w := newBytesWriterAt(&buf)
	sb := createMockSuperblock()

	heap := NewLocalHeap(16)
	offA, err := heap.AddString("abcdefghij")
	require.NoError(t, err)
	_, err = heap.AddString("klmnopqrst")
	require.ErrorIs(t, err, ErrLocalHeapFull)
	require.NoError(t, heap.WriteTo(w, 0))

	loaded, err := LoadLocalHeap(bytes.NewReader(buf.Bytes()), 0, sb)
	require.NoError(t, err)
	require.Equal(t, uint64(32), loaded.DataSegmentAddress)
	require.NoError(t, loaded.PrepareForModification())
	require.Equal(t, uint64(11), loaded.UsedSize())

	loaded.Relocate(256, 64)
	offB, err := loaded.AddString("klmnopqrst")
	require.NoError(t, err)
	require.NoError(t, loaded.WriteTo(w, 0))

	again, err := LoadLocalHeap(bytes.NewReader(buf.Bytes()), 0, sb)
	require.NoError(t, err)
	require.Equal(t, uint64(256), again.DataSegmentAddress)
	a, err := again.GetString(offA)
	require.NoError(t, err)
	require.Equal(t, "abcdefghij", a)
	b, err := again.GetString(offB)
	require.NoError(t, err)
	require.Equal(t, "klmnopqrst", b)
}
