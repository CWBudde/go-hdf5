package writer

import (
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMemoryBuffer(t *testing.T) {
	var m MemoryBuffer
	n, err := m.WriteAt([]byte("abc"), 5)
	require.NoError(t, err)
	require.Equal(t, 3, n)
	require.Equal(t, []byte{0, 0, 0, 0, 0, 'a', 'b', 'c'}, m.Bytes())
	require.EqualValues(t, 8, m.Size())

	buf := make([]byte, 4)
	n, err = m.ReadAt(buf, 6)
	require.ErrorIs(t, err, io.EOF)
	require.Equal(t, 2, n)
	require.Equal(t, "bc", string(buf[:n]))
	_, err = m.ReadAt(buf, 8)
	require.ErrorIs(t, err, io.EOF)
	_, err = m.ReadAt(buf, -1)
	require.Error(t, err)
	_, err = m.WriteAt(buf, -1)
	require.Error(t, err)

	require.NoError(t, m.Truncate(2))
	require.NoError(t, m.Truncate(6)) // re-extension is zero-filled
	require.Equal(t, make([]byte, 6), m.Bytes())
	require.Error(t, m.Truncate(-1))
}

func TestMemoryFileWriter(t *testing.T) {
	w := NewMemoryFileWriter(48)
	require.Nil(t, w.File())
	addr, err := w.WriteAtWithAllocation([]byte("hdf5"))
	require.NoError(t, err)
	require.EqualValues(t, 48, addr)
	size, err := w.Size()
	require.NoError(t, err)
	require.EqualValues(t, 52, size)
	require.NoError(t, w.Truncate(64))
	require.Len(t, w.Bytes(), 64)
	require.NoError(t, w.Flush())
	_, err = w.Seek(0, io.SeekStart)
	require.Error(t, err)
	got := make([]byte, 4)
	_, err = w.Reader().ReadAt(got, 48)
	require.NoError(t, err)
	require.Equal(t, "hdf5", string(got))
	require.NoError(t, w.Close())
	require.Nil(t, w.Bytes())
	_, err = w.Size()
	require.Error(t, err)
	require.Error(t, w.Truncate(1))
}
