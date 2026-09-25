package utils

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

type plainReaderAt struct{ data []byte }

func (p plainReaderAt) ReadAt(b []byte, off int64) (int, error) {
	return bytes.NewReader(p.data).ReadAt(b, off)
}

func TestReaderSize(t *testing.T) {
	n, ok := ReaderSize(bytes.NewReader(make([]byte, 10)))
	require.True(t, ok)
	require.Equal(t, int64(10), n)

	p := filepath.Join(t.TempDir(), "f")
	require.NoError(t, os.WriteFile(p, make([]byte, 7), 0o600))
	f, err := os.Open(p)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	n, ok = ReaderSize(f)
	require.True(t, ok)
	require.Equal(t, int64(7), n)

	_, ok = ReaderSize(plainReaderAt{})
	require.False(t, ok)
}

func TestCheckReadBounds(t *testing.T) {
	r := bytes.NewReader(make([]byte, 100))
	require.NoError(t, CheckReadBounds(r, 0, 100, "x"))
	require.NoError(t, CheckReadBounds(r, 50, 50, "x"))
	require.Error(t, CheckReadBounds(r, 50, 51, "x"))
	require.Error(t, CheckReadBounds(r, 101, 0, "x"))
	// The ~131 TB local heap from the original crasher.
	require.Error(t, CheckReadBounds(r, 0, 0x780000000000, "local heap"))
	require.Error(t, CheckReadBounds(r, math.MaxUint64, 1, "x"))
	require.Error(t, CheckReadBounds(r, 1, math.MaxUint64, "x"))

	// Unknown size: capped.
	require.NoError(t, CheckReadBounds(plainReaderAt{}, 0, 1024, "x"))
	require.Error(t, CheckReadBounds(plainReaderAt{}, 0, MaxUnknownSizeRead+1, "x"))
}

func TestReadAtChecked(t *testing.T) {
	r := bytes.NewReader([]byte("hello world"))
	b, err := ReadAtChecked(r, 6, 5, "x")
	require.NoError(t, err)
	require.Equal(t, "world", string(b))

	_, err = ReadAtChecked(r, 6, 1<<40, "x")
	require.Error(t, err)
}

func TestCheckDecodedSize(t *testing.T) {
	small := bytes.NewReader(make([]byte, 1000))
	require.NoError(t, CheckDecodedSize(small, MaxDecodedDataSize, "x"))
	require.Error(t, CheckDecodedSize(small, MaxDecodedDataSize+1, "x"))
	// The ~4.5 TB dataset from the original crasher.
	require.Error(t, CheckDecodedSize(small, 4587767463936, "dataset"))
	require.Error(t, CheckDecodedSize(small, math.MaxUint64, "x"))
}

func TestElementsSize(t *testing.T) {
	n, total, err := ElementsSize([]uint64{2, 3, 4}, 8)
	require.NoError(t, err)
	require.Equal(t, uint64(24), n)
	require.Equal(t, uint64(192), total)

	n, total, err = ElementsSize(nil, 8)
	require.NoError(t, err)
	require.Equal(t, uint64(1), n)
	require.Equal(t, uint64(8), total)

	_, _, err = ElementsSize([]uint64{math.MaxUint64, 2}, 1)
	require.Error(t, err)
	_, _, err = ElementsSize([]uint64{1 << 62}, 8)
	require.Error(t, err)
}
