package core

import (
	"bytes"
	"compress/zlib"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestReadAllLimitedHint checks that the size hint only affects the
// initial buffer: output larger than the hint grows it, and the limit is
// enforced either way.
func TestReadAllLimitedHint(t *testing.T) {
	data := bytes.Repeat([]byte("0123456789"), 1000)
	for _, hint := range []int{0, 1, 100, len(data), 2 * len(data)} {
		got, err := readAllLimitedHint(bytes.NewReader(data), len(data), hint)
		require.NoError(t, err, "hint %d", hint)
		require.Equal(t, data, got, "hint %d", hint)

		_, err = readAllLimitedHint(bytes.NewReader(data), len(data)-1, hint)
		require.ErrorContains(t, err, "exceeds expected size", "hint %d", hint)
	}
}

// TestApplyDeflateLimitPrealloc decompresses a highly compressible chunk
// whose output is far above the input size.
func TestApplyDeflateLimitPrealloc(t *testing.T) {
	want := make([]byte, 1<<20)
	var buf bytes.Buffer
	w := zlib.NewWriter(&buf)
	_, err := w.Write(want)
	require.NoError(t, err)
	require.NoError(t, w.Close())

	got, err := applyDeflateLimit(buf.Bytes(), len(want))
	require.NoError(t, err)
	require.Equal(t, want, got)
	_, err = applyDeflateLimit(buf.Bytes(), len(want)-1)
	require.Error(t, err)
}
