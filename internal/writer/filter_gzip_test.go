package writer

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewGZIPFilter(t *testing.T) {
	tests := []struct {
		name          string
		inputLevel    int
		expectedLevel int
	}{
		{"valid level 1", 1, 1},
		{"valid level 6", 6, 6},
		{"valid level 9", 9, 9},
		{"invalid level 0", 0, 6},   // Should default to 6
		{"invalid level 10", 10, 6}, // Should default to 6
		{"invalid level -1", -1, 6}, // Should default to 6
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewGZIPFilter(tt.inputLevel)
			require.NotNil(t, filter)
			require.Equal(t, tt.expectedLevel, filter.level)
		})
	}
}

func TestGZIPFilter_ID(t *testing.T) {
	filter := NewGZIPFilter(6)
	require.Equal(t, FilterGZIP, filter.ID())
	require.Equal(t, FilterID(1), filter.ID())
}

func TestGZIPFilter_Name(t *testing.T) {
	filter := NewGZIPFilter(6)
	require.Equal(t, "deflate", filter.Name())
}

func TestGZIPFilter_Encode(t *testing.T) {
	tests := []struct {
		name  string
		level int
	}{
		{"level 1", 1},
		{"level 6", 6},
		{"level 9", 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewGZIPFilter(tt.level)
			flags, cdValues := filter.Encode()

			require.Equal(t, uint16(0), flags)
			require.Equal(t, 1, len(cdValues))
			require.Equal(t, uint32(tt.level), cdValues[0])
		})
	}
}

func TestGZIPFilter_CompressSmallData(t *testing.T) {
	filter := NewGZIPFilter(6)
	data := []byte{1, 2, 3, 4, 5}

	compressed, err := filter.Apply(data)
	require.NoError(t, err)
	require.NotNil(t, compressed)
	require.NotEqual(t, data, compressed)
	// Compressed data should have GZIP headers
	require.Greater(t, len(compressed), 10) // GZIP has minimum overhead
}

func TestGZIPFilter_CompressMediumData(t *testing.T) {
	filter := NewGZIPFilter(6)
	// Create 1KB of data
	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	compressed, err := filter.Apply(data)
	require.NoError(t, err)
	require.NotNil(t, compressed)
	require.NotEqual(t, data, compressed)
}

func TestGZIPFilter_CompressLargeData(t *testing.T) {
	filter := NewGZIPFilter(6)
	// Create 100KB of data
	data := make([]byte, 100*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	compressed, err := filter.Apply(data)
	require.NoError(t, err)
	require.NotNil(t, compressed)
	require.NotEqual(t, data, compressed)
}

func TestGZIPFilter_CompressRepetitiveData(t *testing.T) {
	filter := NewGZIPFilter(6)
	// Create highly repetitive data (should compress well)
	data := bytes.Repeat([]byte{42}, 10000)

	compressed, err := filter.Apply(data)
	require.NoError(t, err)

	// Should achieve good compression ratio (>50%)
	compressionRatio := float64(len(data)) / float64(len(compressed))
	require.Greater(t, compressionRatio, 2.0, "Expected compression ratio > 2:1 for repetitive data")
}

func TestGZIPFilter_DecompressData(t *testing.T) {
	filter := NewGZIPFilter(6)
	original := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	// First compress
	compressed, err := filter.Apply(original)
	require.NoError(t, err)

	// Then decompress
	decompressed, err := filter.Remove(compressed)
	require.NoError(t, err)
	require.Equal(t, original, decompressed)
}

func TestGZIPFilter_RoundTrip_SmallData(t *testing.T) {
	filter := NewGZIPFilter(6)
	original := []byte("Hello, HDF5 GZIP filter!")

	compressed, err := filter.Apply(original)
	require.NoError(t, err)

	decompressed, err := filter.Remove(compressed)
	require.NoError(t, err)

	require.Equal(t, original, decompressed)
}

func TestGZIPFilter_RoundTrip_LargeData(t *testing.T) {
	filter := NewGZIPFilter(6)
	// Create 1MB of varied data
	original := make([]byte, 1024*1024)
	for i := range original {
		original[i] = byte(i * 7 % 256)
	}

	compressed, err := filter.Apply(original)
	require.NoError(t, err)

	decompressed, err := filter.Remove(compressed)
	require.NoError(t, err)

	require.Equal(t, original, decompressed)
}

func TestGZIPFilter_RoundTrip_EmptyData(t *testing.T) {
	filter := NewGZIPFilter(6)
	original := []byte{}

	compressed, err := filter.Apply(original)
	require.NoError(t, err)

	decompressed, err := filter.Remove(compressed)
	require.NoError(t, err)

	require.Equal(t, original, decompressed)
}

func TestGZIPFilter_RoundTrip_SingleByte(t *testing.T) {
	filter := NewGZIPFilter(6)
	original := []byte{42}

	compressed, err := filter.Apply(original)
	require.NoError(t, err)

	decompressed, err := filter.Remove(compressed)
	require.NoError(t, err)

	require.Equal(t, original, decompressed)
}

func TestGZIPFilter_CompressionLevels(t *testing.T) {
	data := bytes.Repeat([]byte("The quick brown fox jumps over the lazy dog. "), 100)

	tests := []struct {
		name  string
		level int
	}{
		{"level 1 (fast)", 1},
		{"level 6 (default)", 6},
		{"level 9 (best)", 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewGZIPFilter(tt.level)

			compressed, err := filter.Apply(data)
			require.NoError(t, err)

			decompressed, err := filter.Remove(compressed)
			require.NoError(t, err)

			require.Equal(t, data, decompressed)

			// All levels should achieve reasonable compression
			compressionRatio := float64(len(data)) / float64(len(compressed))
			require.Greater(t, compressionRatio, 1.5)
		})
	}
}

// TestGZIPFilter_CompressionRatio_Comparison checks that the compression
// level is honored. It used to require level 9 output to be strictly smaller
// than level 1 output, which DEFLATE does not guarantee: for highly redundant
// input both levels can reach the same encoding, and since Go 1.27's
// compress/flate they do for this input (121 bytes each). Instead the test
// checks properties that hold for every conforming encoder: each level
// round-trips, compresses redundant data well, and the zlib header advertises
// the requested level (RFC 1950 FLEVEL: 0 = fastest ... 3 = maximum).
func TestGZIPFilter_CompressionRatio_Comparison(t *testing.T) {
	data := bytes.Repeat([]byte("Lorem ipsum dolor sit amet. "), 1000)

	tests := []struct {
		level  int
		flevel byte
	}{
		{1, 0},
		{6, 2},
		{9, 3},
	}

	for _, tt := range tests {
		filter := NewGZIPFilter(tt.level)

		compressed, err := filter.Apply(data)
		require.NoError(t, err)

		// zlib header: CMF (deflate, 32K window) + FLG with FLEVEL in bits 6-7.
		require.GreaterOrEqual(t, len(compressed), 2)
		require.Equal(t, byte(0x78), compressed[0], "level %d: CMF", tt.level)
		require.Equal(t, tt.flevel, compressed[1]>>6, "level %d: FLEVEL", tt.level)
		require.Zero(t, (uint16(compressed[0])<<8|uint16(compressed[1]))%31, "level %d: FCHECK", tt.level)

		// 28 KB of a repeated 28-byte phrase must shrink by far more than 10x
		// at any level.
		require.Less(t, len(compressed)*10, len(data), "level %d: %d bytes", tt.level, len(compressed))

		decompressed, err := filter.Remove(compressed)
		require.NoError(t, err)
		require.Equal(t, data, decompressed)
	}
}

func TestGZIPFilter_Remove_InvalidData(t *testing.T) {
	filter := NewGZIPFilter(6)

	// Try to decompress invalid data
	invalidData := []byte{1, 2, 3, 4, 5}
	_, err := filter.Remove(invalidData)
	require.Error(t, err)
	require.Contains(t, err.Error(), "gzip")
}

func TestGZIPFilter_Remove_CorruptedData(t *testing.T) {
	filter := NewGZIPFilter(6)
	original := []byte("Test data for corruption")

	compressed, err := filter.Apply(original)
	require.NoError(t, err)

	// Corrupt the compressed data
	if len(compressed) > 10 {
		corrupted := make([]byte, len(compressed))
		copy(corrupted, compressed)
		corrupted[len(corrupted)/2] ^= 0xFF // Flip bits in middle

		_, err = filter.Remove(corrupted)
		require.Error(t, err)
	}
}

func TestGZIPFilter_IntegrationWithPipeline(t *testing.T) {
	// Test GZIP filter in a pipeline
	pipeline := NewFilterPipeline()
	filter := NewGZIPFilter(6)
	pipeline.AddFilter(filter)

	original := bytes.Repeat([]byte{1, 2, 3, 4, 5}, 1000)

	// Apply pipeline
	filtered, err := pipeline.Apply(original)
	require.NoError(t, err)
	require.NotEqual(t, original, filtered)

	// Remove pipeline
	restored, err := pipeline.Remove(filtered)
	require.NoError(t, err)
	require.Equal(t, original, restored)
}

func TestGZIPFilter_BinaryData(t *testing.T) {
	filter := NewGZIPFilter(6)

	// Test with various binary patterns
	tests := []struct {
		name string
		data []byte
	}{
		{"all zeros", make([]byte, 1000)}, // Using make instead of bytes.Repeat for zero slice
		{"all ones", bytes.Repeat([]byte{0xFF}, 1000)},
		{"alternating", func() []byte {
			data := make([]byte, 1000)
			for i := range data {
				data[i] = byte(i % 2)
			}
			return data
		}()},
		{"random pattern", func() []byte {
			data := make([]byte, 1000)
			for i := range data {
				data[i] = byte((i*13 + 7) % 256)
			}
			return data
		}()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			compressed, err := filter.Apply(tt.data)
			require.NoError(t, err)

			decompressed, err := filter.Remove(compressed)
			require.NoError(t, err)

			require.Equal(t, tt.data, decompressed)
		})
	}
}
