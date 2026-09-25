package writer

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"fmt"
	"io"
)

// GZIPFilter implements GZIP compression (FilterID = 1).
// This filter uses the DEFLATE compression algorithm to reduce data size.
// In HDF5, this filter is named "deflate" following zlib terminology.
//
// Compression levels:
//
//	1 = fastest compression, larger files
//	6 = balanced (default)
//	9 = best compression, slower
type GZIPFilter struct {
	level int // Compression level (1-9)
}

// NewGZIPFilter creates a GZIP filter with the specified compression level.
//
// Valid levels:
//
//	1 = Fast compression, lower ratio
//	6 = Default (balanced)
//	9 = Best compression, slower
//
// Invalid levels are automatically adjusted to 6 (default).
func NewGZIPFilter(level int) *GZIPFilter {
	if level < 1 || level > 9 {
		level = 6 // Default compression level
	}
	return &GZIPFilter{level: level}
}

// ID returns the HDF5 filter identifier for GZIP.
func (f *GZIPFilter) ID() FilterID {
	return FilterGZIP
}

// Name returns the HDF5 filter name.
// HDF5 uses "deflate" (the underlying algorithm) rather than "gzip".
func (f *GZIPFilter) Name() string {
	return "deflate"
}

// Apply compresses data using GZIP/DEFLATE algorithm.
// Returns compressed data suitable for storage.
//
// The HDF5 deflate filter (H5Z_FILTER_DEFLATE) stores zlib streams
// (RFC 1950: 2-byte header + deflate data + Adler-32), not gzip files.
func (f *GZIPFilter) Apply(data []byte) ([]byte, error) {
	var buf bytes.Buffer

	// Create zlib writer with specified compression level
	w, err := zlib.NewWriterLevel(&buf, f.level)
	if err != nil {
		return nil, fmt.Errorf("gzip writer creation failed: %w", err)
	}

	// Compress data
	if _, err := w.Write(data); err != nil {
		_ = w.Close() // Ignore close error on write failure
		return nil, fmt.Errorf("gzip compression failed: %w", err)
	}

	// Flush and close to ensure all data is written
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("gzip close failed: %w", err)
	}

	return buf.Bytes(), nil
}

// Remove decompresses GZIP-compressed data.
// Returns the original uncompressed data.
//
// This method reverses the Apply operation, restoring the original data.
func (f *GZIPFilter) Remove(data []byte) ([]byte, error) {
	buf := bytes.NewReader(data)

	// HDF5 deflate data is a zlib stream. Older versions of this library
	// wrote gzip streams; accept those too (gzip magic 0x1f 0x8b).
	var r io.ReadCloser
	var err error
	if len(data) >= 2 && data[0] == 0x1f && data[1] == 0x8b {
		r, err = gzip.NewReader(buf)
	} else {
		r, err = zlib.NewReader(buf)
	}
	if err != nil {
		return nil, fmt.Errorf("gzip reader creation failed: %w", err)
	}
	defer func() { _ = r.Close() }() // Ignore error in defer

	// Decompress data
	decompressed, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("gzip decompression failed: %w", err)
	}

	return decompressed, nil
}

// Encode returns the filter parameters for the Pipeline message.
//
// For GZIP, the client data contains a single value: the compression level.
// Flags are always 0 for GZIP.
func (f *GZIPFilter) Encode() (flags uint16, cdValues []uint32) {
	return 0, []uint32{uint32(f.level)} //nolint:gosec // G115: Compression level is 1-9, always fits in uint32
}
