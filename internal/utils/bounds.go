package utils

import (
	"fmt"
	"io"
	"math"
	"os"
)

// MaxUnknownSizeRead caps single reads from readers whose total size cannot
// be determined (they implement neither Size() nor Stat()). Sizes read from
// the file must never drive unbounded allocations.
const MaxUnknownSizeRead = 256 * 1024 * 1024 // 256MB

// MaxDecodedDataSize caps the in-memory size of data whose size is derived from
// on-disk metadata but which is not necessarily stored byte-for-byte in the file
// (e.g. chunked datasets with compression or unallocated chunks).
// It is combined with the file size, see MaxDecodedSize.
const MaxDecodedDataSize = 256 * 1024 * 1024 // 256MB

// maxCompressionRatio bounds how much larger decoded data may be than the file
// that contains it (deflate tops out around 1032:1).
const maxCompressionRatio = 1100

// ReaderSize returns the total size of r if it can be determined.
// Supported: types with Size() int64 (bytes.Reader, io.SectionReader,
// strings.Reader) and types with Stat() (os.File).
func ReaderSize(r io.ReaderAt) (int64, bool) {
	switch v := r.(type) {
	case interface{ Size() int64 }:
		return v.Size(), true
	case interface{ Stat() (os.FileInfo, error) }:
		fi, err := v.Stat()
		if err != nil {
			return 0, false
		}
		return fi.Size(), true
	}
	return 0, false
}

// CheckReadBounds verifies that reading n bytes at offset from r stays within
// the reader's size (or within MaxUnknownSizeRead if the size is unknown).
// It must be called before allocating a buffer whose size comes from the file.
func CheckReadBounds(r io.ReaderAt, offset, n uint64, what string) error {
	if n > math.MaxInt64 || offset > math.MaxInt64 {
		return fmt.Errorf("%s: size %d at offset %d out of range", what, n, offset)
	}
	size, ok := ReaderSize(r)
	if !ok {
		if n > MaxUnknownSizeRead {
			return fmt.Errorf("%s: size %d exceeds limit %d", what, n, MaxUnknownSizeRead)
		}
		return nil
	}
	//nolint:gosec // G115: size is non-negative
	fsz := uint64(size)
	if offset > fsz || n > fsz-offset {
		return fmt.Errorf("%s: %d bytes at offset %d exceed file size %d", what, n, offset, fsz)
	}
	return nil
}

// ReadAtChecked allocates a buffer of n bytes and fills it from r at offset,
// after verifying the read stays within the reader's bounds.
func ReadAtChecked(r io.ReaderAt, offset, n uint64, what string) ([]byte, error) {
	if err := CheckReadBounds(r, offset, n, what); err != nil {
		return nil, err
	}
	buf := make([]byte, n)
	//nolint:gosec // G115: bounds checked above
	if _, err := r.ReadAt(buf, int64(offset)); err != nil {
		return nil, WrapError(what+" read failed", err)
	}
	return buf, nil
}

// MaxDecodedSize returns the maximum number of bytes that may be allocated for
// decoded (possibly compressed or fill-value) data from r.
func MaxDecodedSize(r io.ReaderAt) uint64 {
	size, ok := ReaderSize(r)
	if !ok {
		return MaxDecodedDataSize
	}
	//nolint:gosec // G115: size is non-negative
	limit := uint64(size)
	if limit > math.MaxUint64/maxCompressionRatio {
		return math.MaxUint64
	}
	limit *= maxCompressionRatio
	if limit < MaxDecodedDataSize {
		limit = MaxDecodedDataSize
	}
	return limit
}

// CheckDecodedSize returns an error if n bytes of decoded data would exceed
// the limit returned by MaxDecodedSize, or cannot be represented as an int.
func CheckDecodedSize(r io.ReaderAt, n uint64, what string) error {
	if n > uint64(math.MaxInt) {
		return fmt.Errorf("%s: size %d exceeds addressable memory", what, n)
	}
	if limit := MaxDecodedSize(r); n > limit {
		return fmt.Errorf("%s: size %d exceeds limit %d", what, n, limit)
	}
	return nil
}

// ElementsSize computes product(dims) * elemSize, returning an error on
// overflow. A zero-length dims yields elemSize (scalar).
func ElementsSize(dims []uint64, elemSize uint64) (numElements, total uint64, err error) {
	numElements = 1
	for i, d := range dims {
		if d != 0 && numElements > math.MaxUint64/d {
			return 0, 0, fmt.Errorf("dimension product overflow at dimension %d", i)
		}
		numElements *= d
	}
	if elemSize != 0 && numElements > math.MaxUint64/elemSize {
		return 0, 0, fmt.Errorf("data size overflow: %d elements of %d bytes", numElements, elemSize)
	}
	return numElements, numElements * elemSize, nil
}
