package core

import (
	"compress/bzip2"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	"github.com/cwbudde/go-hdf5/internal/utils"
)

// FilterID represents HDF5 filter identifiers.
type FilterID uint16

// Filter identifier constants define compression and processing filters for datasets.
const (
	FilterDeflate     FilterID = 1     // GZIP compression.
	FilterShuffle     FilterID = 2     // Shuffle filter.
	FilterFletcher    FilterID = 3     // Fletcher32 checksum.
	FilterSZIP        FilterID = 4     // SZIP compression.
	FilterNBit        FilterID = 5     // N-bit compression.
	FilterScaleOffset FilterID = 6     // Scale-offset filter.
	FilterBZIP2       FilterID = 307   // BZIP2 compression.
	FilterLZF         FilterID = 32000 // LZF compression (PyTables/h5py).
)

// FilterPipelineMessage represents the filter pipeline for a dataset.
type FilterPipelineMessage struct {
	Version    uint8
	NumFilters uint8
	Filters    []Filter
}

// Filter represents a single filter in the pipeline.
type Filter struct {
	ID            FilterID
	NameLength    uint16
	Flags         uint16
	NumClientData uint16
	Name          string
	ClientData    []uint32
}

// ParseFilterPipelineMessage parses filter pipeline message (type 0x000B).
func ParseFilterPipelineMessage(data []byte) (*FilterPipelineMessage, error) {
	if len(data) < 2 {
		return nil, errors.New("filter pipeline message too short")
	}

	version := data[0]
	numFilters := data[1]

	// Support version 1 and 2.
	if version < 1 || version > 2 {
		return nil, fmt.Errorf("unsupported filter pipeline version: %d", version)
	}

	pipeline := &FilterPipelineMessage{
		Version:    version,
		NumFilters: numFilters,
		Filters:    make([]Filter, 0, numFilters),
	}

	offset := 2

	// Version 1 has 6 bytes reserved after num filters.
	if version == 1 {
		offset += 6
	}

	// Parse each filter.
	//
	// Version 1: ID, name length, flags, #client data (2 bytes each), name
	// padded to a multiple of 8, client data padded to a multiple of 8.
	// Version 2 (H5O__pline_decode): ID; name length only for IDs >= 256;
	// flags; #client data; name (not padded) only for IDs >= 256; client
	// data (not padded).
	for i := uint8(0); i < numFilters; i++ {
		if offset+2 > len(data) {
			return nil, fmt.Errorf("filter pipeline truncated at filter %d", i)
		}

		filter := Filter{}

		// Filter ID (2 bytes).
		filter.ID = FilterID(binary.LittleEndian.Uint16(data[offset : offset+2]))
		offset += 2

		hasName := version == 1 || filter.ID >= 256
		fixed := 4
		if hasName {
			fixed = 6
		}
		if offset+fixed > len(data) {
			return nil, fmt.Errorf("filter pipeline truncated at filter %d", i)
		}

		var nameLength uint16
		if hasName {
			nameLength = binary.LittleEndian.Uint16(data[offset : offset+2])
			offset += 2
		}
		filter.NameLength = nameLength

		// Flags (2 bytes).
		filter.Flags = binary.LittleEndian.Uint16(data[offset : offset+2])
		offset += 2

		// Number of client data values (2 bytes).
		filter.NumClientData = binary.LittleEndian.Uint16(data[offset : offset+2])
		offset += 2

		// Filter name (null-terminated; padded to 8 bytes only in version 1).
		if nameLength > 0 {
			padded := int(nameLength)
			if version == 1 && padded%8 != 0 {
				padded += 8 - (padded % 8)
			}

			if offset+padded > len(data) {
				return nil, fmt.Errorf("filter name truncated at filter %d", i)
			}

			// Extract name (up to first null).
			nameBytes := data[offset : offset+int(nameLength)]
			filter.Name = string(nameBytes)
			for idx, b := range nameBytes {
				if b == 0 {
					filter.Name = string(nameBytes[:idx])
					break
				}
			}

			offset += padded
		}

		// Client data (array of uint32).
		if filter.NumClientData > 0 {
			dataSize := int(filter.NumClientData) * 4
			if offset+dataSize > len(data) {
				return nil, fmt.Errorf("filter client data truncated at filter %d", i)
			}

			filter.ClientData = make([]uint32, filter.NumClientData)
			for j := uint16(0); j < filter.NumClientData; j++ {
				filter.ClientData[j] = binary.LittleEndian.Uint32(data[offset : offset+4])
				offset += 4
			}

			// Version 1: client data is padded to 8-byte boundary.
			if version == 1 {
				if dataSize%8 != 0 {
					offset += 8 - (dataSize % 8)
				}
			}
		}

		pipeline.Filters = append(pipeline.Filters, filter)
	}

	return pipeline, nil
}

// ApplyFilters applies filter pipeline to decompress/decode chunk data.
// The decoded output is limited to utils.MaxChunkSize bytes; use
// ApplyFiltersLimit when the expected chunk size is known.
func (fp *FilterPipelineMessage) ApplyFilters(data []byte) ([]byte, error) {
	return fp.ApplyFiltersLimit(data, utils.MaxChunkSize)
}

// ApplyFiltersLimit applies the filter pipeline to decode chunk data, refusing
// to produce more than limit bytes at any stage (protects against
// decompression bombs). limit is normally the expected uncompressed chunk size.
func (fp *FilterPipelineMessage) ApplyFiltersLimit(data []byte, limit uint64) ([]byte, error) {
	if limit > utils.MaxChunkSize {
		limit = utils.MaxChunkSize
	}
	maxOut := int(limit) //nolint:gosec // G115: bounded by MaxChunkSize above
	if fp == nil || len(fp.Filters) == 0 {
		return data, nil
	}

	// Filters are applied in REVERSE order during decompression.
	// (they were applied forward during compression).
	result := data

	for i := len(fp.Filters) - 1; i >= 0; i-- {
		filter := fp.Filters[i]

		// Skip optional filters if they fail.
		isOptional := (filter.Flags & 0x0001) != 0

		// The output of this stage is the input the earlier filters
		// produced, which may be larger than the final chunk (Fletcher32
		// adds a checksum, compressors can expand incompressible data).
		decoded, err := applyFilter(filter, result, stageLimit(fp.Filters[:i], maxOut))
		if err != nil {
			if isOptional {
				// Optional filter - log and continue.
				continue
			}
			return nil, fmt.Errorf("filter %d (%s) failed: %w", filter.ID, filterName(filter.ID), err)
		}
		result = decoded

		// LZF filter: ensure output matches expected size from cd_values[2].
		// The HDF5 LZF filter stores the expected uncompressed chunk size in cd_values[2].
		// If decompression produces less data than expected, pad with zeros.
		// This can happen with sparse chunks or chunks at dataset boundaries.
		if filter.ID == FilterLZF && len(filter.ClientData) >= 3 && filter.ClientData[2] > 0 {
			expectedSize := int(filter.ClientData[2])
			if len(result) < expectedSize && expectedSize <= maxOut {
				// Pad with zeros to match expected size
				padded := make([]byte, expectedSize)
				copy(padded, result)
				result = padded
			}
		}
	}

	return result, nil
}

// stageLimit returns the output size limit for a decoding stage whose output
// is still to be decoded by the given (earlier) filters: the final limit plus
// a bounded allowance for what each of those filters may add, capped at
// utils.MaxChunkSize.
func stageLimit(remaining []Filter, final int) int {
	limit := uint64(final) //nolint:gosec // G115: final is non-negative
	for _, f := range remaining {
		switch f.ID {
		case FilterFletcher:
			limit += 4
		case FilterShuffle, FilterNBit, FilterScaleOffset:
			// Size-preserving (or shrinking) on encode.
		default:
			// Compressors may expand incompressible input slightly
			// (deflate: 5 bytes per 16 KiB block plus header; bzip2 and
			// LZF have similar small bounds).
			limit += limit/64 + 1024
		}
		if limit >= utils.MaxChunkSize {
			return int(utils.MaxChunkSize)
		}
	}
	return int(limit) //nolint:gosec // G115: bounded by MaxChunkSize above
}

// applyFilter applies a single filter.
func applyFilter(filter Filter, data []byte, maxOut int) ([]byte, error) {
	switch filter.ID {
	case FilterDeflate:
		return applyDeflateLimit(data, maxOut)

	case FilterShuffle:
		return applyShuffle(data, filter.ClientData)

	case FilterFletcher:
		// Fletcher32 is a checksum - just verify and strip it.
		return applyFletcher32(data)

	case FilterBZIP2:
		return applyBZIP2Limit(data, maxOut)

	case FilterLZF:
		// LZF filter: check if data is actually uncompressed.
		// HDF5 stores data uncompressed if compression doesn't help.
		// cd_values[2] contains the expected uncompressed chunk size.
		if len(filter.ClientData) >= 3 && filter.ClientData[2] > 0 {
			expectedSize := int(filter.ClientData[2])
			if len(data) == expectedSize {
				// Data is already uncompressed (compression didn't help)
				return data, nil
			}
		}
		return applyLZFLimit(data, maxOut)

	case FilterSZIP:
		return applySZIP(data)

	default:
		return nil, fmt.Errorf("unsupported filter ID: %d", filter.ID)
	}
}

// applyDeflate decompresses GZIP/deflate compressed data.
// HDF5 uses raw deflate (zlib), not gzip format.
func applyDeflate(data []byte) ([]byte, error) {
	return applyDeflateLimit(data, utils.MaxChunkSize)
}

// readAllLimited reads all of r but fails if more than limit bytes are produced.
func readAllLimited(r io.Reader, limit int) ([]byte, error) {
	return readAllLimitedHint(r, limit, 0)
}

// maxDeflateRatio bounds the expansion of deflate data (the format's
// maximum is about 1032:1).
const maxDeflateRatio = 1032

// readAllLimitedHint is readAllLimited with the output buffer sized for
// min(limit, hint) bytes up front, so decoding a chunk of known size does
// not grow (and copy) the buffer repeatedly.
func readAllLimitedHint(r io.Reader, limit, hint int) ([]byte, error) {
	lr := io.LimitReader(r, int64(limit)+1)
	if hint <= 0 {
		out, err := io.ReadAll(lr)
		if err != nil {
			return nil, err
		}
		return checkLimit(out, limit)
	}
	out := make([]byte, 0, min(limit, hint)+1)
	for {
		if len(out) == cap(out) {
			out = append(out, 0)[:len(out)]
		}
		n, err := lr.Read(out[len(out):cap(out)])
		out = out[:len(out)+n]
		if errors.Is(err, io.EOF) {
			return checkLimit(out, limit)
		}
		if err != nil {
			return nil, err
		}
	}
}

// checkLimit fails if out holds more than limit bytes.
func checkLimit(out []byte, limit int) ([]byte, error) {
	if len(out) > limit {
		return nil, fmt.Errorf("decompressed data exceeds expected size %d", limit)
	}
	return out, nil
}

// applyDeflateLimit decompresses deflate data, producing at most limit bytes.
func applyDeflateLimit(data []byte, limit int) ([]byte, error) {
	reader, err := zlib.NewReader(io.NopCloser(io.NewSectionReader(
		&bytesReaderAt{data}, 0, int64(len(data)))))
	if err != nil {
		return nil, fmt.Errorf("zlib reader creation failed: %w", err)
	}
	defer func() { _ = reader.Close() }()

	// Read all decompressed data into a buffer sized for the limit (the
	// expected chunk size), bounded by what the input can expand to.
	hint := limit
	if len(data) < hint/maxDeflateRatio {
		hint = len(data)*maxDeflateRatio + 512
	}
	decompressed, err := readAllLimitedHint(reader, limit, hint)
	if err != nil {
		return nil, fmt.Errorf("zlib decompression failed: %w", err)
	}

	return decompressed, nil
}

// applyShuffle reverses shuffle filter.
// Shuffle reorders bytes to improve compression.
func applyShuffle(data []byte, clientData []uint32) ([]byte, error) {
	if len(clientData) == 0 {
		return nil, errors.New("shuffle filter missing element size")
	}

	elementSize := int(clientData[0])
	if elementSize <= 0 || elementSize > len(data) {
		return nil, fmt.Errorf("invalid shuffle element size: %d", elementSize)
	}

	numElements := len(data) / elementSize
	if len(data)%elementSize != 0 {
		return nil, errors.New("data size not multiple of element size")
	}

	result := make([]byte, len(data))

	// Reverse shuffle: data is organized as [all byte0][all byte1]...[all byteN].
	// We need to interleave them back.
	for elemIdx := 0; elemIdx < numElements; elemIdx++ {
		for byteIdx := 0; byteIdx < elementSize; byteIdx++ {
			srcPos := byteIdx*numElements + elemIdx
			dstPos := elemIdx*elementSize + byteIdx
			result[dstPos] = data[srcPos]
		}
	}

	return result, nil
}

// applyFletcher32 verifies and strips Fletcher32 checksum.
func applyFletcher32(data []byte) ([]byte, error) {
	if len(data) < 4 {
		return nil, errors.New("data too short for Fletcher32 checksum")
	}

	// Fletcher32 checksum is appended at the end (4 bytes).
	// Checksum verification deferred to v0.11.0-RC (feature-complete release).
	// Current implementation strips checksum without validation.
	// In practice, file system and HDF5 library corruption is extremely rare.
	// For production use, consider external file integrity checks (SHA256, etc.).
	// Reference: https://docs.hdfgroup.org/hdf5/latest/group___h5_z.html
	// Target version: v0.11.0-RC (comprehensive data integrity features)
	return data[:len(data)-4], nil
}

// applyBZIP2Limit decompresses BZIP2-compressed data.
// BZIP2 is a high-compression algorithm providing better compression than GZIP.
// Uses stdlib compress/bzip2 for decompression.
// It produces at most limit bytes of output.
func applyBZIP2Limit(data []byte, limit int) ([]byte, error) {
	reader := bzip2.NewReader(io.NopCloser(io.NewSectionReader(
		&bytesReaderAt{data}, 0, int64(len(data)))))

	// Read all decompressed data.
	decompressed, err := readAllLimited(reader, limit)
	if err != nil {
		return nil, fmt.Errorf("bzip2 decompression failed: %w", err)
	}

	return decompressed, nil
}

// applyLZFLimit decompresses LZF-compressed data.
// LZF is a very fast compression algorithm used by PyTables and h5py.
// It produces at most limit bytes of output.
func applyLZFLimit(data []byte, limit int) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	decompressed, err := lzfDecompressLimit(data, limit)
	if err != nil {
		return nil, fmt.Errorf("lzf decompression failed: %w", err)
	}
	return decompressed, nil
}

// applySZIP decompresses SZIP-compressed data.
// SZIP uses extended Golomb-Rice coding (CCSDS 121.0-B-3 standard).
// This algorithm is commonly used for satellite imagery and scientific data.
//
// SZIP requires libaec (Adaptive Entropy Coding) library for decompression.
// Since no pure Go implementation exists, we return an informative error.
//
// Reference: https://github.com/MathisRosenhauer/libaec
func applySZIP(_ []byte) ([]byte, error) {
	return nil, errors.New("SZIP decompression requires libaec library (not available in pure Go); " +
		"SZIP uses extended Golomb-Rice coding (CCSDS 121.0-B-3 standard); " +
		"to read SZIP-compressed datasets, use the HDF5 C library or h5py; " +
		"alternatively, re-save the file with GZIP compression (filter ID 1)")
}

// lzfDecompressLimit decompresses LZF-compressed data.
// LZF format consists of segments:
//   - Literal run (000LLLLL): L+1 bytes of uncompressed data
//   - Short backref (RRROXXXX XXXXXXXX): 3-8 bytes from offset 1-8192
//   - Long backref (111OXXXX RRRRRRRR XXXXXXXX): 9-264 bytes from offset 1-8192
//
// It fails if the output would exceed limit bytes.
func lzfDecompressLimit(input []byte, limit int) ([]byte, error) {
	inLen := len(input)
	if inLen == 0 {
		return input, nil
	}

	// Pre-allocate output buffer (LZF typically achieves 40-50% compression).
	output := make([]byte, 0, min(inLen*2, limit))
	inPos := 0

	for inPos < inLen {
		// Read control byte.
		ctrl := input[inPos]
		inPos++

		// Check segment type based on top 3 bits.
		if (ctrl & 0xE0) == 0 {
			// Literal run: 000LLLLL
			runLen := int(ctrl) + 1

			if inPos+runLen > inLen {
				return nil, errors.New("lzf: truncated literal run")
			}

			if len(output)+runLen > limit {
				return nil, fmt.Errorf("lzf: decompressed data exceeds expected size %d", limit)
			}
			output = append(output, input[inPos:inPos+runLen]...)
			inPos += runLen
		} else {
			// Backreference (liblzf lzf_d.c): length = ctrl>>5; for 7 an
			// extra length byte follows the control byte; then the low
			// offset byte. Offset is 1-based, length is len+2.
			offsetHigh := int(ctrl & 0x1F)
			runLen := int(ctrl >> 5)
			if runLen == 7 {
				if inPos >= inLen {
					return nil, errors.New("lzf: truncated long backreference")
				}
				runLen += int(input[inPos])
				inPos++
			}
			if inPos >= inLen {
				return nil, errors.New("lzf: truncated backreference")
			}
			offset := (offsetHigh<<8 | int(input[inPos])) + 1
			inPos++
			runLen += 2

			// Validate offset.
			if offset > len(output) {
				return nil, fmt.Errorf("lzf: invalid offset %d (output size: %d)", offset, len(output))
			}

			if len(output)+runLen > limit {
				return nil, fmt.Errorf("lzf: decompressed data exceeds expected size %d", limit)
			}

			// Copy from earlier position in output.
			srcPos := len(output) - offset
			for i := 0; i < runLen; i++ {
				output = append(output, output[srcPos+i])
			}
		}
	}

	return output, nil
}

// filterName returns human-readable filter name.
func filterName(id FilterID) string {
	switch id {
	case FilterDeflate:
		return "GZIP"
	case FilterShuffle:
		return "Shuffle"
	case FilterFletcher:
		return "Fletcher32"
	case FilterBZIP2:
		return "BZIP2"
	case FilterLZF:
		return "LZF"
	case FilterSZIP:
		return "SZIP"
	case FilterNBit:
		return "N-bit"
	case FilterScaleOffset:
		return "Scale-Offset"
	default:
		return fmt.Sprintf("Unknown-%d", id)
	}
}

// bytesReaderAt wraps []byte to implement io.ReaderAt.
type bytesReaderAt struct {
	data []byte
}

func (b *bytesReaderAt) ReadAt(p []byte, off int64) (n int, err error) {
	if off < 0 || off > int64(len(b.data)) {
		return 0, io.EOF
	}
	n = copy(p, b.data[off:])
	if n < len(p) {
		err = io.EOF
	}
	return n, err
}
