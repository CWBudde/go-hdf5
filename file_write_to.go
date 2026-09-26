package hdf5

import (
	"errors"
	"io"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/writer"
)

// CreateForWriteTo creates a new HDF5 file that is assembled in memory and
// written to w in a single pass when the returned FileWriter is closed.
// It accepts the same options as CreateForWrite.
//
// Writing HDF5 needs random access (headers are patched after the data they
// describe), so the whole file is buffered in memory until Close; nothing is
// written to w before then. Close returns any error from w. To target an
// io.WriterAt (e.g. a region of a larger file), wrap it with
// io.NewOffsetWriter(wa, off).
//
// Example:
//
//	var buf bytes.Buffer
//	fw, err := hdf5.CreateForWriteTo(&buf)
//	if err != nil {
//	    return err
//	}
//	// ... create datasets, groups, attributes ...
//	if err := fw.Close(); err != nil { // buf now holds the file
//	    return err
//	}
func CreateForWriteTo(w io.Writer, opts ...interface{}) (*FileWriter, error) {
	if w == nil {
		return nil, errors.New("nil writer")
	}
	cfg, tempFW, err := parseCreateOptions(opts)
	if err != nil {
		return nil, err
	}
	fw, err := newFileWriter(writer.NewMemoryFileWriter(cfg.superblockSize()), "", cfg, tempFW)
	if err != nil {
		return nil, err
	}
	fw.sink = w
	return fw, nil
}

// superblockSize returns the size of the superblock for the configured
// version: 96 bytes for v0, 48 for v2/v3.
func (cfg *FileWriteConfig) superblockSize() uint64 {
	if cfg.SuperblockVersion == core.Version0 {
		return 96
	}
	return 48
}
