// Package hdf5 provides a pure Go implementation for reading HDF5 files.
// It supports HDF5 format versions 0, 2, and 3, with capabilities for
// reading datasets, groups, attributes, and various data layouts.
package hdf5

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/utils"
)

// File represents an open HDF5 file with its metadata and root group.
type File struct {
	// osFile is the reader all metadata and data are read from. It is an
	// io.SectionReader bounded to the file size, so bounds checks see the
	// size given to OpenReader (or the size of the file opened by Open).
	osFile        io.ReaderAt
	closer        io.Closer // Closed by Close; nil for OpenReader.
	sb            *core.Superblock
	root          *Group
	visitedBTrees map[uint64]bool // Track visited B-tree addresses to prevent cycles

	// expandedGroups records group addresses whose children were already
	// loaded. A group reached again (hard-link cycle or shared subgroup) is
	// returned without children so corrupted or cyclic files cannot cause
	// unbounded recursion or exponential work.
	expandedGroups map[uint64]bool
	// loadDepth is the current object-loading recursion depth.
	loadDepth int

	// objects maps object header addresses to objects and their paths,
	// built on first use by reference resolution (see objectIndex).
	objects map[uint64]objectEntry
}

// maxLoadDepth bounds object nesting while loading the group hierarchy.
const maxLoadDepth = 512

// markGroupExpanded records that the children of the group at address are
// being loaded. It returns false if that group was already expanded.
func (f *File) markGroupExpanded(address uint64) bool {
	if f.expandedGroups == nil {
		f.expandedGroups = make(map[uint64]bool)
	}
	if f.expandedGroups[address] {
		return false
	}
	f.expandedGroups[address] = true
	return true
}

// Open opens an HDF5 file for reading and returns a File handle.
// The file must be a valid HDF5 file with a supported format version.
func Open(filename string) (*File, error) {
	//nolint:gosec // G304: User-provided filename is intentional for HDF5 file library
	f, err := os.Open(filename)
	if err != nil {
		return nil, utils.WrapError("file open failed", err)
	}

	// Get file size for address validation.
	fi, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, utils.WrapError("file stat failed", err)
	}

	file, err := OpenReader(f, fi.Size())
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	file.closer = f
	return file, nil
}

// OpenReader reads an HDF5 file of size bytes from r, e.g. a bytes.Reader
// over an in-memory file or an embedded asset. All reads stay within the
// first size bytes of r; size is also the bound used to reject corrupt
// addresses and lengths.
//
// The returned File reads from r lazily (dataset data is read on demand),
// so r must remain valid while the File is in use. Close does not close r.
func OpenReader(r io.ReaderAt, size int64) (*File, error) {
	if r == nil {
		return nil, errors.New("nil reader")
	}
	if size < 0 {
		return nil, fmt.Errorf("negative file size %d", size)
	}
	sr := io.NewSectionReader(r, 0, size)

	// Verify HDF5 signature before reading superblock.
	if !isHDF5File(sr) {
		return nil, errors.New("not an HDF5 file")
	}

	sb, err := core.ReadSuperblock(sr)
	if err != nil {
		return nil, utils.WrapError("superblock read failed", err)
	}

	file := &File{
		osFile:        sr,
		sb:            sb,
		visitedBTrees: make(map[uint64]bool),
	}

	// Validate root group address.
	if sb.RootGroup >= uint64(size) {
		return nil, fmt.Errorf("root group address %d beyond file size %d",
			sb.RootGroup, size)
	}

	// For all versions, sb.RootGroup now contains the correct object header address.
	file.root, err = loadGroup(file, sb.RootGroup)
	if err != nil {
		return nil, utils.WrapError("root group load failed", err)
	}

	// Ensure root group always has name "/" (may be empty from object header)
	file.root.name = "/"

	return file, nil
}

// isHDF5File verifies HDF5 file signature.
func isHDF5File(r utils.ReaderAt) bool {
	buf := utils.GetBuffer(8)
	defer utils.ReleaseBuffer(buf)

	if _, err := r.ReadAt(buf, 0); err != nil {
		return false
	}
	return string(buf) == core.Signature
}

// Close closes the HDF5 file and releases associated resources.
// It is safe to call Close multiple times.
//
// For a File from OpenReader, Close does not close the underlying reader.
func (f *File) Close() error {
	if f.closer == nil {
		return nil // Already closed, or nothing to close.
	}
	err := f.closer.Close()
	f.closer = nil // Prevent double close.
	return err
}

// Root returns the root group of the HDF5 file.
func (f *File) Root() *Group {
	return f.root
}

// Walk traverses the entire file structure, calling fn for each object.
// Objects are visited in depth-first order starting from the root group.
func (f *File) Walk(fn func(path string, obj Object)) {
	walkGroup(f.root, "/", fn)
}

func walkGroup(g *Group, currentPath string, fn func(string, Object)) {
	fn(currentPath, g)

	for _, child := range g.Children() {
		childPath := currentPath + child.Name()

		if childGroup, ok := child.(*Group); ok {
			walkGroup(childGroup, childPath+"/", fn)
		} else {
			fn(childPath, child)
		}
	}
}

// SuperblockVersion returns the HDF5 superblock format version (0, 2, or 3).
func (f *File) SuperblockVersion() uint8 {
	return f.sb.Version
}

// Superblock returns the file's superblock metadata structure.
func (f *File) Superblock() *core.Superblock {
	return f.sb
}

// Reader returns the underlying file reader for low-level access.
func (f *File) Reader() io.ReaderAt {
	return f.osFile
}

// readSignature reads 4 bytes at address and returns string.
func readSignature(r io.ReaderAt, address uint64) string {
	buf := make([]byte, 4)
	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
	if _, err := r.ReadAt(buf, int64(address)); err != nil {
		return ""
	}
	return string(buf)
}
