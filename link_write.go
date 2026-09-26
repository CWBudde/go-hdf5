package hdf5

import (
	"fmt"
	"strings"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/writer"
)

// CreateHardLink creates a hard link to an existing object.
//
// Hard links are additional names for the same object. All hard links point to
// the same object header address. When one link is modified, changes are visible
// through all other links because they share the same data.
//
// Parameters:
//   - linkPath: Path where the new link will be created (e.g., "/group1/link_name")
//   - targetPath: Path to the existing object to link to (e.g., "/group2/dataset1")
//
// Returns:
//   - error: Non-nil if link creation fails
//
// Behavior:
//   - Validates both paths exist and are properly formatted
//   - Looks up target object's header address
//   - Increments reference count on target object header
//   - Creates link entry in parent group pointing to target address
//   - Supports linking datasets and groups
//   - Works with both symbol table and dense group formats
//
// Example:
//
//	fw, _ := hdf5.CreateForWrite("data.h5", hdf5.CreateTruncate)
//	defer fw.Close()
//
//	// Create dataset
//	fw.CreateDataset("/data/temperature", []float64{1.0, 2.0, 3.0})
//
//	// Create hard link to dataset
//	err := fw.CreateHardLink("/data/temp_link", "/data/temperature")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	// Now /data/temperature and /data/temp_link point to the same dataset
//
// Limitations (MVP v0.11.5-beta):
//   - Target must exist before creating link
//   - Parent group must exist before creating link
//   - Reference count stored in object header (v1) or RefCount message (v2)
//   - No link deletion support yet (DeleteLink not implemented)
//   - No circular link detection
//
// Reference: H5L.c - H5Lcreate_hard().
func (fw *FileWriter) CreateHardLink(linkPath, targetPath string) error {
	// Validate paths
	if err := validateLinkPath(linkPath); err != nil {
		return fmt.Errorf("invalid link path: %w", err)
	}
	if err := validateLinkPath(targetPath); err != nil {
		return fmt.Errorf("invalid target path: %w", err)
	}

	// Resolve target object address
	targetAddr, err := fw.resolveObjectAddress(targetPath)
	if err != nil {
		return fmt.Errorf("failed to resolve target %q: %w", targetPath, err)
	}

	// Read target object header to increment reference count
	targetHeader, err := core.ReadObjectHeader(fw.writer, targetAddr, fw.file.sb)
	if err != nil {
		return fmt.Errorf("failed to read target object header at 0x%x: %w", targetAddr, err)
	}

	// Increment reference count
	newRefCount := targetHeader.IncrementReferenceCount()

	// Write updated object header back to file
	if err := writeObjectHeaderWithRefCount(fw, targetAddr, targetHeader); err != nil {
		return fmt.Errorf("failed to update reference count: %w", err)
	}

	// Parse link path to find parent and link name
	parent, linkName := parsePath(linkPath)

	// Validate parent exists
	if parent != "" && parent != "/" {
		if _, exists := fw.groups[parent]; !exists {
			return fmt.Errorf("parent group %q does not exist (create it first)", parent)
		}
	}

	// Create link in parent group
	if err := fw.linkToParent(parent, linkName, targetAddr); err != nil {
		// Rollback: decrement reference count
		targetHeader.DecrementReferenceCount()
		_ = writeObjectHeaderWithRefCount(fw, targetAddr, targetHeader)
		return fmt.Errorf("failed to create link in parent group: %w", err)
	}

	// Success
	_ = newRefCount // Used for debugging if needed
	return nil
}

// validateLinkPath validates that a link path is properly formatted.
func validateLinkPath(path string) error {
	if path == "" {
		return fmt.Errorf("path cannot be empty")
	}
	if !strings.HasPrefix(path, "/") {
		return fmt.Errorf("path must start with '/' (got %q)", path)
	}
	if path == "/" {
		return fmt.Errorf("cannot create link to root group")
	}
	if strings.Contains(path, "//") {
		return fmt.Errorf("path cannot contain consecutive slashes (got %q)", path)
	}
	return nil
}

// writeObjectHeaderWithRefCount writes an object header back to disk with updated reference count.
// This handles the difference between v1 (refcount in header) and v2 (RefCount message).
//
// For v1: Updates refcount field directly in header bytes 4-7
// For v2: Updates or creates RefCount message (type 0x0016)
//
// Parameters:
//   - fw: FileWriter
//   - addr: Address of object header
//   - oh: Object header with updated ReferenceCount field
//
// Returns:
//   - error: Non-nil if write fails
func writeObjectHeaderWithRefCount(fw *FileWriter, addr uint64, oh *core.ObjectHeader) error {
	switch oh.Version {
	case 1:
		return writeV1RefCount(fw, addr, oh)
	case 2:
		return writeV2RefCount(fw, addr, oh)
	default:
		return fmt.Errorf("unsupported object header version: %d", oh.Version)
	}
}

// writeV1RefCount writes reference count for v1 object header.
func writeV1RefCount(fw *FileWriter, addr uint64, oh *core.ObjectHeader) error {
	// V1: Reference count is stored directly in header bytes 4-7
	// Write only the refcount field (4 bytes at offset 4)
	refCountBytes := make([]byte, 4)
	fw.file.sb.Endianness.PutUint32(refCountBytes, oh.ReferenceCount)

	_, err := fw.writer.WriteAt(refCountBytes, int64(addr+4)) //nolint:gosec // Safe: addr is file address
	if err != nil {
		return fmt.Errorf("failed to write v1 refcount: %w", err)
	}
	return nil
}

// writeV2RefCount writes reference count for v2 object header.
func writeV2RefCount(fw *FileWriter, addr uint64, oh *core.ObjectHeader) error {
	// V2: Reference count stored in RefCount message (type 0x0016)
	// If refcount > 1, we need to add/update RefCount message
	if oh.ReferenceCount > 1 {
		if err := ensureRefCountMessage(fw, oh); err != nil {
			return err
		}
	}

	// Write entire object header back to disk
	err := core.WriteObjectHeader(fw.writer, addr, oh, fw.file.sb)
	if err != nil {
		return fmt.Errorf("failed to write v2 object header: %w", err)
	}
	return nil
}

// ensureRefCountMessage ensures RefCount message exists and is updated.
func ensureRefCountMessage(fw *FileWriter, oh *core.ObjectHeader) error {
	// Check if RefCount message already exists
	for _, msg := range oh.Messages {
		if msg.Type == core.MsgRefCount && len(msg.Data) >= 4 {
			// Update existing RefCount message data
			fw.file.sb.Endianness.PutUint32(msg.Data[0:4], oh.ReferenceCount)
			return nil
		}
	}

	// Create new RefCount message
	refCountData := make([]byte, 4)
	fw.file.sb.Endianness.PutUint32(refCountData, oh.ReferenceCount)

	// Add message to header
	err := core.AddMessageToObjectHeader(oh, core.MsgRefCount, refCountData)
	if err != nil {
		return fmt.Errorf("failed to add RefCount message: %w", err)
	}
	return nil
}

// CreateSoftLink creates a symbolic link to a path within the HDF5 file.
//
// Soft links (symbolic links) store a path string that is resolved when accessed.
// Unlike hard links, soft links do not increment reference counts and can point
// to objects that don't exist yet (dangling links are allowed).
//
// Parameters:
//   - linkPath: Path where the soft link will be created (e.g., "/group1/link_to_dataset")
//   - targetPath: Target path within file (e.g., "/group2/dataset1")
//
// Returns:
//   - error: Non-nil if link creation fails
//
// Behavior:
//   - Validates linkPath format (must be absolute path)
//   - Target path does NOT need to exist (dangling links allowed)
//   - Creates link message with soft link type
//   - Adds it to the parent group (see Limitations)
//   - Link stores target path as string (not object address)
//   - When accessed, target path is resolved dynamically
//
// Example:
//
//	fw, _ := hdf5.CreateForWrite("data.h5", hdf5.CreateTruncate)
//	defer fw.Close()
//
//	// Create dataset
//	fw.CreateDataset("/data/temperature", []float64{1.0, 2.0, 3.0})
//
//	// Create soft link (target exists)
//	err := fw.CreateSoftLink("/links/temp_link", "/data/temperature")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Create dangling link (target doesn't exist yet)
//	err = fw.CreateSoftLink("/links/future_link", "/data/future_dataset")
//	// This is allowed - target can be created later
//
// Limitations:
//   - Only a new-style parent (the root group of superblock v2/v3 files)
//     stores the link as a Link message, as the HDF5 library does. A
//     symbol table group gets an entry for a separate object header holding
//     the Link message, which the HDF5 library does not read as a link.
//   - No soft link resolution yet (reading soft links not implemented)
//   - No circular link detection
//
// HDF5 Spec: Section IV.A.2.f "Link Message" - Type 1 (Soft Link)
// Reference: H5L.c - H5Lcreate_soft().
func (fw *FileWriter) CreateSoftLink(linkPath, targetPath string) error {
	// Validate paths
	if err := validateLinkPath(linkPath); err != nil {
		return fmt.Errorf("invalid link path: %w", err)
	}

	if err := validateSoftLinkTargetPath(targetPath); err != nil {
		return fmt.Errorf("invalid target path: %w", err)
	}

	// Parse link path to find parent and link name
	parent, linkName := parsePath(linkPath)

	// Validate parent exists
	if parent != "" && parent != "/" {
		if _, exists := fw.groups[parent]; !exists {
			return fmt.Errorf("parent group %q does not exist (create it first)", parent)
		}
	}

	if len(targetPath) > 0xFFFF {
		return fmt.Errorf("target path too long: %d bytes (max 65535)", len(targetPath))
	}

	// A new-style parent (the root of superblock v2/v3 files) stores the
	// link as a Link message, like the HDF5 library.
	added, err := fw.linkMessageToParent(parent, linkName, func(order int64) []byte {
		return writer.EncodeSoftLinkMessage(linkName, targetPath, order)
	})
	if err != nil {
		return fmt.Errorf("failed to add soft link to parent group: %w", err)
	}
	if added {
		return nil
	}

	// Create soft link message
	linkMsg := &core.LinkMessage{
		Version:   1,
		Flags:     core.LinkFlagLinkTypeFieldBit | core.LinkFlagCharSetBit, // Bits 3 + 4 set
		Type:      core.LinkTypeSoft,
		CharSet:   0, // ASCII
		Name:      linkName,
		LinkValue: []byte(targetPath),
	}

	// Encode link message
	linkMsgData, err := core.EncodeLinkMessage(linkMsg, fw.file.sb)
	if err != nil {
		return fmt.Errorf("failed to encode soft link message: %w", err)
	}

	// Create object header writer for the soft link
	linkOHW := &core.ObjectHeaderWriter{
		Version: 2, // Use v2 for modern format
		Flags:   0,
		Messages: []core.MessageWriter{
			{
				Type: core.MsgLinkMessage, // 0x0006 - Link message
				Data: linkMsgData,
			},
		},
	}

	// Calculate object header size
	headerSize, err := calculateObjectHeaderSize(linkOHW)
	if err != nil {
		return fmt.Errorf("failed to calculate header size: %w", err)
	}

	// Allocate space for object header
	linkAddr, err := fw.writer.Allocate(headerSize)
	if err != nil {
		return fmt.Errorf("failed to allocate space for soft link object header: %w", err)
	}

	// Write object header
	_, err = linkOHW.WriteTo(fw.writer, linkAddr)
	if err != nil {
		return fmt.Errorf("failed to write soft link object header: %w", err)
	}

	// Add link to parent group's symbol table
	if err := fw.linkToParent(parent, linkName, linkAddr); err != nil {
		return fmt.Errorf("failed to add soft link to parent group: %w", err)
	}

	return nil
}

// validateSoftLinkTargetPath validates the target path format for soft links.
// Unlike hard links, target doesn't need to exist, but must be valid format.
func validateSoftLinkTargetPath(path string) error {
	if path == "" {
		return fmt.Errorf("target path cannot be empty")
	}
	if !strings.HasPrefix(path, "/") {
		return fmt.Errorf("target path must be absolute (start with '/'), got %q", path)
	}
	if strings.Contains(path, "//") {
		return fmt.Errorf("target path cannot contain consecutive slashes (got %q)", path)
	}
	return nil
}

// resolveSoftLink follows a soft link and returns the target object address.
//
// **STATUS: NOT YET IMPLEMENTED (MVP v0.11.5-beta)** - Planned for v0.12.0
//
// Planned functionality:
//   - Chain resolution (A→B→C): Follows multiple soft links
//   - Cycle detection: Detects circular references (A→B→A)
//   - Depth limiting: Maximum 32 resolution levels
//   - Dangling links: Returns error if target not found
//
// Parameters:
//   - linkAddr: Address of the soft link message
//   - visitedPaths: Set of visited paths for cycle detection (pass nil to start)
//
// Returns:
//   - uint64: Address of target object
//   - error: if target not found, cycle detected, or max depth exceeded
//
// Internal use only - will be called when accessing soft link in v0.12.0.
func (fw *FileWriter) resolveSoftLink(linkAddr uint64, visitedPaths map[string]bool) (uint64, error) { //nolint:unused // Future use in v0.12.0
	_ = linkAddr
	_ = visitedPaths
	return 0, fmt.Errorf("soft link resolution not yet implemented (planned for v0.12.0)")
}

// CreateExternalLink creates a link to an object in another HDF5 file.
// The link stores the external file path and object path within that file.
// Both files must exist when the external link is accessed (lazy resolution).
//
// Parameters:
//   - linkPath: Path where external link will be created (e.g., "/links/external1")
//   - fileName: External HDF5 file name (absolute or relative path)
//   - objectPath: Path to object within external file (e.g., "/dataset1")
//
// Returns:
//   - error: if validation fails or creation fails
//
// Examples:
//
//	fw.CreateExternalLink("/links/ext1", "other.h5", "/data/dataset1")
//	fw.CreateExternalLink("/links/ext2", "/absolute/path/file.h5", "/group1")
//
// Behavior:
//   - Validates all paths
//   - Creates link message with external link type
//   - Stores external file name and object path
//   - Adds it to the parent group (see Limitations)
//   - No file existence check (lazy resolution)
//
// Security:
//   - Path traversal prevention (blocks ".." in file names)
//   - File path stored as-is (absolute or relative)
//
// Limitations:
//   - Only a new-style parent (the root group of superblock v2/v3 files)
//     stores the link as a Link message, as the HDF5 library does. A
//     symbol table group gets an entry for a separate object header holding
//     the Link message, which the HDF5 library does not read as a link.
//   - No external link resolution yet (reading external links not implemented)
//   - No file caching or performance optimization
//
// HDF5 Spec: Section IV.A.2.f "Link Message" - Type 64 (External Link)
// Reference: H5Lcreate_external() in H5L.c.
func (fw *FileWriter) CreateExternalLink(linkPath, fileName, objectPath string) error {
	// Validate link path
	if err := validateLinkPath(linkPath); err != nil {
		return fmt.Errorf("invalid link path: %w", err)
	}

	// Validate file name
	if err := validateExternalFileName(fileName); err != nil {
		return fmt.Errorf("invalid file name: %w", err)
	}

	// Validate object path (use same rules as soft link target)
	if err := validateSoftLinkTargetPath(objectPath); err != nil {
		return fmt.Errorf("invalid object path: %w", err)
	}

	// Parse link path to find parent and link name
	parent, linkName := parsePath(linkPath)

	// Validate parent exists
	if parent != "" && parent != "/" {
		if _, exists := fw.groups[parent]; !exists {
			return fmt.Errorf("parent group %q does not exist (create it first)", parent)
		}
	}

	linkValue := core.EncodeExternalLinkValue(fileName, objectPath)
	if len(linkValue) > 0xFFFF {
		return fmt.Errorf("file name and object path too long: %d bytes (max 65535)", len(linkValue))
	}

	// A new-style parent (the root of superblock v2/v3 files) stores the
	// link as a Link message, like the HDF5 library.
	added, err := fw.linkMessageToParent(parent, linkName, func(order int64) []byte {
		return writer.EncodeExternalLinkMessage(linkName, fileName, objectPath, order)
	})
	if err != nil {
		return fmt.Errorf("failed to add external link to parent group: %w", err)
	}
	if added {
		return nil
	}

	// Create external link message
	linkMsg := &core.LinkMessage{
		Version:   1,
		Flags:     core.LinkFlagLinkTypeFieldBit | core.LinkFlagCharSetBit, // Bits 3 + 4 set
		Type:      core.LinkTypeExternal,
		CharSet:   0, // ASCII
		Name:      linkName,
		LinkValue: linkValue,
	}

	// Encode link message
	linkMsgData, err := core.EncodeLinkMessage(linkMsg, fw.file.sb)
	if err != nil {
		return fmt.Errorf("failed to encode external link message: %w", err)
	}

	// Create object header writer for the external link
	linkOHW := &core.ObjectHeaderWriter{
		Version: 2, // Use v2 for modern format
		Flags:   0,
		Messages: []core.MessageWriter{
			{
				Type: core.MsgLinkMessage, // 0x0006 - Link message
				Data: linkMsgData,
			},
		},
	}

	// Calculate object header size
	headerSize, err := calculateObjectHeaderSize(linkOHW)
	if err != nil {
		return fmt.Errorf("failed to calculate header size: %w", err)
	}

	// Allocate space for object header
	linkAddr, err := fw.writer.Allocate(headerSize)
	if err != nil {
		return fmt.Errorf("failed to allocate space for external link object header: %w", err)
	}

	// Write object header
	_, err = linkOHW.WriteTo(fw.writer, linkAddr)
	if err != nil {
		return fmt.Errorf("failed to write external link object header: %w", err)
	}

	// Add link to parent group's symbol table
	if err := fw.linkToParent(parent, linkName, linkAddr); err != nil {
		return fmt.Errorf("failed to add external link to parent group: %w", err)
	}

	return nil
}

// validateExternalFileName validates the external file name.
// Must not be empty, should have .h5 or .hdf5 extension.
func validateExternalFileName(fileName string) error {
	if fileName == "" {
		return fmt.Errorf("file name cannot be empty")
	}

	// Prevent path traversal attacks
	if strings.Contains(fileName, "..") {
		return fmt.Errorf("file name cannot contain '..' (path traversal)")
	}

	// Check extension (warning-level check, not strict)
	// In full v0.12.0 implementation, we would log a warning here if extension is not .h5 or .hdf5
	// For now, we accept any extension to support custom file formats

	return nil
}

// resolveExternalLink resolves an external link and returns the target object.
//
// **STATUS: NOT YET IMPLEMENTED (MVP v0.11.5-beta)** - Planned for v0.12.0
//
// Planned functionality:
//   - Open external HDF5 file (cache for performance)
//   - Resolve objectPath within external file
//   - Handle file not found, object not found errors
//   - Support absolute and relative file paths
//   - Prevent circular external references
//
// Parameters:
//   - linkAddr: Address of the external link message
//   - visitedFiles: Set of visited files for cycle detection (pass nil to start)
//
// Returns:
//   - *File: Opened external file (caller must cache/close)
//   - uint64: Address of target object in external file
//   - error: if file/object not found or cycle detected
//
// Internal use only - will be called when accessing external link in v0.12.0.
func (fw *FileWriter) resolveExternalLink(linkAddr uint64, visitedFiles map[string]bool) (*File, uint64, error) { //nolint:unused // Future use in v0.12.0
	_ = linkAddr
	_ = visitedFiles
	return nil, 0, fmt.Errorf("external link resolution not yet implemented in MVP v0.11.5-beta (planned for v0.12.0)")
}

// Note: The following methods are already implemented in group_write.go and are reused here:
// - parsePath(path string) (parent, name string)
// - linkToParent(parentPath, childName string, childAddr uint64) error
// - resolveObjectAddress(path string) (uint64, error)
