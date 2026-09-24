package hdf5

import (
	"fmt"
	"sort"
	"strings"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/structures"
	"github.com/cwbudde/go-hdf5/internal/writer"
)

// GroupWriter represents an HDF5 group opened for writing.
// Groups organize datasets and other groups in a hierarchical structure.
//
// This type enables writing attributes to groups, similar to datasets.
// It provides a clean, object-oriented API consistent with DatasetWriter.
//
// Example:
//
//	fw, _ := hdf5.CreateForWrite("data.h5", hdf5.CreateTruncate)
//	defer fw.Close()
//
//	// Create group
//	group, _ := fw.CreateGroup("/mygroup")
//
//	// Write attributes to group
//	group.WriteAttribute("description", "My data group")
//	group.WriteAttribute("version", int32(1))
//
// Note: This is a write-only handle. For reading group contents, use
// the file-level Walk() or Group() methods after reopening the file.
type GroupWriter struct {
	// path is the full path of this group (e.g., "/mygroup" or "/data/experiments")
	path string

	// headerAddr is the address of the group's object header in the HDF5 file.
	// This is used for writing attributes and linking to this group.
	headerAddr uint64

	// file is a reference to the parent FileWriter.
	// This is needed for attribute operations and accessing file-level structures.
	file *FileWriter
}

// WriteAttribute writes an attribute to this group.
//
// Storage strategy (automatic):
//   - 0-7 attributes: Compact storage (object header messages)
//   - 8+ attributes: Dense storage (Fractal Heap + B-tree v2)
//
// Supported value types:
//   - Scalars: int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64
//   - Arrays: []int32, []float64, etc. (1D arrays only)
//   - Strings: string (fixed-length, converted to byte array)
//
// Parameters:
//   - name: Attribute name (ASCII, no null bytes)
//   - value: Attribute value (Go scalar, slice, or string)
//
// Returns:
//   - error: If attribute cannot be written
//
// Example:
//
//	group, _ := fw.CreateGroup("/mygroup")
//	group.WriteAttribute("MATLAB_class", "double")
//	group.WriteAttribute("MATLAB_complex", uint8(1))
//	group.WriteAttribute("description", "Temperature measurements")
//
// Limitations:
//   - No variable-length strings
//   - No compound types
//   - Attributes cannot be modified after creation (write-once)
//   - No attribute deletion
func (g *GroupWriter) WriteAttribute(name string, value interface{}) error {
	// Delegate to existing attribute writing infrastructure
	// This reuses the same code path as DatasetWriter.WriteAttribute
	return writeAttribute(g.file, g.headerAddr, name, value)
}

// Path returns the full path of this group.
//
// This can be used to display the group's location in the file hierarchy
// or for debugging purposes.
//
// Returns:
//   - string: The group's path (e.g., "/mygroup" or "/data/experiments")
//
// Example:
//
//	group, _ := fw.CreateGroup("/mygroup")
//	fmt.Println(group.Path()) // Output: /mygroup
func (g *GroupWriter) Path() string {
	return g.path
}

// validateGroupPath validates group path is not empty, starts with '/', and is not root.
func validateGroupPath(path string) error {
	if path == "" {
		return fmt.Errorf("group path cannot be empty")
	}
	if path[0] != '/' {
		return fmt.Errorf("group path must start with '/' (got %q)", path)
	}
	if path == "/" {
		return fmt.Errorf("root group already exists")
	}
	return nil
}

// createGroupStructures creates and writes the local heap, symbol table node, and B-tree for a group.
// Returns (heapAddr, stNodeAddr, btreeAddr, error).
func (fw *FileWriter) createGroupStructures() (uint64, uint64, uint64, error) {
	offsetSize := int(fw.file.sb.OffsetSize)

	// Create local heap
	heap := structures.NewLocalHeap(256)
	heapAddr, err := fw.writer.Allocate(heap.Size())
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to allocate heap: %w", err)
	}

	// Create symbol table node
	stNode := structures.NewSymbolTableNode(32)
	entrySize := 2*offsetSize + 4 + 4 + 16
	stNodeSize := uint64(8 + 32*entrySize) //nolint:gosec // Safe: small constant calculation
	stNodeAddr, err := fw.writer.Allocate(stNodeSize)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to allocate symbol table node: %w", err)
	}

	if err := stNode.WriteAt(fw.writer, stNodeAddr, uint8(offsetSize), 32, fw.file.sb.Endianness); err != nil { //nolint:gosec // Safe: offsetSize is 8
		return 0, 0, 0, fmt.Errorf("failed to write symbol table node: %w", err)
	}

	// Create B-tree
	btree := structures.NewBTreeNodeV1(0, 16)
	if err := btree.AddKey(0, stNodeAddr); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to add B-tree key: %w", err)
	}

	btreeSize := uint64(24 + (2*16+1)*offsetSize + 2*16*offsetSize) //nolint:gosec // Safe: small constant calculation
	btreeAddr, err := fw.writer.Allocate(btreeSize)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to allocate B-tree: %w", err)
	}

	if err := btree.WriteAt(fw.writer, btreeAddr, uint8(offsetSize), 16, fw.file.sb.Endianness); err != nil { //nolint:gosec // Safe: offsetSize is 8
		return 0, 0, 0, fmt.Errorf("failed to write B-tree: %w", err)
	}

	// Write heap
	if err := heap.WriteTo(fw.writer, heapAddr); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to write local heap: %w", err)
	}

	return heapAddr, stNodeAddr, btreeAddr, nil
}

// CreateGroup creates a new empty group in the HDF5 file.
// Groups organize datasets and other groups in a hierarchical structure.
//
// This method creates an empty group using symbol table format (old HDF5 format).
// For groups with many links, consider using CreateDenseGroup() or CreateGroupWithLinks().
//
// Parameters:
//   - path: Group path (must start with "/", e.g., "/data" or "/data/experiments")
//
// Returns:
//   - *GroupWriter: Handle for writing attributes to the group
//   - error: If creation fails
//
// Example:
//
//	fw, _ := hdf5.CreateForWrite("data.h5", hdf5.CreateTruncate)
//	defer fw.Close()
//
//	// Create root-level group
//	group, _ := fw.CreateGroup("/data")
//	group.WriteAttribute("description", "My data group")
//
//	// Create nested group
//	nested, _ := fw.CreateGroup("/data/experiments")
//	nested.WriteAttribute("MATLAB_class", "double")
//
// RootGroup returns a GroupWriter for the root "/" group.
// This allows writing attributes to the file's root group, which is required
// for formats like SOFA/netCDF that store global attributes at the root level.
//
// Example:
//
//	fw, _ := hdf5.CreateForWrite("data.h5", hdf5.CreateTruncate)
//	defer fw.Close()
//
//	// Write global attributes to root
//	root, _ := fw.RootGroup()
//	root.WriteAttribute("Conventions", "SOFA")
//	root.WriteAttribute("Version", "1.0")
//	root.WriteAttribute("Title", "My HRTF Dataset")
func (fw *FileWriter) RootGroup() (*GroupWriter, error) {
	return &GroupWriter{
		path:       "/",
		headerAddr: fw.rootGroupAddr,
		file:       fw,
	}, nil
}

// Limitations for MVP (v0.11.0-beta):
//   - Only symbol table structure (no indexed groups)
//   - No link creation time tracking
//   - Maximum 32 entries per group (symbol table node capacity)
//   - Parent group must exist (create parents first)
func (fw *FileWriter) CreateGroup(path string) (*GroupWriter, error) {
	// Validate path
	if err := validateGroupPath(path); err != nil {
		return nil, err
	}

	// Parse path into parent and name
	parent, name := parsePath(path)

	// Validate parent exists (if not root)
	if parent != "" && parent != "/" {
		if _, exists := fw.groups[parent]; !exists {
			return nil, fmt.Errorf("parent group %q does not exist (create it first)", parent)
		}
	}

	// Create group structures (heap, symbol table, B-tree)
	heapAddr, stNodeAddr, btreeAddr, err := fw.createGroupStructures()
	if err != nil {
		return nil, err
	}

	// Store group metadata for nested dataset linking
	fw.groups[path] = &GroupMetadata{
		heapAddr:   heapAddr,
		stNodeAddr: stNodeAddr,
		btreeAddr:  btreeAddr,
	}

	// Create object header for the group
	// Message 1: Symbol Table Message (type 0x11)
	stMsg := core.EncodeSymbolTableMessage(btreeAddr, heapAddr, int(fw.file.sb.OffsetSize), int(fw.file.sb.LengthSize))

	ohw := &core.ObjectHeaderWriter{
		Version: 2,
		Flags:   0,
		Messages: []core.MessageWriter{
			{Type: core.MsgSymbolTable, Data: stMsg},
		},
	}

	// Calculate object header size (prefix + messages + checksum)
	headerSize := ohw.Size()

	headerAddr, err := fw.writer.Allocate(headerSize)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate object header: %w", err)
	}

	// Write object header
	writtenSize, err := ohw.WriteTo(fw.writer, headerAddr)
	if err != nil {
		return nil, fmt.Errorf("failed to write object header: %w", err)
	}

	if writtenSize != headerSize {
		return nil, fmt.Errorf("header size mismatch: expected %d, wrote %d", headerSize, writtenSize)
	}

	// Link to parent group
	if err := fw.linkToParent(parent, name, headerAddr); err != nil {
		return nil, fmt.Errorf("failed to link to parent: %w", err)
	}

	// Return GroupWriter handle
	return &GroupWriter{
		path:       path,
		headerAddr: headerAddr,
		file:       fw,
	}, nil
}

// parsePath splits a path into parent directory and name.
// Examples:
//   - "/group1" → ("", "group1")
//   - "/data/experiments" → ("/data", "experiments")
//   - "/" → ("", "")
func parsePath(path string) (parent, name string) {
	if path == "/" {
		return "", ""
	}

	// Remove trailing slash if present
	path = strings.TrimSuffix(path, "/")

	// Find last slash
	lastSlash := strings.LastIndex(path, "/")
	if lastSlash == 0 {
		// Root-level path like "/group1"
		return "", path[1:] // Return ("", "group1")
	}

	// Nested path like "/data/experiments"
	return path[:lastSlash], path[lastSlash+1:]
}

// linkToParent links a child object to its parent group.
// Links the child by adding an entry to the parent's symbol table.
//
// Parameters:
//   - parentPath: Path to parent group ("" or "/" for root)
//   - childName: Name of the child object
//   - childAddr: Address of the child's object header
//
// Returns:
//   - error: If linking fails
func (fw *FileWriter) linkToParent(parentPath, childName string, childAddr uint64) error {
	// Get parent group metadata
	var heapAddr, btreeAddr uint64
	if parentPath == "" || parentPath == "/" {
		// Root group - use root metadata
		heapAddr = fw.rootHeapAddr
		btreeAddr = fw.rootBTreeAddr
	} else {
		// Non-root group - look up metadata
		meta, exists := fw.groups[parentPath]
		if !exists {
			return fmt.Errorf("parent group %q not found (create it first)", parentPath)
		}
		heapAddr = meta.heapAddr
		btreeAddr = meta.btreeAddr
	}

	// Step 1: Read the group B-tree (single leaf level) and all its symbol table nodes.
	snodAddrs, err := fw.readGroupBTreeChildren(btreeAddr)
	if err != nil {
		return err
	}
	entries, err := fw.readGroupEntries(snodAddrs)
	if err != nil {
		return err
	}

	// Step 2: Read existing local heap and resolve existing names.
	heap, err := fw.readLocalHeap(heapAddr)
	if err != nil {
		return fmt.Errorf("read local heap: %w", err)
	}
	names, err := resolveEntryNames(heap, entries)
	if err != nil {
		return err
	}
	for _, n := range names {
		if n == childName {
			return fmt.Errorf("link %q already exists in group %q", childName, parentPath)
		}
	}

	// The HDF5 library reserves heap offset 0 for the empty string: it is the
	// left-most key of the group B-tree (H5G__stab_create_components).
	if len(entries) == 0 {
		if _, err := heap.AddString(""); err != nil {
			return fmt.Errorf("reserve empty name in heap: %w", err)
		}
	}

	// Step 3: Add child name to heap and the new entry.
	nameOffset, err := heap.AddString(childName)
	if err != nil {
		return fmt.Errorf("add string to heap: %w", err)
	}
	names[nameOffset] = childName
	entries = append(entries, structures.SymbolTableEntry{
		LinkNameOffset: nameOffset,
		ObjectAddress:  childAddr,
	})

	// Entries are sorted by name across the nodes: the HDF5 library
	// binary-searches B-tree keys and node entries (H5G__node_found).
	sort.SliceStable(entries, func(i, j int) bool {
		return names[entries[i].LinkNameOffset] < names[entries[j].LinkNameOffset]
	})
	for i := range entries {
		// Scratch-pad data is not preserved by the node writer, so no
		// cached addresses may be advertised.
		entries[i].CacheType = 0
	}

	// Step 4: Write updated heap.
	if err := heap.WriteTo(fw.writer, heapAddr); err != nil {
		return fmt.Errorf("write heap: %w", err)
	}

	// Step 5: Distribute entries over symbol table nodes of at most 2*leafK
	// entries (the C library reads nodes with exactly that capacity) and
	// rewrite the group B-tree leaf.
	return fw.writeGroupSymbolTable(btreeAddr, snodAddrs, entries)
}

// readGroupEntries collects the entries of all given symbol table nodes.
func (fw *FileWriter) readGroupEntries(snodAddrs []uint64) ([]structures.SymbolTableEntry, error) {
	var entries []structures.SymbolTableEntry
	for _, a := range snodAddrs {
		node, err := fw.readSymbolTableNode(a)
		if err != nil {
			return nil, fmt.Errorf("read symbol table node: %w", err)
		}
		entries = append(entries, node.Entries...)
	}
	return entries, nil
}

// resolveEntryNames maps each entry's heap offset to its link name.
func resolveEntryNames(heap *structures.LocalHeap, entries []structures.SymbolTableEntry) (map[uint64]string, error) {
	names := make(map[uint64]string, len(entries)+1)
	for _, e := range entries {
		n, err := heap.GetString(e.LinkNameOffset)
		if err != nil {
			return nil, fmt.Errorf("read link name at heap offset %d: %w", e.LinkNameOffset, err)
		}
		names[e.LinkNameOffset] = n
	}
	return names, nil
}

// Group B-tree / symbol table node parameters. Files written by this library
// use the HDF5 defaults: group leaf node K = 4 (so a symbol table node holds
// up to 8 entries) and group internal node K = 16 (up to 32 children).
const (
	groupLeafK     = 4
	groupInternalK = 16
)

// readGroupBTreeChildren returns the symbol table node addresses referenced
// by a single-level (leaf) v1 group B-tree.
func (fw *FileWriter) readGroupBTreeChildren(btreeAddr uint64) ([]uint64, error) {
	sb := fw.file.sb
	o := int(sb.OffsetSize)
	l := int(sb.LengthSize)
	hdr := make([]byte, 8+2*o)
	if _, err := fw.writer.ReadAt(hdr, int64(btreeAddr)); err != nil { //nolint:gosec // Safe: file address
		return nil, fmt.Errorf("read group B-tree header: %w", err)
	}
	if string(hdr[0:4]) != "TREE" || hdr[4] != 0 {
		return nil, fmt.Errorf("no group B-tree at address %d", btreeAddr)
	}
	if hdr[5] != 0 {
		return nil, fmt.Errorf("group B-tree at %d has level %d; only single-level trees are supported", btreeAddr, hdr[5])
	}
	n := int(sb.Endianness.Uint16(hdr[6:8]))
	body := make([]byte, n*(l+o))
	if _, err := fw.writer.ReadAt(body, int64(btreeAddr)+int64(len(hdr))); err != nil { //nolint:gosec // Safe: file address
		return nil, fmt.Errorf("read group B-tree entries: %w", err)
	}
	addrs := make([]uint64, n)
	for i := 0; i < n; i++ {
		pos := i*(l+o) + l // skip key i
		addrs[i] = readUintN(body[pos:pos+o], sb)
	}
	return addrs, nil
}

func readUintN(b []byte, sb *core.Superblock) uint64 {
	switch len(b) {
	case 8:
		return sb.Endianness.Uint64(b)
	case 4:
		return uint64(sb.Endianness.Uint32(b))
	case 2:
		return uint64(sb.Endianness.Uint16(b))
	}
	return 0
}

func putUintN(b []byte, v uint64, sb *core.Superblock) {
	switch len(b) {
	case 8:
		sb.Endianness.PutUint64(b, v)
	case 4:
		sb.Endianness.PutUint32(b, uint32(v)) //nolint:gosec // G115: value fits the configured size
	case 2:
		sb.Endianness.PutUint16(b, uint16(v)) //nolint:gosec // G115: value fits the configured size
	}
}

// writeGroupSymbolTable writes sorted entries into symbol table nodes
// (reusing existingNodes, allocating more if needed) and rewrites the
// single-leaf group B-tree at btreeAddr to reference them.
func (fw *FileWriter) writeGroupSymbolTable(btreeAddr uint64, existingNodes []uint64, entries []structures.SymbolTableEntry) error {
	sb := fw.file.sb
	const perNode = 2 * groupLeafK
	numNodes := (len(entries) + perNode - 1) / perNode
	if numNodes == 0 {
		numNodes = 1
	}
	if numNodes > 2*groupInternalK {
		return fmt.Errorf("group is full: at most %d links are supported per symbol-table group", 2*groupInternalK*perNode)
	}

	entrySize := 2*int(sb.OffsetSize) + 4 + 4 + 16
	nodes := append([]uint64(nil), existingNodes...)
	for len(nodes) < numNodes {
		addr, err := fw.writer.Allocate(uint64(8 + perNode*entrySize)) //nolint:gosec // G115: small constant
		if err != nil {
			return fmt.Errorf("allocate symbol table node: %w", err)
		}
		nodes = append(nodes, addr)
	}

	o := int(sb.OffsetSize)
	l := int(sb.LengthSize)
	bt := make([]byte, 8+2*o+numNodes*(l+o)+l)
	copy(bt[0:4], "TREE")
	bt[4] = 0                                          // node type: group
	bt[5] = 0                                          // level: leaf
	sb.Endianness.PutUint16(bt[6:8], uint16(numNodes)) //nolint:gosec // G115: <= 32
	putUintN(bt[8:8+o], ^uint64(0), sb)                // left sibling: UNDEF
	putUintN(bt[8+o:8+2*o], ^uint64(0), sb)            // right sibling: UNDEF
	pos := 8 + 2*o
	putUintN(bt[pos:pos+l], 0, sb) // key 0: empty string
	pos += l

	for i := 0; i < numNodes; i++ {
		lo := i * perNode
		hi := lo + perNode
		if hi > len(entries) {
			hi = len(entries)
		}
		node := structures.NewSymbolTableNode(perNode)
		for _, e := range entries[lo:hi] {
			if err := node.AddEntry(e); err != nil {
				return fmt.Errorf("add entry to symbol table node: %w", err)
			}
		}
		if err := node.WriteAt(fw.writer, nodes[i], sb.OffsetSize, perNode, sb.Endianness); err != nil {
			return fmt.Errorf("write symbol table node: %w", err)
		}

		putUintN(bt[pos:pos+o], nodes[i], sb) // child i
		pos += o
		var maxKey uint64
		if hi > lo {
			maxKey = entries[hi-1].LinkNameOffset // key i+1: largest name in child i
		}
		putUintN(bt[pos:pos+l], maxKey, sb)
		pos += l
	}

	if err := fw.writer.WriteAtAddress(bt, btreeAddr); err != nil {
		return fmt.Errorf("write group B-tree: %w", err)
	}
	return nil
}

// readLocalHeap reads a local heap from the file at the specified address.
// This is used to modify the heap by adding new strings for linking.
//
// Parameters:
//   - addr: Address of the local heap in the file
//
// Returns:
//   - *structures.LocalHeap: The heap structure (writable)
//   - error: If read fails
func (fw *FileWriter) readLocalHeap(addr uint64) (*structures.LocalHeap, error) {
	// Load existing heap from disk
	heap, err := structures.LoadLocalHeap(fw.writer, addr, fw.file.sb)
	if err != nil {
		return nil, fmt.Errorf("load local heap: %w", err)
	}

	// Convert to writable mode (copies Data to internal strings buffer)
	if err := heap.PrepareForModification(); err != nil {
		return nil, fmt.Errorf("prepare heap for modification: %w", err)
	}

	// Set write-mode fields
	// Note: DataSegmentAddress is set by WriteTo(), not here
	heap.OffsetToHeadFreeList = 1 // MVP: no free list (1 = H5HL_FREE_NULL)

	return heap, nil
}

// readSymbolTableNode reads a symbol table node from the file at the specified address.
// This is used to modify the node by adding new entries for linking.
//
// Parameters:
//   - addr: Address of the symbol table node in the file
//
// Returns:
//   - *structures.SymbolTableNode: The node structure (writable)
//   - error: If read fails
func (fw *FileWriter) readSymbolTableNode(addr uint64) (*structures.SymbolTableNode, error) {
	// Use the existing ParseSymbolTableNode function from structures package
	return structures.ParseSymbolTableNode(fw.writer, addr, fw.file.sb)
}

// CreateDenseGroup creates new dense group (HDF5 1.8+ format).
//
// Dense groups are more efficient for large numbers of links (>8).
// They use fractal heap + B-tree v2 instead of symbol table.
//
// Parameters:
//   - name: Group name (must start with "/")
//   - links: Map of link_name → target_path
//
// Returns:
//   - error: Non-nil if creation fails
//
// Example:
//
//	err := fw.CreateDenseGroup("/large_group", map[string]string{
//	    "dataset1": "/data/dataset1",
//	    "dataset2": "/data/dataset2",
//	    // ... many links
//	})
//
// Reference: H5Gcreate.c - H5Gcreate2().
func (fw *FileWriter) CreateDenseGroup(name string, links map[string]string) error {
	// Validate name
	if !strings.HasPrefix(name, "/") {
		return fmt.Errorf("group name must start with /: %s", name)
	}

	// Create DenseGroupWriter
	dgw := writer.NewDenseGroupWriter(name)

	// Add all links
	for linkName, targetPath := range links {
		// Resolve target path to object header address
		targetAddr, err := fw.resolveObjectAddress(targetPath)
		if err != nil {
			return fmt.Errorf("failed to resolve target %s: %w", targetPath, err)
		}

		err = dgw.AddLink(linkName, targetAddr)
		if err != nil {
			return fmt.Errorf("failed to add link %s: %w", linkName, err)
		}
	}

	// Write dense group
	ohAddr, err := dgw.WriteToFile(fw.writer, fw.writer.Allocator(), fw.file.sb)
	if err != nil {
		return fmt.Errorf("failed to write dense group: %w", err)
	}

	// Link to parent
	parent, childName := parsePath(name)

	// Validate parent exists (if not root)
	if parent != "" && parent != "/" {
		if _, exists := fw.groups[parent]; !exists {
			return fmt.Errorf("parent group %q does not exist (create it first)", parent)
		}
	}

	if err := fw.linkToParent(parent, childName, ohAddr); err != nil {
		return fmt.Errorf("failed to link to parent: %w", err)
	}

	return nil
}

// resolveObjectAddress resolves object path to file address.
//
// This is a helper for link creation - looks up the target object's
// address in the file by its path. Supports both root-level and nested objects.
//
// Parameters:
//   - path: Object path (e.g., "/data/dataset1" or "/dataset1")
//
// Returns:
//   - uint64: File address of object header
//   - error: Non-nil if object not found or parent doesn't exist
func (fw *FileWriter) resolveObjectAddress(path string) (uint64, error) {
	// Handle root group
	if path == "/" {
		return fw.rootGroupAddr, nil
	}

	if !strings.HasPrefix(path, "/") {
		return 0, fmt.Errorf("path must start with /: %s", path)
	}

	// Parse path
	parent, name := parsePath(path)

	// Get parent group metadata
	var btreeAddr, heapAddr uint64
	if parent == "" || parent == "/" {
		// Root group
		btreeAddr = fw.rootBTreeAddr
		heapAddr = fw.rootHeapAddr
	} else {
		// Non-root group - look up metadata
		meta, exists := fw.groups[parent]
		if !exists {
			return 0, fmt.Errorf("parent group %q not found", parent)
		}
		btreeAddr = meta.btreeAddr
		heapAddr = meta.heapAddr
	}

	// Read the parent group's symbol table nodes to find the object
	snodAddrs, err := fw.readGroupBTreeChildren(btreeAddr)
	if err != nil {
		return 0, fmt.Errorf("failed to read symbol table: %w", err)
	}

	heap, err := fw.readLocalHeap(heapAddr)
	if err != nil {
		return 0, fmt.Errorf("failed to read local heap: %w", err)
	}

	for _, a := range snodAddrs {
		stNode, err := fw.readSymbolTableNode(a)
		if err != nil {
			return 0, fmt.Errorf("failed to read symbol table: %w", err)
		}

		// Search for object in symbol table
		for _, entry := range stNode.Entries {
			// Get link name from heap
			linkName, err := heap.GetString(entry.LinkNameOffset)
			if err != nil {
				continue
			}

			if linkName == name {
				return entry.ObjectAddress, nil
			}
		}
	}

	return 0, fmt.Errorf("object not found: %s", path)
}

// Dense group threshold (HDF5 default: switch to dense when >8 links).
const denseGroupThreshold = 8

// CreateGroupWithLinks creates group with automatic format selection.
//
// This method automatically chooses the most efficient storage format:
//   - Symbol table (old format) for ≤8 links (compact)
//   - Dense format (new format) for >8 links (scalable)
//
// This matches HDF5 1.8+ behavior: start compact, use dense when needed.
//
// Parameters:
//   - name: Group name (must start with "/")
//   - links: Map of link_name → target_path (can be empty)
//
// Returns:
//   - error: Non-nil if creation fails
//
// Example:
//
//	// Small group (will use symbol table)
//	fw.CreateGroupWithLinks("/small", map[string]string{
//	    "data1": "/dataset1",
//	    "data2": "/dataset2",
//	})
//
//	// Large group (will use dense format)
//	largeLinks := make(map[string]string)
//	for i := 0; i < 100; i++ {
//	    largeLinks[fmt.Sprintf("link%d", i)] = fmt.Sprintf("/dataset%d", i)
//	}
//	fw.CreateGroupWithLinks("/large", largeLinks)
//
// Reference: H5Gint.c - H5G_convert_to_dense().
func (fw *FileWriter) CreateGroupWithLinks(name string, links map[string]string) error {
	if len(links) > denseGroupThreshold {
		// Use dense format for large groups
		return fw.CreateDenseGroup(name, links)
	}

	// Use symbol table format for small groups
	// Create empty group first
	_, err := fw.CreateGroup(name)
	if err != nil {
		return err
	}

	// For MVP: linking is handled by CreateDenseGroup for dense groups
	// For symbol table groups, links would need to be added via linkToParent
	// This is a limitation of the MVP - symbol table groups can be created empty,
	// but adding links after creation requires manual linkToParent calls

	// Future: implement addLinkToGroup() to add links to existing symbol table groups

	if len(links) > 0 {
		return fmt.Errorf("adding links to symbol table groups not yet supported in MVP (group %s has %d links)", name, len(links))
	}

	return nil
}
