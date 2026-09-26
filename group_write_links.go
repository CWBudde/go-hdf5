package hdf5

import (
	"fmt"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/structures"
	"github.com/cwbudde/go-hdf5/internal/writer"
)

// This file adds links to new-style groups (HDF5 1.8+), whose object header
// carries a Link Info message instead of a Symbol Table message. Like
// libhdf5, links are stored as compact Link messages in the object header
// up to the Group Info's max_compact (8) and in dense storage (a fractal
// heap of Link messages plus a v2 B-tree name index) above it.
//
// The root group of superblock v2/v3 files is a new-style group; see
// createRootGroupStructureV2.
//
// Reference: H5Gobj.c - H5G_obj_insert(), H5Gcompact.c, H5Gdense.c.

// groupLinks is a group's object header and the link storage it describes.
type groupLinks struct {
	addr uint64
	oh   *core.ObjectHeader

	// New-style groups.
	linkInfo    *core.LinkInfoMessage // nil for symbol table groups
	linkInfoMsg *core.HeaderMessage
	groupInfo   *core.GroupInfoMessage // libhdf5 defaults when absent

	// Symbol table groups.
	btreeAddr, heapAddr uint64
}

// dense reports whether a new-style group stores its links densely.
func (g *groupLinks) dense() bool {
	return g.linkInfo.FractalHeapAddress != 0 && g.linkInfo.FractalHeapAddress != undefinedAddress
}

// readGroupLinks reads the object header of the group at addr.
func (fw *FileWriter) readGroupLinks(addr uint64) (*groupLinks, error) {
	sb := fw.file.Superblock()
	oh, err := core.ReadObjectHeader(fw.writer.Reader(), addr, sb)
	if err != nil {
		return nil, fmt.Errorf("read group object header at %d: %w", addr, err)
	}
	g := &groupLinks{addr: addr, oh: oh, groupInfo: &core.GroupInfoMessage{}}
	for _, m := range oh.Messages {
		switch m.Type {
		case core.MsgLinkInfo:
			if g.linkInfo, err = core.ParseLinkInfoMessage(m.Data, sb); err != nil {
				return nil, fmt.Errorf("group at %d: %w", addr, err)
			}
			g.linkInfoMsg = m
		case core.MsgGroupInfo:
			if g.groupInfo, err = core.ParseGroupInfoMessage(m.Data); err != nil {
				return nil, fmt.Errorf("group at %d: %w", addr, err)
			}
		case core.MsgSymbolTable:
			n := int(sb.OffsetSize)
			if len(m.Data) < 2*n {
				return nil, fmt.Errorf("group at %d: symbol table message too short", addr)
			}
			g.btreeAddr = decodeAddress(m.Data[:n])
			g.heapAddr = decodeAddress(m.Data[n : 2*n])
		}
	}
	if g.linkInfo == nil && g.btreeAddr == 0 {
		return nil, fmt.Errorf("object at %d is not a group", addr)
	}
	return g, nil
}

// decodeAddress decodes a little-endian file address of 1-8 bytes.
func decodeAddress(b []byte) uint64 {
	var v uint64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

// addLink adds a hard link name → childAddr to the new-style group g. When
// the group tracks creation order, the link gets the next order and the Link
// Info message's maximum is advanced (H5G__obj_insert).
func (fw *FileWriter) addLink(g *groupLinks, groupPath, name string, childAddr uint64) error {
	sb := fw.file.Superblock()
	order := int64(-1)
	tracked := g.linkInfo.HasCreationOrderTracking()
	if tracked {
		order = g.linkInfo.MaxCreationOrder
	}
	link := writer.EncodedLink{Name: name, Message: writer.EncodeHardLinkMessage(name, childAddr, order, sb)}
	if tracked {
		link.CreationOrder = uint64(order) //nolint:gosec // G115: MaxCreationOrder is non-negative
	}

	if g.dense() {
		rebuilt, err := fw.insertDenseLink(g, groupPath, link)
		if err != nil {
			return err
		}
		if !tracked && !rebuilt {
			return nil // the object header is unchanged
		}
	} else if err := fw.addCompactLink(g, groupPath, link); err != nil {
		return err
	}
	if tracked {
		g.linkInfo.MaxCreationOrder++
	}
	return fw.writeGroupHeader(g)
}

// addCompactLink adds link to the header of the compact group g, or moves
// all links to dense storage when the group already holds max_compact
// links. The caller writes the header.
func (fw *FileWriter) addCompactLink(g *groupLinks, groupPath string, link writer.EncodedLink) error {
	compact, err := compactLinks(g, fw.file.Superblock())
	if err != nil {
		return err
	}
	for _, l := range compact {
		if l.Name == link.Name {
			return fmt.Errorf("link %q already exists in group %q", link.Name, groupPath)
		}
	}
	if len(compact) < int(g.groupInfo.MaxCompactLinks()) {
		if err := core.AddMessageToObjectHeader(g.oh, core.MsgLinkMessage, link.Message); err != nil {
			return fmt.Errorf("add link message: %w", err)
		}
		return nil
	}
	return fw.convertToDenseLinks(g, append(compact, link))
}

// writeGroupHeader re-encodes g's Link Info message and rewrites its object
// header in place.
func (fw *FileWriter) writeGroupHeader(g *groupLinks) error {
	sb := fw.file.Superblock()
	data, err := core.EncodeLinkInfoMessage(g.linkInfo, sb)
	if err != nil {
		return fmt.Errorf("encode link info message: %w", err)
	}
	g.linkInfoMsg.Data = data
	return core.WriteObjectHeader(fw.writer, g.addr, g.oh, sb)
}

// compactLinks returns the Link messages in g's object header.
func compactLinks(g *groupLinks, sb *core.Superblock) ([]writer.EncodedLink, error) {
	var links []writer.EncodedLink
	for _, m := range g.oh.Messages {
		if m.Type != core.MsgLinkMessage {
			continue
		}
		lm, err := core.ParseLinkMessage(m.Data, sb)
		if err != nil {
			return nil, fmt.Errorf("group at %d: %w", g.addr, err)
		}
		links = append(links, writer.EncodedLink{Name: lm.Name, Message: m.Data, CreationOrder: lm.CreationOrder})
	}
	return links, nil
}

// convertToDenseLinks moves the links of a compact group, plus the new one,
// to dense storage: it writes the fractal heap and indexes, points the Link
// Info message at them and drops the Link messages from the header
// (H5G__obj_insert, "convert to dense storage"). The caller writes the
// header.
func (fw *FileWriter) convertToDenseLinks(g *groupLinks, links []writer.EncodedLink) error {
	if err := fw.writeDenseLinks(g, links); err != nil {
		return err
	}
	msgs := g.oh.Messages[:0]
	for _, m := range g.oh.Messages {
		if m.Type != core.MsgLinkMessage {
			msgs = append(msgs, m)
		}
	}
	g.oh.Messages = msgs
	return nil
}

// writeDenseLinks writes new dense storage holding links for g: the fractal
// heap, the name index and, when g indexes creation order, the creation
// order index (H5G__dense_create). g's Link Info message is updated; the
// caller writes the header.
func (fw *FileWriter) writeDenseLinks(g *groupLinks, links []writer.EncodedLink) error {
	indexed := g.linkInfo.HasCreationOrderIndex()
	st, err := writer.WriteDenseLinkStorage(fw.writer, fw.writer.Allocator(), fw.file.Superblock(), links, indexed)
	if err != nil {
		return fmt.Errorf("write dense link storage: %w", err)
	}
	g.linkInfo.FractalHeapAddress = st.HeapAddress
	g.linkInfo.NameBTreeAddress = st.NameIndexAddress
	if indexed {
		g.linkInfo.CreationOrderBTreeAddress = st.CreationOrderIndexAddress
	}
	return nil
}

// editableDenseLinks is the dense link storage of a group, loaded for
// in-place modification.
type editableDenseLinks struct {
	heap   *structures.WritableFractalHeap
	names  *structures.WritableBTreeV2
	corder *structures.WritableBTreeV2 // nil when creation order is not indexed
}

// denseLinksEditable reports whether the dense link storage of g has the
// layout this library writes, so that a Link message of msgSize bytes can
// be added in place: a fractal heap whose root is a single direct block,
// without a free-space manager, huge objects or I/O filters, and indexes of
// depth 0.
//
// Storage written by the HDF5 C library does not qualify: its heaps track
// free space in a free-space manager that in-place additions would leave
// stale, and larger groups have indirect heap blocks and multi-level
// B-trees.
func (fw *FileWriter) denseLinksEditable(g *groupLinks, msgSize int) (bool, error) {
	sb := fw.file.Superblock()
	r := fw.writer.Reader()
	fh, err := structures.OpenFractalHeap(r, g.linkInfo.FractalHeapAddress, sb.LengthSize, sb.OffsetSize, sb.Endianness)
	if err != nil {
		return false, fmt.Errorf("open link heap: %w", err)
	}
	h := fh.Header
	if h.CurrentRowCount != 0 || h.IOFiltersLen != 0 || h.HugeObjCount != 0 ||
		(h.FreeSpaceSectionAddr != 0 && h.FreeSpaceSectionAddr != undefinedAddress) ||
		uint64(msgSize) > uint64(h.MaxManagedObjSize) { //nolint:gosec // G115: msgSize is a length
		return false, nil
	}
	ok, err := fw.linkIndexEditable(g.linkInfo.NameBTreeAddress, structures.BTreeV2TypeLinkNameIndex, 11)
	if !ok || err != nil || !g.linkInfo.HasCreationOrderIndex() {
		return ok, err
	}
	return fw.linkIndexEditable(g.linkInfo.CreationOrderBTreeAddress, structures.BTreeV2TypeLinkCreationOrderIndex, 15)
}

// linkIndexEditable reports whether the v2 B-tree at addr is a single leaf
// of type typ with records of recordSize bytes, as this library writes them.
func (fw *FileWriter) linkIndexEditable(addr uint64, typ uint8, recordSize uint16) (bool, error) {
	if addr == 0 || addr == undefinedAddress {
		return false, nil
	}
	info, err := core.ReadBTreeV2Info(fw.writer.Reader(), addr, fw.file.Superblock())
	if err != nil {
		return false, fmt.Errorf("read link index header: %w", err)
	}
	return info.Type == typ && info.RecordSize == recordSize && info.Depth == 0, nil
}

// loadDenseLinks loads the fractal heap and indexes of a dense group for
// in-place modification (see denseLinksEditable).
func (fw *FileWriter) loadDenseLinks(g *groupLinks) (*editableDenseLinks, error) {
	sb := fw.file.Superblock()
	r := fw.writer.Reader()
	d := &editableDenseLinks{heap: writer.NewLinkHeap(), names: structures.NewWritableBTreeV2(0)}
	if err := d.heap.LoadFromFile(r, g.linkInfo.FractalHeapAddress, sb); err != nil {
		return nil, fmt.Errorf("load link heap: %w", err)
	}
	if err := d.names.LoadFromFile(r, g.linkInfo.NameBTreeAddress, sb); err != nil {
		return nil, fmt.Errorf("load link name index: %w", err)
	}
	if g.linkInfo.HasCreationOrderIndex() {
		d.corder = structures.NewWritableLinkCreationOrderBTreeV2(0)
		if err := d.corder.LoadFromFile(r, g.linkInfo.CreationOrderBTreeAddress, sb); err != nil {
			return nil, fmt.Errorf("load link creation order index: %w", err)
		}
	}
	return d, nil
}

// readDenseLinks reads all Link messages of a dense group with the reader,
// which handles every layout the HDF5 C library writes.
func (fw *FileWriter) readDenseLinks(g *groupLinks) ([]writer.EncodedLink, error) {
	sb := fw.file.Superblock()
	r := fw.writer.Reader()
	fh, err := structures.OpenFractalHeap(r, g.linkInfo.FractalHeapAddress, sb.LengthSize, sb.OffsetSize, sb.Endianness)
	if err != nil {
		return nil, fmt.Errorf("open link heap: %w", err)
	}
	ids, err := structures.ReadBTreeV2LinkNameHeapIDs(r, g.linkInfo.NameBTreeAddress, sb)
	if err != nil {
		return nil, fmt.Errorf("read link name index: %w", err)
	}
	links := make([]writer.EncodedLink, 0, len(ids))
	for _, id := range ids {
		// Older go-hdf5 files use 8-byte heap IDs (see loadDenseGroupChildren).
		if n := int(fh.Header.HeapIDLen); n > len(id) {
			id = append(id, make([]byte, n-len(id))...)
		}
		msg, err := fh.ReadObjectSpecCompliant(id)
		if err != nil {
			return nil, fmt.Errorf("read link %x from heap: %w", id, err)
		}
		lm, err := core.ParseLinkMessage(msg, sb)
		if err != nil {
			return nil, fmt.Errorf("group at %d: %w", g.addr, err)
		}
		links = append(links, writer.EncodedLink{Name: lm.Name, Message: msg, CreationOrder: lm.CreationOrder})
	}
	return links, nil
}

// findDenseLink returns the Link message named name in dense storage.
// Names are compared via the heap objects because the name index matches by
// hash only.
func findDenseLink(heap *structures.WritableFractalHeap, btree *structures.WritableBTreeV2, name string, sb *core.Superblock) (lm *core.LinkMessage, found bool, err error) {
	for _, id := range btree.HeapIDsForName(name) {
		obj, err := heap.GetObject(id)
		if err != nil {
			return nil, false, fmt.Errorf("read link heap object: %w", err)
		}
		lm, err := core.ParseLinkMessage(obj, sb)
		if err != nil {
			return nil, false, err
		}
		if lm.Name == name {
			return lm, true, nil
		}
	}
	return nil, false, nil
}

// insertDenseLink adds a link to a dense group. Storage in the layout this
// library writes is extended in place (heap and indexes move when they
// grow; the Link Info message keeps pointing at their headers). Other
// storage, e.g. written by the HDF5 C library, is read completely and
// rewritten with the new link; rebuilt reports that g's Link Info message
// now points to the new storage (the old one is left unused).
func (fw *FileWriter) insertDenseLink(g *groupLinks, groupPath string, link writer.EncodedLink) (rebuilt bool, err error) {
	editable, err := fw.denseLinksEditable(g, len(link.Message))
	if err != nil {
		return false, err
	}
	if !editable {
		return true, fw.rebuildDenseLinks(g, groupPath, link)
	}

	sb := fw.file.Superblock()
	d, err := fw.loadDenseLinks(g)
	if err != nil {
		return false, err
	}
	_, exists, err := findDenseLink(d.heap, d.names, link.Name, sb)
	if err != nil {
		return false, err
	}
	if exists {
		return false, fmt.Errorf("link %q already exists in group %q", link.Name, groupPath)
	}
	if err := writer.InsertDenseLink(d.heap, d.names, d.corder, link); err != nil {
		return false, err
	}
	alloc := fw.writer.Allocator()
	if err := d.heap.WriteAtWithAllocator(fw.writer, alloc, sb); err != nil {
		return false, fmt.Errorf("write link heap: %w", err)
	}
	if err := d.names.WriteAtWithAllocator(fw.writer, alloc, sb); err != nil {
		return false, fmt.Errorf("write link name index: %w", err)
	}
	if d.corder != nil {
		if err := d.corder.WriteAtWithAllocator(fw.writer, alloc, sb); err != nil {
			return false, fmt.Errorf("write link creation order index: %w", err)
		}
	}
	return false, nil
}

// rebuildDenseLinks rewrites the dense storage of g with all its links plus
// link. The caller writes the header.
func (fw *FileWriter) rebuildDenseLinks(g *groupLinks, groupPath string, link writer.EncodedLink) error {
	links, err := fw.readDenseLinks(g)
	if err != nil {
		return err
	}
	for _, l := range links {
		if l.Name == link.Name {
			return fmt.Errorf("link %q already exists in group %q", link.Name, groupPath)
		}
	}
	return fw.writeDenseLinks(g, append(links, link))
}

// lookupLink returns the address of the hard link name in the new-style
// group g.
func (fw *FileWriter) lookupLink(g *groupLinks, name string) (uint64, error) {
	sb := fw.file.Superblock()
	if !g.dense() {
		lm, found, err := findCompactLink(g, name, sb)
		if err != nil {
			return 0, err
		}
		if !found {
			return 0, fmt.Errorf("object not found: %s", name)
		}
		return lm.GetHardLinkAddress(sb)
	}
	links, err := fw.readDenseLinks(g)
	if err != nil {
		return 0, err
	}
	for _, l := range links {
		if l.Name != name {
			continue
		}
		lm, err := core.ParseLinkMessage(l.Message, sb)
		if err != nil {
			return 0, err
		}
		return lm.GetHardLinkAddress(sb)
	}
	return 0, fmt.Errorf("object not found: %s", name)
}

// findCompactLink returns the Link message named name in g's object header.
func findCompactLink(g *groupLinks, name string, sb *core.Superblock) (lm *core.LinkMessage, found bool, err error) {
	for _, m := range g.oh.Messages {
		if m.Type != core.MsgLinkMessage {
			continue
		}
		lm, err := core.ParseLinkMessage(m.Data, sb)
		if err != nil {
			return nil, false, err
		}
		if lm.Name == name {
			return lm, true, nil
		}
	}
	return nil, false, nil
}
