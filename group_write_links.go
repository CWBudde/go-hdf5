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

// addLink adds a hard link name → childAddr to the new-style group g.
func (fw *FileWriter) addLink(g *groupLinks, groupPath, name string, childAddr uint64) error {
	sb := fw.file.Superblock()
	link := writer.EncodedLink{Name: name, Message: writer.EncodeHardLinkMessage(name, childAddr, sb)}
	if g.dense() {
		return fw.insertDenseLink(g, groupPath, link)
	}

	compact, err := compactLinks(g, sb)
	if err != nil {
		return err
	}
	for _, l := range compact {
		if l.Name == name {
			return fmt.Errorf("link %q already exists in group %q", name, groupPath)
		}
	}

	if len(compact) < int(g.groupInfo.MaxCompactLinks()) {
		if err := core.AddMessageToObjectHeader(g.oh, core.MsgLinkMessage, link.Message); err != nil {
			return fmt.Errorf("add link message: %w", err)
		}
		return core.WriteObjectHeader(fw.writer, g.addr, g.oh, sb)
	}
	return fw.convertToDenseLinks(g, append(compact, link))
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
		links = append(links, writer.EncodedLink{Name: lm.Name, Message: m.Data})
	}
	return links, nil
}

// convertToDenseLinks moves the links of a compact group, plus the new one,
// to dense storage: it writes the fractal heap and name index, points the
// Link Info message at them and drops the Link messages from the header
// (H5G__obj_insert, "convert to dense storage").
func (fw *FileWriter) convertToDenseLinks(g *groupLinks, links []writer.EncodedLink) error {
	sb := fw.file.Superblock()
	heapAddr, btreeAddr, err := writer.WriteDenseLinkStorage(fw.writer, fw.writer.Allocator(), sb, links)
	if err != nil {
		return fmt.Errorf("write dense link storage: %w", err)
	}
	li := *g.linkInfo
	li.FractalHeapAddress = heapAddr
	li.NameBTreeAddress = btreeAddr
	data, err := core.EncodeLinkInfoMessage(&li, sb)
	if err != nil {
		return fmt.Errorf("encode link info message: %w", err)
	}
	g.linkInfoMsg.Data = data

	msgs := g.oh.Messages[:0]
	for _, m := range g.oh.Messages {
		if m.Type != core.MsgLinkMessage {
			msgs = append(msgs, m)
		}
	}
	g.oh.Messages = msgs
	return core.WriteObjectHeader(fw.writer, g.addr, g.oh, sb)
}

// loadDenseLinks loads the fractal heap and name index of a dense group.
func (fw *FileWriter) loadDenseLinks(g *groupLinks) (*structures.WritableFractalHeap, *structures.WritableBTreeV2, error) {
	sb := fw.file.Superblock()
	heap := writer.NewLinkHeap()
	if err := heap.LoadFromFile(fw.writer.Reader(), g.linkInfo.FractalHeapAddress, sb); err != nil {
		return nil, nil, fmt.Errorf("load link heap: %w", err)
	}
	btree := structures.NewWritableBTreeV2(0)
	if err := btree.LoadFromFile(fw.writer.Reader(), g.linkInfo.NameBTreeAddress, sb); err != nil {
		return nil, nil, fmt.Errorf("load link name index: %w", err)
	}
	return heap, btree, nil
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

// insertDenseLink adds a link to a dense group by rewriting its heap and name
// index in place (they move when they grow; the Link Info message keeps
// pointing at their headers).
func (fw *FileWriter) insertDenseLink(g *groupLinks, groupPath string, link writer.EncodedLink) error {
	sb := fw.file.Superblock()
	heap, btree, err := fw.loadDenseLinks(g)
	if err != nil {
		return err
	}
	_, exists, err := findDenseLink(heap, btree, link.Name, sb)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("link %q already exists in group %q", link.Name, groupPath)
	}
	if err := writer.InsertDenseLink(heap, btree, link); err != nil {
		return err
	}
	if err := heap.WriteAtWithAllocator(fw.writer, fw.writer.Allocator(), sb); err != nil {
		return fmt.Errorf("write link heap: %w", err)
	}
	if err := btree.WriteAtWithAllocator(fw.writer, fw.writer.Allocator(), sb); err != nil {
		return fmt.Errorf("write link name index: %w", err)
	}
	return nil
}

// lookupLink returns the address of the hard link name in the new-style
// group g.
func (fw *FileWriter) lookupLink(g *groupLinks, name string) (uint64, error) {
	sb := fw.file.Superblock()
	var lm *core.LinkMessage
	var found bool
	var err error
	if g.dense() {
		heap, btree, loadErr := fw.loadDenseLinks(g)
		if loadErr != nil {
			return 0, loadErr
		}
		lm, found, err = findDenseLink(heap, btree, name, sb)
	} else {
		lm, found, err = findCompactLink(g, name, sb)
	}
	if err != nil {
		return 0, err
	}
	if !found {
		return 0, fmt.Errorf("object not found: %s", name)
	}
	return lm.GetHardLinkAddress(sb)
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
