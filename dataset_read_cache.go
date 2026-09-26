package hdf5

import (
	"container/list"
	"fmt"
	"strconv"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/utils"
)

// Default bounds of the per-Dataset cache of decompressed chunks used by
// ReadSlice and ReadHyperslab; see Dataset.SetChunkCacheSize.
const (
	DefaultChunkCacheChunks = 8
	DefaultChunkCacheBytes  = 16 << 20
)

// datasetCache holds what ReadSlice and ReadHyperslab would otherwise read
// and parse again on every call: the dataset's datatype, dataspace, layout
// and filter pipeline, the chunk index of a chunked dataset, and recently
// used chunks (decompressed).
//
// A File from Open or OpenReader is read-only and assumes the file does not
// change while it is open, as it already does for the group hierarchy it
// loads at open time. Writing goes through FileWriter and DatasetWriter,
// which never share a Dataset value with a File, so nothing can make the
// cache stale in-process; File.Close drops it.
//
// All fields are guarded by Dataset.mu.
type datasetCache struct {
	meta   *parsedHyperslabMessages
	chunks *chunkIndex
	lru    chunkLRU
}

// chunkIndex maps the scaled coordinates of each stored chunk (in dataset
// rank) to its location in the file.
type chunkIndex struct {
	entries map[string]chunkIndexEntry
	coords  [][]uint64 // Scaled coordinates of every stored chunk.
}

// chunkLRU is a bounded least-recently-used cache of decoded chunks keyed
// by file address.
type chunkLRU struct {
	configured bool // false until the defaults or SetChunkCacheSize apply.
	maxChunks  int
	maxBytes   int64
	bytes      int64
	order      *list.List // Of *lruChunk, most recently used first.
	items      map[uint64]*list.Element
}

type lruChunk struct {
	address uint64
	data    []byte
}

// SetChunkCacheSize bounds the cache of decompressed chunks that
// ReadSlice and ReadHyperslab keep for this dataset: at most maxChunks
// chunks and maxBytes bytes of decoded data (DefaultChunkCacheChunks and
// DefaultChunkCacheBytes unless set). Repeated small reads that fall into
// the same chunks then decompress each chunk once. Zero or a negative
// value for either bound disables the cache and frees the chunks it
// holds; a chunk larger than maxBytes is never cached. The parsed
// metadata and chunk index are cached regardless.
//
// It is safe to call concurrently with reads of the dataset.
func (d *Dataset) SetChunkCacheSize(maxChunks int, maxBytes int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.cache.lru.configure(maxChunks, maxBytes)
}

// ChunkShape returns the chunk dimensions of a chunked dataset (one per
// dataset dimension) and true. For compact and contiguous datasets it
// returns nil and false. No data is read.
func (d *Dataset) ChunkShape() ([]uint64, bool, error) {
	meta, err := d.readMeta()
	if err != nil {
		return nil, false, err
	}
	if !meta.layout.IsChunked() {
		return nil, false, nil
	}
	rank := len(meta.dataspace.Dimensions)
	if len(meta.layout.ChunkSize) < rank {
		return nil, false, fmt.Errorf("dataset %s: chunk rank %d smaller than dataspace rank %d",
			d.name, len(meta.layout.ChunkSize), rank)
	}
	// The stored chunk dimensions carry the element size as an extra
	// trailing dimension.
	return append([]uint64(nil), meta.layout.ChunkSize[:rank]...), true, nil
}

// readMeta returns the dataset's parsed datatype, dataspace, layout and
// filter pipeline, reading the object header on first use only. The
// result is shared and must not be modified.
func (d *Dataset) readMeta() (*parsedHyperslabMessages, error) {
	d.mu.Lock()
	meta := d.cache.meta
	d.mu.Unlock()
	if meta != nil {
		return meta, nil
	}

	header, err := core.ReadObjectHeader(d.file.osFile, d.address, d.file.sb)
	if err != nil {
		return nil, fmt.Errorf("failed to read object header: %w", err)
	}
	messages, err := extractHyperslabMessages(header)
	if err != nil {
		return nil, err
	}
	meta, err = parseHyperslabMessages(messages, d.file.sb)
	if err != nil {
		return nil, err
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.cache.meta == nil {
		d.cache.meta = meta
	}
	return d.cache.meta, nil
}

// readChunkIndex returns the chunk index of a chunked dataset, walking the
// chunk B-tree on first use only.
func (d *Dataset) readChunkIndex(layout *core.DataLayoutMessage, rank int) (*chunkIndex, error) {
	d.mu.Lock()
	idx := d.cache.chunks
	d.mu.Unlock()
	if idx != nil {
		return idx, nil
	}

	chunkDims := layout.ChunkSize
	btreeNode, err := core.ParseBTreeV1Node(
		d.file.osFile,
		layout.DataAddress,
		d.file.sb.OffsetSize,
		len(chunkDims),
		chunkDims,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to parse chunk B-tree: %w", err)
	}
	allChunks, err := btreeNode.CollectAllChunks(d.file.osFile, d.file.sb.OffsetSize, chunkDims)
	if err != nil {
		return nil, fmt.Errorf("failed to get chunk index: %w", err)
	}

	idx = &chunkIndex{
		entries: make(map[string]chunkIndexEntry, len(allChunks)),
		coords:  make([][]uint64, 0, len(allChunks)),
	}
	for _, chunk := range allChunks {
		if len(chunk.Key.Scaled) < rank {
			return nil, fmt.Errorf("chunk key has %d dimensions, need %d", len(chunk.Key.Scaled), rank)
		}
		coords := chunk.Key.Scaled[:rank:rank]
		key := chunkCoordsToKey(coords)
		if _, dup := idx.entries[key]; !dup {
			idx.coords = append(idx.coords, coords)
		}
		idx.entries[key] = chunkIndexEntry{
			address: chunk.Address,
			nbytes:  uint64(chunk.Key.Nbytes),
		}
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.cache.chunks == nil {
		d.cache.chunks = idx
	}
	return d.cache.chunks, nil
}

// readChunk returns the decoded data of the chunk at entry, from the chunk
// cache when present. The returned slice is shared with the cache and must
// not be modified.
func (d *Dataset) readChunk(entry chunkIndexEntry, filterPipeline *core.FilterPipelineMessage,
	expectedChunkBytes uint64,
) ([]byte, error) {
	d.mu.Lock()
	data, ok := d.cache.lru.get(entry.address)
	d.mu.Unlock()
	if ok {
		return data, nil
	}

	data, err := utils.ReadAtChecked(d.file.osFile, entry.address, entry.nbytes, "chunk data")
	if err != nil {
		return nil, fmt.Errorf("failed to read chunk data: %w", err)
	}
	// Decompress if needed, never producing more than one chunk's worth of data.
	if filterPipeline != nil {
		data, err = filterPipeline.ApplyFiltersLimit(data, expectedChunkBytes)
		if err != nil {
			return nil, fmt.Errorf("failed to apply filters: %w", err)
		}
	}

	d.mu.Lock()
	d.cache.lru.put(entry.address, data)
	d.mu.Unlock()
	return data, nil
}

// dropCache forgets everything cached for the dataset.
func (d *Dataset) dropCache() {
	d.mu.Lock()
	defer d.mu.Unlock()
	lru := d.cache.lru
	d.cache = datasetCache{}
	if lru.configured {
		d.cache.lru.configure(lru.maxChunks, lru.maxBytes)
	}
}

// configure sets the bounds and evicts chunks beyond them.
func (c *chunkLRU) configure(maxChunks int, maxBytes int64) {
	c.configured = true
	c.maxChunks, c.maxBytes = max(maxChunks, 0), max(maxBytes, 0)
	if c.maxChunks == 0 || c.maxBytes == 0 {
		c.order, c.items, c.bytes = nil, nil, 0
		return
	}
	c.evict()
}

// get returns the chunk at address and marks it most recently used.
func (c *chunkLRU) get(address uint64) ([]byte, bool) {
	el, ok := c.items[address]
	if !ok {
		return nil, false
	}
	c.order.MoveToFront(el)
	return el.Value.(*lruChunk).data, true //nolint:forcetypeassert // only *lruChunk is stored
}

// put adds the chunk at address unless it alone exceeds the byte bound.
func (c *chunkLRU) put(address uint64, data []byte) {
	if !c.configured {
		c.configure(DefaultChunkCacheChunks, DefaultChunkCacheBytes)
	}
	size := int64(len(data))
	if c.maxChunks == 0 || size > c.maxBytes {
		return
	}
	if _, ok := c.items[address]; ok {
		return // Read concurrently by another goroutine.
	}
	if c.items == nil {
		c.order, c.items = list.New(), make(map[uint64]*list.Element)
	}
	c.items[address] = c.order.PushFront(&lruChunk{address: address, data: data})
	c.bytes += size
	c.evict()
}

// evict removes least recently used chunks until the bounds hold.
func (c *chunkLRU) evict() {
	for c.order != nil && (c.order.Len() > c.maxChunks || c.bytes > c.maxBytes) {
		el := c.order.Back()
		ch := c.order.Remove(el).(*lruChunk) //nolint:forcetypeassert // only *lruChunk is stored
		delete(c.items, ch.address)
		c.bytes -= int64(len(ch.data))
	}
}

// overlapping returns the scaled coordinates of the stored chunks that
// overlap the selection's bounding box, in row-major order. When the box
// holds fewer chunk positions than the file stores chunks, the positions
// are looked up in the index; otherwise every stored chunk is tested.
func (idx *chunkIndex) overlapping(sel *HyperslabSelection, chunkDims, datasetDims []uint64) [][]uint64 {
	ndims := len(sel.Start)
	firstChunk := make([]uint64, ndims)
	lastChunk := make([]uint64, ndims)
	positions := uint64(1)
	for i := 0; i < ndims; i++ {
		firstChunk[i] = sel.Start[i] / chunkDims[i]
		endPos := sel.Start[i] + (sel.Count[i]-1)*sel.Stride[i] + sel.Block[i] - 1
		if endPos >= datasetDims[i] {
			endPos = datasetDims[i] - 1
		}
		lastChunk[i] = endPos / chunkDims[i]
		if positions <= uint64(len(idx.coords)) {
			positions *= lastChunk[i] - firstChunk[i] + 1
		}
	}
	if positions > uint64(len(idx.coords)) {
		return findOverlappingStoredChunks(firstChunk, lastChunk, idx.coords)
	}

	result := make([][]uint64, 0, positions)
	coord := append([]uint64(nil), firstChunk...)
	for {
		if _, ok := idx.entries[chunkCoordsToKey(coord)]; ok {
			result = append(result, append([]uint64(nil), coord...))
		}
		i := ndims - 1
		for ; i >= 0; i-- {
			if coord[i] < lastChunk[i] {
				coord[i]++
				break
			}
			coord[i] = firstChunk[i]
		}
		if i < 0 {
			return result
		}
	}
}

// chunkCoordsToKey converts chunk coordinates to a string key for map lookup.
func chunkCoordsToKey(coords []uint64) string {
	buf := make([]byte, 0, 8*len(coords))
	for i, c := range coords {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = strconv.AppendUint(buf, c, 10)
	}
	return string(buf)
}
