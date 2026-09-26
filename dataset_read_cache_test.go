package hdf5

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// writeChunkedDeflate writes a [rows, cols] float64 dataset "/x" with value
// row*cols+col, chunked as chunk and gzip-compressed, and opens the file.
func writeChunkedDeflate(tb testing.TB, rows, cols uint64, chunk []uint64) (*File, *Dataset) {
	tb.Helper()
	path := filepath.Join(tb.TempDir(), "chunked_deflate.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(tb, err)
	dw, err := fw.CreateDataset("/x", Float64, []uint64{rows, cols}, WithChunkDims(chunk), WithGZIPCompression(6))
	require.NoError(tb, err)
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = float64(i)
	}
	require.NoError(tb, dw.Write(data))
	require.NoError(tb, fw.Close())

	f, err := Open(path)
	require.NoError(tb, err)
	tb.Cleanup(func() { _ = f.Close() })
	ds, ok := findDatasetByName(f, "x")
	require.True(tb, ok)
	return f, ds
}

// cachedChunks returns the number of chunks and bytes in the chunk cache.
func cachedChunks(ds *Dataset) (int, int64) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	if ds.cache.lru.order == nil {
		return 0, 0
	}
	return ds.cache.lru.order.Len(), ds.cache.lru.bytes
}

// TestReadSlice_ChunkCacheBounded reads rows of a chunked, deflated dataset
// in random order under several cache bounds and checks the values and that
// the cache never exceeds its bounds.
func TestReadSlice_ChunkCacheBounded(t *testing.T) {
	const rows, cols = 64, 40
	_, ds := writeChunkedDeflate(t, rows, cols, []uint64{4, 10}) // 16 x 4 chunks of 320 B.
	const chunkBytes = 4 * 10 * 8

	for _, tc := range []struct {
		chunks    int
		bytes     int64
		wantMaxCh int
	}{
		{DefaultChunkCacheChunks, DefaultChunkCacheBytes, DefaultChunkCacheChunks},
		{3, 1 << 20, 3},
		{100, 5 * chunkBytes, 5},
		{100, chunkBytes - 1, 0}, // Chunks larger than the byte bound are never cached.
		{0, 0, 0},                // Disabled.
	} {
		t.Run(fmt.Sprintf("%d_chunks_%d_bytes", tc.chunks, tc.bytes), func(t *testing.T) {
			ds.SetChunkCacheSize(tc.chunks, tc.bytes)
			rng := rand.New(rand.NewSource(1))
			for range 300 {
				r := uint64(rng.Intn(rows))
				c := uint64(rng.Intn(cols - 5))
				got, err := ds.ReadSlice([]uint64{r, c}, []uint64{1, 5})
				require.NoError(t, err)
				want := make([]float64, 5)
				for k := range want {
					want[k] = float64(r*cols + c + uint64(k))
				}
				require.Equal(t, want, got)
				n, b := cachedChunks(ds)
				require.LessOrEqual(t, n, tc.wantMaxCh)
				require.LessOrEqual(t, b, tc.bytes)
			}
		})
	}

	// Shrinking the bounds evicts at once.
	ds.SetChunkCacheSize(8, 1<<20)
	_, err := ds.ReadSlice([]uint64{0, 0}, []uint64{rows, cols})
	require.NoError(t, err)
	n, _ := cachedChunks(ds)
	require.Equal(t, 8, n)
	ds.SetChunkCacheSize(2, 1<<20)
	n, _ = cachedChunks(ds)
	require.Equal(t, 2, n)
}

// TestReadSlice_PropertiesCacheSizes repeats the ReadSlice/ReadHyperslab
// property checks on chunked datasets with the chunk cache disabled and
// holding a single chunk.
func TestReadSlice_PropertiesCacheSizes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "slices_cache.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	var chunked []sliceTestShape
	for _, sh := range sliceTestShapes {
		if sh.chunk == nil {
			continue
		}
		chunked = append(chunked, sh)
		total := uint64(1)
		for _, n := range sh.dims {
			total *= n
		}
		data := make([]float64, total)
		for i := range data {
			data[i] = sliceTestValue(uint64(i))
		}
		ds, err := fw.CreateDataset("/"+sh.name, Float64, sh.dims, WithChunkDims(sh.chunk), WithGZIPCompression(1))
		require.NoError(t, err, sh.name)
		require.NoError(t, ds.Write(data), sh.name)
	}
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer f.Close()

	for _, size := range []int{0, 1} {
		for _, sh := range chunked {
			t.Run(fmt.Sprintf("%s_cache%d", sh.name, size), func(t *testing.T) {
				ds, ok := findDatasetByName(f, sh.name)
				require.True(t, ok)
				ds.SetChunkCacheSize(size, 1<<20)
				checkSliceProperties(t, ds, sh.dims, sliceTestValue)
			})
		}
	}
}

// TestReadSlice_Concurrent reads one chunked dataset from several
// goroutines (run with -race).
func TestReadSlice_Concurrent(t *testing.T) {
	const rows, cols = 32, 16
	_, ds := writeChunkedDeflate(t, rows, cols, []uint64{4, 4})
	ds.SetChunkCacheSize(3, 1<<20)
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := readRowsConcurrently(ds, g, rows, cols); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// readRowsConcurrently is one goroutine of TestReadSlice_Concurrent.
func readRowsConcurrently(ds *Dataset, g int, rows, cols uint64) error {
	for i := range 100 {
		r := uint64(g*7+i*5) % rows
		got, err := ds.ReadSlice([]uint64{r, 0}, []uint64{1, cols})
		if err != nil {
			return err
		}
		for k, v := range got.([]float64) {
			if v != float64(r*cols+uint64(k)) {
				return fmt.Errorf("row %d [%d] = %v", r, k, v)
			}
		}
		if i%25 == 0 {
			ds.SetChunkCacheSize(2+g%3, 1<<20)
		}
	}
	return nil
}

// TestReadSlice_CloseDropsCache checks that a closed file is not read from
// the cache.
func TestReadSlice_CloseDropsCache(t *testing.T) {
	f, ds := writeChunkedDeflate(t, 8, 8, []uint64{4, 4})
	_, err := ds.ReadSlice([]uint64{0, 0}, []uint64{8, 8})
	require.NoError(t, err)
	n, _ := cachedChunks(ds)
	require.Equal(t, 4, n)
	require.NoError(t, f.Close())
	n, _ = cachedChunks(ds)
	require.Zero(t, n)
	_, err = ds.ReadSlice([]uint64{0, 0}, []uint64{1, 1})
	require.Error(t, err)
}

// TestDataset_ChunkShape checks ChunkShape for chunked, contiguous and
// compact datasets.
func TestDataset_ChunkShape(t *testing.T) {
	path := filepath.Join(t.TempDir(), "chunk_shape.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	for name, opts := range map[string][]DatasetOption{
		"chunked":    {WithChunkDims([]uint64{2, 1, 4})},
		"contiguous": nil,
	} {
		dw, err := fw.CreateDataset("/"+name, Float64, []uint64{4, 2, 8}, opts...)
		require.NoError(t, err)
		require.NoError(t, dw.Write(make([]float64, 64)))
	}
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer f.Close()
	for name, want := range map[string][]uint64{"chunked": {2, 1, 4}, "contiguous": nil} {
		ds, ok := findDatasetByName(f, name)
		require.True(t, ok)
		got, chunked, err := ds.ChunkShape()
		require.NoError(t, err)
		require.Equal(t, want != nil, chunked, name)
		require.Equal(t, want, got, name)
	}
}

// BenchmarkReadSlice_ChunkedDeflate reads single rows of a chunked,
// deflated [4096, 256] dataset (chunks of 64 x 256) in order, as a
// streaming consumer would, with the default chunk cache and without.
func BenchmarkReadSlice_ChunkedDeflate(b *testing.B) {
	const rows, cols = 4096, 256
	for _, bc := range []struct {
		name   string
		chunks int
	}{{"cache", DefaultChunkCacheChunks}, {"nocache", 0}} {
		b.Run(bc.name, func(b *testing.B) {
			_, ds := writeChunkedDeflate(b, rows, cols, []uint64{64, cols})
			ds.SetChunkCacheSize(bc.chunks, DefaultChunkCacheBytes)
			b.SetBytes(cols * 8)
			b.ReportAllocs()
			r := uint64(0)
			for b.Loop() {
				if _, err := ds.ReadSlice([]uint64{r, 0}, []uint64{1, cols}); err != nil {
					b.Fatal(err)
				}
				r = (r + 1) % rows
			}
		})
	}
}

// BenchmarkReadSlice_Contiguous reads single rows of a contiguous
// [4096, 256] dataset in order; the object header is parsed once.
func BenchmarkReadSlice_Contiguous(b *testing.B) {
	const rows, cols = 4096, 256
	path := filepath.Join(b.TempDir(), "contiguous.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(b, err)
	dw, err := fw.CreateDataset("/x", Float64, []uint64{rows, cols})
	require.NoError(b, err)
	require.NoError(b, dw.Write(make([]float64, rows*cols)))
	require.NoError(b, fw.Close())
	f, err := Open(path)
	require.NoError(b, err)
	defer f.Close()
	ds, ok := findDatasetByName(f, "x")
	require.True(b, ok)

	b.SetBytes(cols * 8)
	b.ReportAllocs()
	r := uint64(0)
	for b.Loop() {
		if _, err := ds.ReadSlice([]uint64{r, 0}, []uint64{1, cols}); err != nil {
			b.Fatal(err)
		}
		r = (r + 1) % rows
	}
}
