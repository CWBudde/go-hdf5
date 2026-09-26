package hdf5

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// attributeStorage reports the number of compact attribute messages and
// whether dense attribute storage is used by the object at addr.
func attributeStorage(t *testing.T, path string, addr uint64) (compact int, dense bool) {
	t.Helper()
	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	oh, err := core.ReadObjectHeader(f.Reader(), addr, f.Superblock())
	require.NoError(t, err)
	for _, m := range oh.Messages {
		switch m.Type {
		case core.MsgAttribute:
			compact++
		case core.MsgAttributeInfo:
			dense = true
		}
	}
	return compact, dense
}

// TestAttributesStayCompactWhenHeaderGrows checks that dataset attributes
// stay in the object header even when they no longer fit its first chunk,
// also beyond libhdf5's max_compact = 8 (libmysofa cannot read dense
// DIMENSION_LIST attributes).
func TestAttributesStayCompactWhenHeaderGrows(t *testing.T) {
	dir := t.TempDir()
	long := strings.Repeat("v", 120)

	for _, n := range []int{8, 9, 12} {
		path := filepath.Join(dir, fmt.Sprintf("attrs%d.h5", n))
		fw, err := CreateForWrite(path, CreateTruncate)
		require.NoError(t, err)
		ds, err := fw.CreateDataset("/d", Float64, []uint64{2})
		require.NoError(t, err)
		require.NoError(t, ds.Write([]float64{1, 2}))
		next, err := fw.CreateDataset("/after", Float64, []uint64{2})
		require.NoError(t, err)
		require.NoError(t, next.Write([]float64{3, 4}))
		for i := 0; i < n; i++ {
			require.NoError(t, ds.WriteAttribute(fmt.Sprintf("a%d", i), fmt.Sprintf("%s-%d", long, i)))
		}
		// Upserting an existing attribute keeps compact storage.
		require.NoError(t, ds.WriteAttribute("a0", "short"))
		addr := ds.Address()
		require.NoError(t, fw.Close())

		compact, dense := attributeStorage(t, path, addr)
		require.False(t, dense, "%d attributes must stay compact", n)
		require.Equal(t, n, compact)

		f, err := Open(path)
		require.NoError(t, err)
		var got *Dataset
		f.Walk(func(p string, obj Object) {
			if d, ok := obj.(*Dataset); ok && p == "/d" {
				got = d
			}
		})
		require.NotNil(t, got)
		v, err := got.ReadAttribute("a0")
		require.NoError(t, err)
		require.Equal(t, "short", v)
		v, err = got.ReadAttribute(fmt.Sprintf("a%d", n-1))
		require.NoError(t, err)
		require.Equal(t, fmt.Sprintf("%s-%d", long, n-1), v)
		require.NoError(t, f.Close())
	}
}

// TestGroupAttributesMoveToDenseStorage checks that group attributes, unlike
// dataset attributes, move to dense storage with the 9th, like libhdf5
// (max_compact = 8).
func TestGroupAttributesMoveToDenseStorage(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []int{MaxCompactAttributes, MaxCompactAttributes + 1} {
		path := filepath.Join(dir, fmt.Sprintf("group%d.h5", n))
		fw, err := CreateForWrite(path, CreateTruncate)
		require.NoError(t, err)
		g, err := fw.CreateGroup("/g")
		require.NoError(t, err)
		for i := 0; i < n; i++ {
			require.NoError(t, g.WriteAttribute(fmt.Sprintf("a%d", i), int32(i)))
		}
		addr := g.headerAddr
		require.NoError(t, fw.Close())

		compact, dense := attributeStorage(t, path, addr)
		if n <= MaxCompactAttributes {
			require.False(t, dense, "%d attributes must stay compact", n)
			require.Equal(t, n, compact)
		} else {
			require.True(t, dense, "%d attributes must use dense storage", n)
			require.Zero(t, compact)
		}
	}
}

// TestContinuationChunkReused checks that rewriting attributes of an object
// whose header spilled into a continuation chunk reuses that chunk instead
// of allocating a new one on every write.
func TestContinuationChunkReused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cont.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/d", Float64, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]float64{1, 2}))
	long := strings.Repeat("x", 200)
	require.NoError(t, ds.WriteAttribute("big", long))
	require.NoError(t, ds.WriteAttribute("n", int32(0)))
	require.NoError(t, fw.writer.Flush())
	before := fw.writer.EndOfFile()
	for i := 1; i < 50; i++ {
		require.NoError(t, ds.WriteAttribute("n", int32(i)))
	}
	require.Equal(t, before, fw.writer.EndOfFile(), "same-size rewrites must not allocate")
	require.NoError(t, fw.Close())

	st, err := os.Stat(path)
	require.NoError(t, err)
	// Everything but the 4 KiB global heap collection reserved at creation.
	require.Less(t, st.Size()-4096, int64(4096))
}
