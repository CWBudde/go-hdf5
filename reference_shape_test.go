package hdf5

import (
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// A reference attribute written from a one-element slice must read back
// as a slice, and one written from a single ObjectRef as a scalar.
func TestReferenceAttributeShapePreserved(t *testing.T) {
	path := filepath.Join(t.TempDir(), "refs.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	target, err := fw.CreateDataset("/target", Float64, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, target.Write([]float64{1, 2}))
	holder, err := fw.CreateDataset("/holder", Float64, []uint64{1})
	require.NoError(t, err)
	require.NoError(t, holder.Write([]float64{0}))
	require.NoError(t, holder.WriteAttribute("one", []ObjectRef{target.Reference()}))
	require.NoError(t, holder.WriteAttribute("scalar", target.Reference()))
	ref := target.Reference()
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { require.NoError(t, f.Close()) }()
	ds := findDataset(f, "/holder")
	require.NotNil(t, ds)

	one, err := ds.ReadAttribute("one")
	require.NoError(t, err)
	require.Equal(t, []ObjectRef{ref}, one)

	scalar, err := ds.ReadAttribute("scalar")
	require.NoError(t, err)
	require.Equal(t, ref, scalar)
}

// Concurrent first use of the object index must not race (run with -race).
func TestObjectIndexConcurrentInit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "idx.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	for _, name := range []string{"/a", "/b", "/c"} {
		ds, err := fw.CreateDataset(name, Float64, []uint64{1})
		require.NoError(t, err)
		require.NoError(t, ds.Write([]float64{1}))
	}
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { require.NoError(t, f.Close()) }()
	ds := findDataset(f, "/b")
	require.NotNil(t, ds)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			require.Equal(t, "/b", ds.Path())
			p, ok := f.ObjectPath(ds.Reference())
			require.True(t, ok)
			require.Equal(t, "/b", p)
			obj, err := f.Dereference(ds.Reference())
			require.NoError(t, err)
			require.NotNil(t, obj)
		}()
	}
	wg.Wait()
}
