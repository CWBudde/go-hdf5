package hdf5

import (
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// mustFindDataset returns the dataset at path in f.
func mustFindDataset(t *testing.T, f *File, path string) *Dataset {
	t.Helper()
	var ds *Dataset
	f.Walk(func(p string, obj Object) {
		if d, ok := obj.(*Dataset); ok && p == path {
			ds = d
		}
	})
	require.NotNil(t, ds, "dataset %s not found", path)
	return ds
}

// grid returns the row-major values v[i][j] = i*100 + j.
func grid(rows, cols int) []float64 {
	out := make([]float64, 0, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out = append(out, float64(i*100+j))
		}
	}
	return out
}

// checkGridReads reads a 7x10 grid dataset fully and through a hyperslab
// that crosses chunk boundaries.
func checkGridReads(t *testing.T, ds *Dataset) {
	t.Helper()
	all, err := ds.Read()
	require.NoError(t, err)
	require.Equal(t, grid(7, 10), all)

	sl, err := ds.ReadSlice([]uint64{2, 3}, []uint64{4, 6})
	require.NoError(t, err)
	var want []float64
	for i := 2; i < 6; i++ {
		for j := 3; j < 9; j++ {
			want = append(want, float64(i*100+j))
		}
	}
	require.Equal(t, want, sl)
}

// TestChunkedMultiChunkReadBack reads multi-chunk datasets (edge chunks,
// several chunks per dimension) written by this library and by libhdf5,
// fully and by hyperslab. Chunk keys store element offsets; the readers
// must convert them to chunk indices exactly once.
func TestChunkedMultiChunkReadBack(t *testing.T) {
	t.Run("go-hdf5", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "grid.h5")
		fw, err := CreateForWrite(path, CreateTruncate)
		require.NoError(t, err)
		ds, err := fw.CreateDataset("/g", Float64, []uint64{7, 10}, WithChunkDims([]uint64{3, 4}))
		require.NoError(t, err)
		require.NoError(t, ds.Write(grid(7, 10)))
		dz, err := fw.CreateDataset("/gz", Float64, []uint64{7, 10}, WithChunkDims([]uint64{3, 4}),
			WithShuffle(), WithFletcher32(), WithGZIPCompression(4))
		require.NoError(t, err)
		require.NoError(t, dz.Write(grid(7, 10)))
		require.NoError(t, fw.Close())

		f, err := Open(path)
		require.NoError(t, err)
		defer func() { _ = f.Close() }()
		checkGridReads(t, mustFindDataset(t, f, "/g"))
		checkGridReads(t, mustFindDataset(t, f, "/gz"))
	})

	t.Run("libhdf5", func(t *testing.T) {
		python, err := exec.LookPath("python3")
		if err != nil {
			t.Skip("python3 not available")
		}
		if err := exec.Command(python, "-c", "import h5py").Run(); err != nil {
			t.Skip("h5py not available")
		}
		// libver v108 writes version 2 filter pipeline messages (no names
		// for predefined filters, unpadded names for IDs >= 256 such as LZF).
		script := `
import sys, h5py, numpy as np
g = np.array([[i*100+j for j in range(10)] for i in range(7)], dtype="f8")
for name, libver in (("v1.h5", "earliest"), ("v2.h5", "v108")):
    with h5py.File(sys.argv[1] + "/" + name, "w", libver=(libver, "v108")) as f:
        f.create_dataset("g", data=g, chunks=(3, 4))
        f.create_dataset("gz", data=g, chunks=(3, 4), shuffle=True, fletcher32=True, compression="gzip")
        f.create_dataset("fz", data=g, chunks=(3, 4), fletcher32=True, compression="gzip")
        f.create_dataset("lzf", data=g, chunks=(3, 4), compression="lzf", fletcher32=True)
        f.create_dataset("lzfrep", data=np.arange(2000) % 7, dtype="f8", chunks=(500,), compression="lzf")
`
		dir := t.TempDir()
		out, err := exec.Command(python, "-c", script, dir).CombinedOutput()
		require.NoError(t, err, "%s", out)
		for _, file := range []string{"v1.h5", "v2.h5"} {
			t.Run(file, func(t *testing.T) {
				f, err := Open(filepath.Join(dir, file))
				require.NoError(t, err)
				defer func() { _ = f.Close() }()
				for _, name := range []string{"/g", "/gz", "/fz", "/lzf"} {
					t.Run(name, func(t *testing.T) { checkGridReads(t, mustFindDataset(t, f, name)) })
				}
				rep, err := mustFindDataset(t, f, "/lzfrep").Read()
				require.NoError(t, err)
				require.Len(t, rep, 2000)
				for i, v := range rep {
					require.Equal(t, float64(i%7), v, "lzfrep[%d]", i)
				}
			})
		}
	})
}

// TestChunkedNeverWrittenReadBack reopens a chunked dataset that was
// created but never written (undefined chunk index address) and reads it as
// all zeros, fully and by hyperslab.
func TestChunkedNeverWrittenReadBack(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty_chunked.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	_, err = fw.CreateDataset("/e", Float64, []uint64{7, 10}, WithChunkDims([]uint64{3, 4}))
	require.NoError(t, err)
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	ds := mustFindDataset(t, f, "/e")
	all, err := ds.Read()
	require.NoError(t, err)
	require.Equal(t, make([]float64, 70), all)
	sl, err := ds.ReadSlice([]uint64{1, 1}, []uint64{3, 3})
	require.NoError(t, err)
	require.Equal(t, make([]float64, 9), sl)
}
