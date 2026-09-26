package hdf5

import (
	"fmt"
	"math/rand"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// sliceTestShape describes a dataset used by the ReadSlice property tests.
type sliceTestShape struct {
	name  string
	dims  []uint64
	chunk []uint64 // nil: contiguous layout
}

var sliceTestShapes = []sliceTestShape{
	{"c1", []uint64{13}, nil},
	{"c2", []uint64{5, 7}, nil},
	{"c3", []uint64{3, 2, 8}, nil},
	{"c4", []uint64{2, 3, 4, 5}, nil},
	{"k1", []uint64{13}, []uint64{4}},
	{"k2", []uint64{5, 7}, []uint64{2, 3}},
	{"k3", []uint64{3, 2, 8}, []uint64{2, 1, 3}},
	{"k4", []uint64{2, 3, 4, 5}, []uint64{1, 2, 3, 2}},
}

// sliceTestValue is the value stored at row-major linear index i.
func sliceTestValue(i uint64) float64 { return float64(i) + 0.5 }

// referenceHyperslab extracts a hyperslab from a full row-major read.
func referenceHyperslab(full []float64, dims []uint64, sel *HyperslabSelection) []float64 {
	ndims := len(dims)
	stride, block := sel.Stride, sel.Block
	if stride == nil {
		stride = ones(ndims)
	}
	if block == nil {
		block = ones(ndims)
	}
	var out []float64
	coords := make([]uint64, ndims)
	var walk func(d int)
	walk = func(d int) {
		if d == ndims {
			out = append(out, full[calculateLinearOffset(coords, dims)])
			return
		}
		for c := uint64(0); c < sel.Count[d]; c++ {
			for b := uint64(0); b < block[d]; b++ {
				coords[d] = sel.Start[d] + c*stride[d] + b
				walk(d + 1)
			}
		}
	}
	walk(0)
	return out
}

func ones(n int) []uint64 {
	v := make([]uint64, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

// forEachStartCount calls fn for every start/count pair that fits in dims.
func forEachStartCount(dims []uint64, fn func(start, count []uint64)) {
	ndims := len(dims)
	start := make([]uint64, ndims)
	count := make([]uint64, ndims)
	var walk func(d int)
	walk = func(d int) {
		if d == ndims {
			fn(append([]uint64(nil), start...), append([]uint64(nil), count...))
			return
		}
		for s := uint64(0); s < dims[d]; s++ {
			for c := uint64(1); s+c <= dims[d]; c++ {
				start[d], count[d] = s, c
				walk(d + 1)
			}
		}
	}
	walk(0)
}

// randomHyperslab returns a random valid selection with stride and block.
func randomHyperslab(rng *rand.Rand, dims []uint64) *HyperslabSelection {
	ndims := len(dims)
	sel := &HyperslabSelection{
		Start:  make([]uint64, ndims),
		Count:  make([]uint64, ndims),
		Stride: make([]uint64, ndims),
		Block:  make([]uint64, ndims),
	}
	for d, n := range dims {
		block := 1 + uint64(rng.Int63n(int64(n)))
		stride := block + uint64(rng.Int63n(int64(n-block)+1))
		start := uint64(rng.Int63n(int64(n-block) + 1))
		// Largest count with start + (count-1)*stride + block <= n.
		maxCount := (n-start-block)/stride + 1
		sel.Start[d], sel.Stride[d], sel.Block[d] = start, stride, block
		sel.Count[d] = 1 + uint64(rng.Int63n(int64(maxCount)))
	}
	return sel
}

// checkSliceProperties compares ReadSlice (every start/count combination)
// and ReadHyperslab (random stride/block selections) against a full read.
// value(i) is the expected value at row-major linear index i.
func checkSliceProperties(t *testing.T, ds *Dataset, dims []uint64, value func(uint64) float64) {
	t.Helper()
	full, err := ds.Read()
	require.NoError(t, err)
	total := uint64(1)
	for _, n := range dims {
		total *= n
	}
	require.Len(t, full, int(total))
	for i, v := range full {
		require.InDelta(t, value(uint64(i)), v, 0, "full read [%d]", i)
	}

	n := 0
	forEachStartCount(dims, func(start, count []uint64) {
		got, err := ds.ReadSlice(start, count)
		require.NoError(t, err, "ReadSlice(%v, %v)", start, count)
		want := referenceHyperslab(full, dims, &HyperslabSelection{Start: start, Count: count})
		require.Equal(t, want, got, "ReadSlice(%v, %v)", start, count)
		n++
	})
	require.Positive(t, n)

	rng := rand.New(rand.NewSource(int64(len(dims)) * 7919))
	for i := 0; i < 200; i++ {
		sel := randomHyperslab(rng, dims)
		desc := fmt.Sprintf("ReadHyperslab(start=%v count=%v stride=%v block=%v)", sel.Start, sel.Count, sel.Stride, sel.Block)
		want := referenceHyperslab(full, dims, sel)
		got, err := ds.ReadHyperslab(sel)
		require.NoError(t, err, desc)
		require.Equal(t, want, got, desc)
	}
}

// TestReadSlice_Properties writes contiguous and chunked datasets of rank 1-4
// and checks every start/count selection against a full read. Regression
// test for 3-D+ contiguous selections that start inside a row, which used to
// return zeros.
func TestReadSlice_Properties(t *testing.T) {
	path := filepath.Join(t.TempDir(), "slices.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	for _, sh := range sliceTestShapes {
		total := uint64(1)
		for _, n := range sh.dims {
			total *= n
		}
		data := make([]float64, total)
		for i := range data {
			data[i] = sliceTestValue(uint64(i))
		}
		var opts []DatasetOption
		if sh.chunk != nil {
			opts = append(opts, WithChunkDims(sh.chunk))
		}
		ds, err := fw.CreateDataset("/"+sh.name, Float64, sh.dims, opts...)
		require.NoError(t, err, sh.name)
		require.NoError(t, ds.Write(data), sh.name)
	}
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer f.Close()

	for _, sh := range sliceTestShapes {
		t.Run(sh.name, func(t *testing.T) {
			ds, ok := findDatasetByName(f, sh.name)
			require.True(t, ok)
			checkSliceProperties(t, ds, sh.dims, sliceTestValue)
		})
	}
}

// TestReadSlice_ContiguousInsideRow is the minimal E5 reproduction: a
// selection on a [3,2,8] contiguous dataset that starts inside a row.
func TestReadSlice_ContiguousInsideRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "inside_row.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/x", Float64, []uint64{3, 2, 8})
	require.NoError(t, err)
	data := make([]float64, 48)
	for i := range data {
		data[i] = float64(i + 1)
	}
	require.NoError(t, ds.Write(data))
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer f.Close()
	d, ok := findDatasetByName(f, "x")
	require.True(t, ok)

	got, err := d.ReadSlice([]uint64{0, 0, 5}, []uint64{1, 1, 3})
	require.NoError(t, err)
	require.Equal(t, []float64{6, 7, 8}, got)

	got, err = d.ReadSlice([]uint64{2, 1, 5}, []uint64{1, 1, 3})
	require.NoError(t, err)
	require.Equal(t, []float64{46, 47, 48}, got)

	// Full last dimension, partial middle dimension: rows are not adjacent.
	got, err = d.ReadSlice([]uint64{0, 1, 0}, []uint64{2, 1, 8})
	require.NoError(t, err)
	require.Equal(t, []float64{9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32}, got)
}

// TestReadSlice_H5pyProperties runs the slice property checks on datasets
// written by h5py (HDF5 C library): contiguous, chunked, chunked+gzip and
// compact layouts with float64, float32 and int32 data.
func TestReadSlice_H5pyProperties(t *testing.T) {
	python := requireH5py(t)
	const script = `
import sys
import numpy as np
import h5py
shapes = {1: (13,), 2: (5, 7), 3: (3, 2, 8), 4: (2, 3, 4, 5)}
chunks = {1: (4,), 2: (2, 3), 3: (2, 1, 3), 4: (1, 2, 3, 2)}
with h5py.File(sys.argv[1], "w") as f:
    for r, shape in shapes.items():
        n = int(np.prod(shape))
        base = (np.arange(n, dtype=np.float64) + 0.5).reshape(shape)
        f.create_dataset("c%d" % r, data=base)
        f.create_dataset("k%d" % r, data=base, chunks=chunks[r])
        f.create_dataset("z%d" % r, data=base.astype(np.float32), chunks=chunks[r], compression="gzip")
        sid = h5py.h5s.create_simple(shape)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_layout(h5py.h5d.COMPACT)
        dset = h5py.h5d.create(f.id, b"m%d" % r, h5py.h5t.NATIVE_DOUBLE, sid, dcpl=dcpl)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, np.ascontiguousarray(base))
        # int32 contiguous: value i + 0.5 is not representable, store 2*i+1
        # and compare against the scaled reference.
        f.create_dataset("i%d" % r, data=(2 * np.arange(n, dtype=np.int32) + 1).reshape(shape))
`
	path := filepath.Join(t.TempDir(), "h5py_slices.h5")
	out, err := exec.Command(python, "-c", script, path).CombinedOutput()
	require.NoError(t, err, "%s", out)

	f, err := Open(path)
	require.NoError(t, err)
	defer f.Close()

	dimsByRank := map[int][]uint64{1: {13}, 2: {5, 7}, 3: {3, 2, 8}, 4: {2, 3, 4, 5}}
	for rank := 1; rank <= 4; rank++ {
		for _, prefix := range []string{"c", "k", "z", "m"} {
			name := fmt.Sprintf("%s%d", prefix, rank)
			t.Run(name, func(t *testing.T) {
				ds, ok := findDatasetByName(f, name)
				require.True(t, ok)
				checkSliceProperties(t, ds, dimsByRank[rank], sliceTestValue)
			})
		}
		name := fmt.Sprintf("i%d", rank)
		t.Run(name, func(t *testing.T) {
			ds, ok := findDatasetByName(f, name)
			require.True(t, ok)
			checkSliceProperties(t, ds, dimsByRank[rank], func(i uint64) float64 { return float64(2*i + 1) })
		})
	}
}
