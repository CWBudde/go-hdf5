package hdf5

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestDenseGroupReadBack writes dense groups (fractal heap + name-index
// B-tree) and reads the links back with this library and with h5py.
func TestDenseGroupReadBack(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dense_links.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	links := map[string]string{}
	var want []string
	for i := 0; i < 12; i++ {
		name := fmt.Sprintf("/d%02d", i)
		ds, err := fw.CreateDataset(name, Float64, []uint64{2})
		require.NoError(t, err)
		require.NoError(t, ds.Write([]float64{float64(i), 1}))
		links[fmt.Sprintf("l%02d", i)] = name
		want = append(want, fmt.Sprintf("/dense/l%02d", i))
	}
	require.NoError(t, fw.CreateDenseGroup("/dense", links))
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	var got []string
	f.Walk(func(p string, obj Object) {
		if d, ok := obj.(*Dataset); ok && filepath.Dir(p) == "/dense" {
			got = append(got, p)
			v, err := d.Read()
			require.NoError(t, err, p)
			require.Len(t, v, 2)
		}
	})
	sort.Strings(got)
	require.Equal(t, want, got)

	python, err := exec.LookPath("python3")
	if err != nil {
		return
	}
	if exec.Command(python, "-c", "import h5py").Run() != nil {
		return
	}
	out, err := exec.Command(python, "-c",
		`import sys, h5py; f = h5py.File(sys.argv[1], "r"); print(",".join(sorted(f["dense"].keys())))`,
		path).CombinedOutput()
	require.NoError(t, err, "%s", out)
	require.Equal(t, "l00,l01,l02,l03,l04,l05,l06,l07,l08,l09,l10,l11\n", string(out))

	// And a dense group written by libhdf5 (7-byte heap IDs).
	libPath := filepath.Join(t.TempDir(), "lib_dense.h5")
	out, err = exec.Command(python, "-c", `
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "w", libver="latest") as f:
    g = f.create_group("dense")
    for i in range(20):
        g.create_dataset("d%02d" % i, data=np.full(2, i, "f8"))
`, libPath).CombinedOutput()
	require.NoError(t, err, "%s", out)
	lf, err := Open(libPath)
	require.NoError(t, err)
	defer func() { _ = lf.Close() }()
	n := 0
	lf.Walk(func(p string, obj Object) {
		if d, ok := obj.(*Dataset); ok && filepath.Dir(p) == "/dense" {
			v, err := d.Read()
			require.NoError(t, err, p)
			require.Len(t, v, 2)
			n++
		}
	})
	require.Equal(t, 20, n)
}
