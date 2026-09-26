package hdf5

import (
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// orderedNames are attribute names in an order that is neither alphabetical
// nor the order of their name hashes.
var orderedNames = []string{"Conventions", "Version", "Zeta", "A", "B", "Y", "C", "X", "D", "W", "E", "V"}

// writeOrderedAttributes writes root and dataset attributes in the order of
// names: some at creation, one later in the same session and the rest in an
// OpenForWrite session.
func writeOrderedAttributes(t *testing.T, path string, names []string) {
	t.Helper()
	half := len(names) / 2
	opts := make([]interface{}, 0, half)
	dsOpts := make([]DatasetOption, 0, half)
	for _, n := range names[:half] {
		opts = append(opts, WithRootAttribute(n, "v "+n))
		dsOpts = append(dsOpts, WithAttribute(n, "v "+n))
	}
	fw, err := CreateForWrite(path, CreateTruncate, opts...)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/x", Float64, []uint64{2}, dsOpts...)
	require.NoError(t, err)
	require.NoError(t, ds.Write([]float64{1, 2}))
	root, err := fw.RootGroup()
	require.NoError(t, err)
	require.NoError(t, root.WriteAttribute(names[half], "v "+names[half]))
	require.NoError(t, ds.WriteAttribute(names[half], "v "+names[half]))
	require.NoError(t, fw.Close())

	fw, err = OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	root, err = fw.RootGroup()
	require.NoError(t, err)
	ds, err = fw.OpenDataset("/x")
	require.NoError(t, err)
	for _, n := range names[half+1:] {
		require.NoError(t, root.WriteAttribute(n, "v "+n))
		require.NoError(t, ds.WriteAttribute(n, "v "+n))
	}
	require.NoError(t, fw.Close())
}

// ncdumpAttributeOrder returns the attribute names ncdump -h lists for
// variable ("" for global attributes), in order.
func ncdumpAttributeOrder(t *testing.T, path, variable string) []string {
	t.Helper()
	ncdump, err := exec.LookPath("ncdump")
	if err != nil {
		t.Skip("ncdump not available")
	}
	out, err := exec.Command(ncdump, "-h", path).CombinedOutput()
	require.NoError(t, err, "ncdump failed:\n%s", out)
	re := regexp.MustCompile(`^\t+` + regexp.QuoteMeta(variable) + `:([A-Za-z0-9_]+) = `)
	var names []string
	for _, line := range strings.Split(string(out), "\n") {
		if m := re.FindStringSubmatch(line); m != nil {
			names = append(names, m[1])
		}
	}
	return names
}

// h5dumpAttributeOrder returns the attribute names of the object at objPath
// ("/" or a dataset) as h5dump lists them in creation order.
func h5dumpAttributeOrder(t *testing.T, path, objPath string) []string {
	t.Helper()
	h5dump := findH5Dump()
	if h5dump == "" {
		t.Skip("h5dump not available")
	}
	args := []string{"-q", "creation_order", "-A"}
	if objPath != "/" {
		args = append(args, "-d", objPath)
	}
	out, err := exec.Command(h5dump, append(args, path)...).CombinedOutput()
	require.NoError(t, err, "h5dump failed:\n%s", out)
	// The object's own attributes are indented once (those of the root's
	// members twice).
	const prefix = `   ATTRIBUTE "`
	var names []string
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, prefix) {
			names = append(names, strings.TrimSuffix(strings.TrimPrefix(line, prefix), `" {`))
		}
	}
	return names
}

// TestAttributeCreationOrderTracked checks that attributes are listed in
// the order they were written (ncdump, h5dump -q creation_order), compact
// and dense, for the root group and for a dataset.
func TestAttributeCreationOrderTracked(t *testing.T) {
	for _, n := range []int{5, len(orderedNames)} {
		names := orderedNames[:n]
		name := "compact"
		if n > MaxCompactAttributes {
			name = "dense"
		}
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "order.nc")
			writeOrderedAttributes(t, path, names)

			f, err := Open(path)
			require.NoError(t, err)
			for _, addr := range []uint64{f.Superblock().RootGroup, findDataset(f, "/x").Address()} {
				oh, err := core.ReadObjectHeader(f.Reader(), addr, f.Superblock())
				require.NoError(t, err)
				require.NotZero(t, oh.Flags&core.ObjectHeaderAttrCreationOrderTracked,
					"header at %d must track attribute creation order", addr)
				require.Zero(t, oh.Flags&core.ObjectHeaderAttrCreationOrderIndexed,
					"header at %d must not index attribute creation order", addr)
			}
			require.NoError(t, f.Close())

			require.Equal(t, names, h5dumpAttributeOrder(t, path, "/"))
			require.Equal(t, names, h5dumpAttributeOrder(t, path, "/x"))
			require.Equal(t, names, ncdumpAttributeOrder(t, path, ""))
			require.Equal(t, names, ncdumpAttributeOrder(t, path, "x"))
		})
	}
}

// copyFixture copies a testdata file into a temporary directory.
func copyFixture(t *testing.T, src string) string {
	t.Helper()
	in, err := os.Open(src)
	require.NoError(t, err)
	defer func() { _ = in.Close() }()
	dst := filepath.Join(t.TempDir(), filepath.Base(src))
	out, err := os.Create(dst)
	require.NoError(t, err)
	_, err = io.Copy(out, in)
	require.NoError(t, err)
	require.NoError(t, out.Close())
	return dst
}

// TestOpenForWriteNetCDFRoot adds a dataset and attributes to netCDF-C
// files, whose object headers track and index attribute creation order.
func TestOpenForWriteNetCDFRoot(t *testing.T) {
	t.Run("compact root attributes", func(t *testing.T) {
		path := copyFixture(t, "testdata/dimscales/netcdf4_dimscales.nc")
		fw, err := OpenForWrite(path, OpenReadWrite)
		require.NoError(t, err)
		ds, err := fw.CreateDataset("/added", Float64, []uint64{2})
		require.NoError(t, err)
		require.NoError(t, ds.Write([]float64{1, 2}))
		root, err := fw.RootGroup()
		require.NoError(t, err)
		require.NoError(t, root.WriteAttribute("Zfirst", "a"))
		require.NoError(t, root.WriteAttribute("Asecond", "b"))
		require.NoError(t, fw.Close())

		// _NCProperties (hidden by ncdump -h) was created first.
		require.Equal(t, []string{"_NCProperties", "Zfirst", "Asecond"}, h5dumpAttributeOrder(t, path, "/"))
		require.Equal(t, []string{"Zfirst", "Asecond"}, ncdumpAttributeOrder(t, path, ""))

		f, err := Open(path)
		require.NoError(t, err)
		defer func() { _ = f.Close() }()
		require.NotNil(t, findDataset(f, "/added"))
	})

	t.Run("dense indexed root attributes", func(t *testing.T) {
		path := copyFixture(t, "testdata/dense/netcdf4_many.nc")
		fw, err := OpenForWrite(path, OpenReadWrite)
		require.NoError(t, err)
		ds, err := fw.CreateDataset("/added", Float64, []uint64{2})
		require.NoError(t, err)
		require.NoError(t, ds.Write([]float64{1, 2}))
		root, err := fw.RootGroup()
		require.NoError(t, err)
		err = root.WriteAttribute("added", "a")
		require.True(t, errors.Is(err, ErrCreationOrderIndexNotSupported), "got %v", err)
		require.NoError(t, fw.Close())

		h5dump := findH5Dump()
		if h5dump == "" {
			t.Skip("h5dump not available")
		}
		out, err := exec.Command(h5dump, "-H", path).CombinedOutput()
		require.NoError(t, err, "h5dump failed:\n%s", out)
		require.Contains(t, string(out), `DATASET "added"`)
	})

	t.Run("indexed attributes going dense", func(t *testing.T) {
		path := copyFixture(t, "testdata/dimscales/netcdf4_dimscales.nc")
		fw, err := OpenForWrite(path, OpenReadWrite)
		require.NoError(t, err)
		root, err := fw.RootGroup()
		require.NoError(t, err)
		// _NCProperties plus 7 more fill the compact storage.
		for _, n := range orderedNames[:MaxCompactAttributes-1] {
			require.NoError(t, root.WriteAttribute(n, "v"))
		}
		err = root.WriteAttribute("one_too_many", "v")
		require.True(t, errors.Is(err, ErrCreationOrderIndexNotSupported), "got %v", err)
		require.NoError(t, fw.Close())
		require.Len(t, h5dumpAttributeOrder(t, path, "/"), MaxCompactAttributes)
	})
}

// TestOpenForWriteH5pyTrackOrder adds a dataset and attributes to a file
// h5py wrote with track_order=True and checks the order with h5py.
func TestOpenForWriteH5pyTrackOrder(t *testing.T) {
	python := requireH5py(t)
	path := filepath.Join(t.TempDir(), "tracked.h5")
	const create = `
import sys, h5py
with h5py.File(sys.argv[1], "w", track_order=True) as f:
    f.attrs["zulu"] = "1"
    f.attrs["alpha"] = "2"
    d = f.create_dataset("d", data=[1.0, 2.0], track_order=True)
    d.attrs["zulu"] = "1"
`
	out, err := exec.Command(python, "-c", create, path).CombinedOutput()
	require.NoError(t, err, "h5py failed:\n%s", out)

	fw, err := OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/added", Float64, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]float64{1, 2}))
	root, err := fw.RootGroup()
	require.NoError(t, err)
	require.NoError(t, root.WriteAttribute("mike", "3"))
	d, err := fw.OpenDataset("/d")
	require.NoError(t, err)
	require.NoError(t, d.WriteAttribute("bravo", "2"))
	require.NoError(t, fw.Close())

	const check = `
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    print(",".join(f.attrs.keys()))
    print(",".join(f["d"].attrs.keys()))
    print(",".join(f.keys()))
`
	out, err = exec.Command(python, "-c", check, path).CombinedOutput()
	require.NoError(t, err, "h5py failed:\n%s", out)
	require.Equal(t, "zulu,alpha,mike\nzulu,bravo\nd,added\n", string(out))
}
