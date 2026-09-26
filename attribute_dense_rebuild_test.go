package hdf5

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// attributeValues returns the attributes of the object at path ("/" for the
// root group) with their values formatted by fmt.Sprint.
func attributeValues(t *testing.T, file, path string) map[string]string {
	t.Helper()
	f, err := Open(file)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	var attrs []*core.Attribute
	if path == "/" {
		attrs, err = f.Root().Attributes()
	} else {
		ds := findDataset(f, path)
		require.NotNil(t, ds, "dataset %s not found", path)
		attrs, err = ds.Attributes()
	}
	require.NoError(t, err)
	values := make(map[string]string, len(attrs))
	for _, a := range attrs {
		v, err := a.ReadValue()
		require.NoError(t, err, "attribute %q", a.Name)
		require.NotContains(t, values, a.Name, "duplicate attribute")
		values[a.Name] = fmt.Sprint(v)
	}
	return values
}

// requireNoCompactAttributesBesideDense checks that an object with dense
// attribute storage keeps no attribute messages in its object header, where
// libhdf5 would not look for them.
func requireNoCompactAttributesBesideDense(t *testing.T, file string, addr uint64) {
	t.Helper()
	f, err := Open(file)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	oh, err := core.ReadObjectHeader(f.Reader(), addr, f.Superblock())
	require.NoError(t, err)
	if !hasDenseAttributeStorage(oh, f.Superblock()) {
		return
	}
	for _, m := range oh.Messages {
		require.NotEqual(t, core.MsgAttribute, m.Type, "attribute message beside dense storage")
	}
}

// h5pyAttributeCount returns the number of attributes h5py (libhdf5) reads
// on the object at path, skipping the test without h5py.
func h5pyAttributeCount(t *testing.T, file, path string) int {
	t.Helper()
	python := requireH5py(t)
	const script = `
import sys, json, h5py
with h5py.File(sys.argv[1], "r") as f:
    a = f[sys.argv[2]].attrs
    print(json.dumps({k: str(a[k]) for k in a.keys()}))
`
	out, err := exec.Command(python, "-c", script, file, path).CombinedOutput()
	require.NoError(t, err, "h5py failed:\n%s", out)
	var attrs map[string]string
	require.NoError(t, json.Unmarshal(out, &attrs), "output: %s", out)
	return len(attrs)
}

// TestOpenForWriteLibhdf5DenseAttributes adds, changes and deletes
// attributes in dense storage written by libhdf5 (h5py): a heap with a
// single direct block and a free-space manager, and larger heaps with an
// indirect root block and B-trees of depth 1.
func TestOpenForWriteLibhdf5DenseAttributes(t *testing.T) {
	for _, tc := range []struct {
		fixture, prefix string
	}{
		{"testdata/dense/h5py_attrs_small.h5", "%02d"},
		{"testdata/dense/h5py_many.h5", "%03d"},
	} {
		t.Run(filepath.Base(tc.fixture), func(t *testing.T) {
			path := copyFixture(t, tc.fixture)
			rootBefore := attributeValues(t, path, "/")
			dataBefore := attributeValues(t, path, "/data")

			fw, err := OpenForWrite(path, OpenReadWrite)
			require.NoError(t, err)
			root, err := fw.RootGroup()
			require.NoError(t, err)
			require.NoError(t, root.WriteAttribute("added", "new value"))
			changed := "attr" + fmt.Sprintf(tc.prefix, 3)
			require.NoError(t, root.WriteAttribute(changed, "changed"))
			require.NoError(t, root.WriteAttribute("added2", "second"))

			ds, err := fw.OpenDataset("/data")
			require.NoError(t, err)
			require.NoError(t, ds.WriteAttribute("added", 99.0))
			dchanged := "dattr" + fmt.Sprintf(tc.prefix, 3)
			ddeleted := "dattr" + fmt.Sprintf(tc.prefix, 5)
			require.NoError(t, ds.WriteAttribute(dchanged, 33.0))
			require.NoError(t, ds.DeleteAttribute(ddeleted))
			require.NoError(t, ds.WriteAttribute("added2", 98.0))
			require.NoError(t, fw.Close())

			rootWant := rootBefore
			rootWant["added"] = "new value"
			rootWant[changed] = "changed"
			rootWant["added2"] = "second"
			require.Equal(t, rootWant, attributeValues(t, path, "/"))

			dataWant := dataBefore
			dataWant["added"] = "99"
			dataWant[dchanged] = "33"
			dataWant["added2"] = "98"
			delete(dataWant, ddeleted)
			require.Equal(t, dataWant, attributeValues(t, path, "/data"))

			f, err := Open(path)
			require.NoError(t, err)
			rootAddr, dataAddr := f.Superblock().RootGroup, findDataset(f, "/data").Address()
			require.NoError(t, f.Close())
			requireNoCompactAttributesBesideDense(t, path, rootAddr)
			requireNoCompactAttributesBesideDense(t, path, dataAddr)

			if h5dump := findH5Dump(); h5dump != "" {
				out, err := exec.Command(h5dump, "-A", path).CombinedOutput()
				require.NoError(t, err, "h5dump failed:\n%s", out)
				require.Contains(t, string(out), `ATTRIBUTE "added2"`)
				require.Contains(t, string(out), `"changed"`)
			}
			require.Equal(t, len(rootWant), h5pyAttributeCount(t, path, "/"))
			require.Equal(t, len(dataWant), h5pyAttributeCount(t, path, "/data"))
		})
	}
}

// bigAttributeValue returns a []uint8 value whose attribute message named
// name is exactly size bytes long.
func bigAttributeValue(t *testing.T, name string, size int) []uint8 {
	t.Helper()
	sb := &core.Superblock{Version: 2, OffsetSize: 8, LengthSize: 8}
	for n := size; n > 0; n-- {
		value := make([]uint8, n)
		dt, ds, err := inferDatatypeFromValue(value)
		require.NoError(t, err)
		msg, err := core.EncodeAttributeFromStruct(&core.Attribute{Name: name, Datatype: dt, Dataspace: ds, Data: value}, sb)
		require.NoError(t, err)
		if len(msg) == size {
			return value
		}
		if len(msg) < size {
			break
		}
	}
	t.Fatalf("no attribute message of %d bytes", size)
	return nil
}

// TestOpenDatasetAttributesAfterDenseTransition writes attributes to a
// dataset opened with OpenDataset after an attribute too large for the
// object header moved its attributes to dense storage in the same session.
func TestOpenDatasetAttributesAfterDenseTransition(t *testing.T) {
	path := filepath.Join(t.TempDir(), "transition.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/d", Float64, []uint64{2}, WithAttribute("a", "first"))
	require.NoError(t, err)
	require.NoError(t, ds.Write([]float64{1, 2}))
	require.NoError(t, fw.Close())

	fw, err = OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	ds, err = fw.OpenDataset("/d")
	require.NoError(t, err)
	// A message of 64 KiB does not fit an object header but fits the heap.
	big := bigAttributeValue(t, "big", 65536)
	require.NoError(t, ds.WriteAttribute("big", big))
	require.NoError(t, ds.WriteAttribute("small", int32(7)))
	require.NoError(t, ds.WriteAttribute("a", "replaced"))
	require.NoError(t, ds.WriteAttribute("gone", int32(1)))
	require.NoError(t, ds.DeleteAttribute("gone"))
	require.NoError(t, fw.Close())

	values := attributeValues(t, path, "/d")
	require.Len(t, values, 3)
	require.Equal(t, "7", values["small"])
	require.Equal(t, "replaced", values["a"])
	require.Contains(t, values, "big")

	f, err := Open(path)
	require.NoError(t, err)
	addr := findDataset(f, "/d").Address()
	oh, err := core.ReadObjectHeader(f.Reader(), addr, f.Superblock())
	require.NoError(t, err)
	require.True(t, hasDenseAttributeStorage(oh, f.Superblock()), "attributes must be dense")
	require.NoError(t, f.Close())
	requireNoCompactAttributesBesideDense(t, path, addr)
}
