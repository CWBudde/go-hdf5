package hdf5

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/structures"
	"github.com/stretchr/testify/require"
)

// This file checks adding links to new-style roots whose dense link storage
// the writer cannot extend in place: roots written by the HDF5 C library
// (multi-level fractal heaps and name indexes, creation order indexes) and
// links whose Link message exceeds libhdf5's 4 KiB managed heap object size.

// longRootLinkName is a link name of about 5 KiB: its Link message is larger
// than the 4 KiB libhdf5 allows for managed objects in a link heap.
func longRootLinkName(i int) string {
	return fmt.Sprintf("%s_%02d", strings.Repeat("long", 1250), i)
}

// writeRootDataset creates the one-element dataset /name holding v.
func writeRootDataset(t *testing.T, fw *FileWriter, name string, v float64) {
	t.Helper()
	ds, err := fw.CreateDataset("/"+name, Float64, []uint64{1})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]float64{v}))
}

// requireRootDatasets checks that the root of the file at path holds the
// one-element datasets want (name → value), plus others names.
func requireRootDatasets(t *testing.T, path string, want map[string]float64, others ...string) {
	t.Helper()
	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	names := append([]string{}, others...)
	for name := range want {
		names = append(names, name)
	}
	children := f.Root().Children()
	got := make([]string, 0, len(children))
	for _, c := range children {
		got = append(got, strings.TrimPrefix(c.Name(), "/"))
	}
	require.ElementsMatch(t, names, got)

	for name, v := range want {
		vals, err := datasetAt(t, f, "/"+name).Read()
		require.NoError(t, err)
		require.Equal(t, []float64{v}, vals, "dataset %.20s...", name)
	}
}

// TestRootGroupNewStyleLongLinkNames adds links whose Link messages exceed
// libhdf5's 4 KiB managed link heap objects: when the compact root converts
// to dense storage, when they are added to dense storage and when the root
// direct block of the link heap must grow beyond 64 KiB.
func TestRootGroupNewStyleLongLinkNames(t *testing.T) {
	path := filepath.Join(t.TempDir(), "long.h5")
	want := map[string]float64{}
	add := func(fw *FileWriter, name string) {
		writeRootDataset(t, fw, name, float64(len(want)))
		want[name] = float64(len(want))
	}

	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	for i := 0; i < 8; i++ {
		add(fw, longRootLinkName(i))
	}
	add(fw, "short")                    // 9th link: moves the long names to dense storage
	add(fw, longRootLinkName(9))        // into dense storage
	add(fw, strings.Repeat("x", 65400)) // does not fit a 64 KiB direct block
	require.NoError(t, fw.Close())
	require.True(t, readRootLinkStorage(t, path).dense)
	requireRootDatasets(t, path, want)

	fw, err = OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	add(fw, longRootLinkName(11))
	_, err = fw.CreateDataset("/"+longRootLinkName(3), Float64, []uint64{1})
	require.ErrorContains(t, err, "already exists")
	require.NoError(t, fw.Close())
	requireRootDatasets(t, path, want)

	t.Run("h5py", func(t *testing.T) {
		res := runH5pyModify(t, path, "name", "short")
		require.Len(t, res.Before, len(want))
		require.ElementsMatch(t, append(res.Before, "by_h5py"), res.After)
		require.Equal(t, []float64{8}, res.Values["short"])
		requireRootDatasets(t, path, want, "by_h5py")
	})
}

// denseRootIndexes describes the dense link storage of a root group.
type denseRootIndexes struct {
	heapRows      uint16 // rows of the fractal heap's root indirect block (0: direct root)
	nameDepth     uint16 // depth of the name index
	corderIndexed bool
	corderDepth   uint16
}

func readDenseRootIndexes(t *testing.T, path string) denseRootIndexes {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	li, sb := readRootLinkInfo(t, path)
	require.True(t, li.HasFractalHeap(), "root must use dense link storage")
	r := bytes.NewReader(data)
	fh, err := structures.OpenFractalHeap(r, li.FractalHeapAddress, sb.LengthSize, sb.OffsetSize, sb.Endianness)
	require.NoError(t, err)
	names, err := core.ReadBTreeV2Info(r, li.NameBTreeAddress, sb)
	require.NoError(t, err)
	d := denseRootIndexes{
		heapRows:      fh.Header.CurrentRowCount,
		nameDepth:     names.Depth,
		corderIndexed: li.HasCreationOrderIndex(),
	}
	if d.corderIndexed {
		corder, err := core.ReadBTreeV2Info(r, li.CreationOrderBTreeAddress, sb)
		require.NoError(t, err)
		d.corderDepth = corder.Depth
	}
	return d
}

// readRootLinkInfo returns the root group's Link Info message.
func readRootLinkInfo(t *testing.T, path string) (*core.LinkInfoMessage, *core.Superblock) {
	t.Helper()
	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	sb := f.Superblock()
	oh, err := core.ReadObjectHeader(f.Reader(), sb.RootGroup, sb)
	require.NoError(t, err)
	for _, m := range oh.Messages {
		if m.Type == core.MsgLinkInfo {
			li, err := core.ParseLinkInfoMessage(m.Data, sb)
			require.NoError(t, err)
			return li, sb
		}
	}
	t.Fatal("root has no Link Info message")
	return nil, nil
}

// rootLinksByCreationOrderIndex returns the root's link names in the order
// of its creation order index (a type 6 v2 B-tree), checking that the index
// holds every link with the creation order its Link message stores.
func rootLinksByCreationOrderIndex(t *testing.T, path string) []string {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	li, sb := readRootLinkInfo(t, path)
	require.True(t, li.HasCreationOrderIndex(), "root must index link creation order")
	require.True(t, li.HasFractalHeap(), "root must use dense link storage")
	r := bytes.NewReader(data)

	info, records, err := core.ReadBTreeV2Records(r, li.CreationOrderBTreeAddress, sb)
	require.NoError(t, err)
	require.Equal(t, uint8(6), info.Type, "creation order index must be a type 6 B-tree")
	require.Equal(t, uint16(15), info.RecordSize, "type 6 record size")
	nameInfo, err := core.ReadBTreeV2Info(r, li.NameBTreeAddress, sb)
	require.NoError(t, err)
	require.Equal(t, nameInfo.TotalRecords, info.TotalRecords, "both indexes hold every link")
	require.Len(t, records, int(info.TotalRecords))

	fh, err := structures.OpenFractalHeap(r, li.FractalHeapAddress, sb.LengthSize, sb.OffsetSize, sb.Endianness)
	require.NoError(t, err)
	names := make([]string, 0, len(records))
	prev := int64(-1)
	for _, rec := range records {
		order := int64(binary.LittleEndian.Uint64(rec[:8]))
		require.Greater(t, order, prev, "creation order index must be sorted")
		prev = order
		obj, err := fh.ReadObjectSpecCompliant(rec[8:15])
		require.NoError(t, err)
		lm, err := structures.ParseLinkMessage(obj, sb)
		require.NoError(t, err)
		require.True(t, lm.CreationOrderValid, "link %q has no creation order", lm.Name)
		require.Equal(t, order, lm.CreationOrder, "creation order of %q", lm.Name)
		names = append(names, lm.Name)
	}
	require.Less(t, prev, li.MaxCreationOrder, "max creation order must exceed every link's")
	return names
}

// TestRootGroupNewStyleModifyLibhdf5DenseRoot adds links to the root of a
// file written by h5py whose link heap has an indirect root block and whose
// name index has depth 1, as libhdf5 writes larger groups.
func TestRootGroupNewStyleModifyLibhdf5DenseRoot(t *testing.T) {
	src, err := os.ReadFile("testdata/dense/h5py_many.h5")
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "many.h5")
	require.NoError(t, os.WriteFile(path, src, 0o600))

	before := readDenseRootIndexes(t, path)
	require.NotZero(t, before.heapRows, "fixture must have a multi-level link heap")
	require.NotZero(t, before.nameDepth, "fixture must have a multi-level name index")

	fw, err := OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	writeRootDataset(t, fw, "added", 42)
	writeRootDataset(t, fw, longRootLinkName(0), 43)
	_, err = fw.CreateDataset("/data", Float64, []uint64{1})
	require.ErrorContains(t, err, "already exists")
	require.NoError(t, fw.Close())

	// Adding more links in a new session extends the rebuilt storage.
	fw, err = OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	writeRootDataset(t, fw, "added2", 44)
	require.NoError(t, fw.Close())

	others := []string{"data"}
	for i := 0; i < 400; i++ {
		others = append(others, fmt.Sprintf("l%03d_a_rather_long_link_name_to_fill_the_fractal_heap_quickly", i))
	}
	want := map[string]float64{"added": 42, longRootLinkName(0): 43, "added2": 44}
	requireRootDatasets(t, path, want, others...)

	f, err := Open(path)
	require.NoError(t, err)
	attrs, err := f.Root().Attributes()
	require.NoError(t, err)
	require.Len(t, attrs, 300, "root attributes must be kept")
	vals, err := datasetAt(t, f, "/l399_a_rather_long_link_name_to_fill_the_fractal_heap_quickly").Read()
	require.NoError(t, err)
	require.Equal(t, []float64{0, 1, 2, 3}, vals)
	require.NoError(t, f.Close())

	t.Run("h5py", func(t *testing.T) {
		res := runH5pyModify(t, path, "name", "added", longRootLinkName(0), "added2")
		require.Len(t, res.Before, 404)
		require.Equal(t, []float64{42}, res.Values["added"])
		require.Equal(t, []float64{43}, res.Values[longRootLinkName(0)])
		require.Equal(t, []float64{44}, res.Values["added2"])
		require.ElementsMatch(t, append(res.Before, "by_h5py"), res.After)
		requireRootDatasets(t, path, want, append(others, "by_h5py")...)
	})
}

// TestRootGroupNewStyleCreationOrderIndex adds links to a root whose Link
// Info message indexes link creation order: the compact → dense conversion
// must build the creation order index and later links must be added to it.
func TestRootGroupNewStyleCreationOrderIndex(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corder.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	writeRootLinks(t, fw, 0, 3)
	require.NoError(t, fw.Close())

	// Index creation order, like H5Pset_link_creation_order(TRACKED|INDEXED).
	fw, err = OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	g, err := fw.readGroupLinks(fw.rootGroupAddr)
	require.NoError(t, err)
	g.linkInfo.Flags |= core.LinkInfoIndexCreationOrder
	g.linkInfo.CreationOrderBTreeAddress = undefAddr
	require.NoError(t, fw.writeGroupHeader(g))
	writeRootLinks(t, fw, 3, 9) // crosses max_compact
	require.NoError(t, fw.Close())

	want := make([]string, 0, 16)
	for i := 0; i < 12; i++ {
		want = append(want, rootLinkName(i))
	}
	require.Equal(t, want, rootLinksByCreationOrderIndex(t, path))

	fw, err = OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	writeRootLinks(t, fw, 12, 3)
	require.NoError(t, fw.Close())
	for i := 12; i < 15; i++ {
		want = append(want, rootLinkName(i))
	}
	require.Equal(t, want, rootLinksByCreationOrderIndex(t, path))
	requireRootLinks(t, path, 15)

	t.Run("h5py", func(t *testing.T) {
		res := runH5pyModify(t, path, "crt", rootLinkName(14))
		require.Equal(t, want, res.Before)
		require.Equal(t, append(want, "by_h5py"), res.After)
		require.Equal(t, []float64{14}, res.Values[rootLinkName(14)])
	})
}

// TestRootGroupNewStyleModifyLibhdf5CreationOrderIndex adds links to roots
// that h5py created with an indexed link creation order: compact, dense in a
// single direct block and leaf, and multi-level dense storage.
func TestRootGroupNewStyleModifyLibhdf5CreationOrderIndex(t *testing.T) {
	python := requireH5py(t)
	for _, tc := range []struct {
		name       string
		links      int
		multiLevel bool
	}{
		{"compact", 5, false},
		{"dense", 20, false},
		{"dense_multi_level", 400, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "corder.h5")
			out, err := exec.Command(python, "-c", h5pyCreationOrderScript, path, fmt.Sprint(tc.links)).CombinedOutput()
			require.NoError(t, err, "%s", out)
			if tc.links > 8 {
				d := readDenseRootIndexes(t, path)
				require.True(t, d.corderIndexed)
				require.Equal(t, tc.multiLevel, d.heapRows > 0 && d.nameDepth > 0 && d.corderDepth > 0,
					"fixture layout: %+v", d)
			}

			fw, err := OpenForWrite(path, OpenReadWrite)
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, 5)
			require.NoError(t, fw.Close())

			want := make([]string, 0, tc.links+6)
			for i := 0; i < tc.links; i++ {
				want = append(want, fmt.Sprintf("h%04d", i))
			}
			for i := 0; i < 5; i++ {
				want = append(want, rootLinkName(i))
			}
			require.Equal(t, want, rootLinksByCreationOrderIndex(t, path))

			res := runH5pyModify(t, path, "crt", rootLinkName(4), "h0000")
			require.Equal(t, want, res.Before)
			require.Equal(t, append(want, "by_h5py"), res.After)
			require.Equal(t, []float64{4}, res.Values[rootLinkName(4)])
			require.Equal(t, []float64{0}, res.Values["h0000"])
			require.Equal(t, append(want, "by_h5py"), rootLinksByCreationOrderIndex(t, path))
		})
	}
}

// h5pyCreationOrderScript writes a file (libver latest) whose root indexes
// link creation order but not attribute creation order, with argv[2]
// one-element datasets h0000, h0001, ...
const h5pyCreationOrderScript = `
import sys
import numpy as np
import h5py
fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
fcpl.set_link_creation_order(h5py.h5p.CRT_ORDER_TRACKED | h5py.h5p.CRT_ORDER_INDEXED)
fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
fid = h5py.h5f.create(sys.argv[1].encode(), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
with h5py.File(fid) as f:
    for i in range(int(sys.argv[2])):
        f.create_dataset("h%04d" % i, data=np.array([float(i)]))
`

// h5pyModifyScript opens a file read-write with h5py, lists the root links
// in native order of the name (argv[2] "name") or creation order ("crt")
// index, reads the datasets argv[3:], adds the dataset by_h5py and lists the
// links again.
const h5pyModifyScript = `
import json, sys
import h5py
idx = h5py.h5.INDEX_CRT_ORDER if sys.argv[2] == "crt" else h5py.h5.INDEX_NAME
def names(f):
    out = []
    f.id.links.iterate(lambda n: out.append(n.decode()), idx_type=idx, order=h5py.h5.ITER_NATIVE)
    return out
with h5py.File(sys.argv[1], "r+") as f:
    before = names(f)
    values = {k: f[k][()].tolist() for k in sys.argv[3:]}
    f["by_h5py"] = [7.0]
    after = names(f)
print(json.dumps({"before": before, "after": after, "values": values}))
`

type h5pyModifyResult struct {
	Before []string             `json:"before"`
	After  []string             `json:"after"`
	Values map[string][]float64 `json:"values"`
}

// runH5pyModify runs h5pyModifyScript on path (skipping the test without
// h5py): the HDF5 C library reads the root links through the given index
// and adds a link to the storage this library wrote.
func runH5pyModify(t *testing.T, path, index string, datasets ...string) h5pyModifyResult {
	t.Helper()
	python := requireH5py(t)
	args := append([]string{"-c", h5pyModifyScript, path, index}, datasets...)
	out, err := exec.Command(python, args...).CombinedOutput()
	require.NoError(t, err, "h5py failed:\n%s", out)
	var res h5pyModifyResult
	require.NoError(t, json.Unmarshal(out, &res), "output: %.2000s", out)
	return res
}
