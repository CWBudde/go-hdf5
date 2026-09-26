package hdf5

import (
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// datasetAt returns the dataset at path, failing the test if absent.
func datasetAt(t *testing.T, f *File, path string) *Dataset {
	t.Helper()
	var found *Dataset
	f.Walk(func(p string, obj Object) {
		if ds, ok := obj.(*Dataset); ok && p == path {
			found = ds
		}
	})
	require.NotNil(t, found, "dataset %s not found", path)
	return found
}

func scalePaths(t *testing.T, scales []*Dataset) []string {
	t.Helper()
	out := make([]string, len(scales))
	for i, s := range scales {
		out[i] = s.Path()
	}
	return out
}

func TestDatasetShape_H5py(t *testing.T) {
	f, err := Open("testdata/dimscales/h5py_dimscales.h5")
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	tests := []struct {
		path     string
		shape    []uint64
		maxShape []uint64
		n        uint64
		class    core.DatatypeClass
		size     uint32
	}{
		{"/data", []uint64{3, 4}, []uint64{Unlimited, 4}, 12, core.DatatypeFloat, 8},
		{"/cube", []uint64{2, 3, 5}, []uint64{2, 3, 5}, 30, core.DatatypeFixed, 2},
		{"/grp/v", []uint64{3}, []uint64{3}, 3, core.DatatypeFloat, 4},
		{"/scalar", []uint64{}, []uint64{}, 1, core.DatatypeFloat, 8},
		{"/empty", nil, nil, 0, core.DatatypeFloat, 8},
		{"/refcompound", []uint64{2}, []uint64{2}, 2, core.DatatypeCompound, 12},
	}
	for _, tc := range tests {
		t.Run(tc.path, func(t *testing.T) {
			ds := datasetAt(t, f, tc.path)
			shape, err := ds.Shape()
			require.NoError(t, err)
			require.Equal(t, tc.shape, shape)
			maxShape, err := ds.MaxShape()
			require.NoError(t, err)
			require.Equal(t, tc.maxShape, maxShape)
			n, err := ds.NumElements()
			require.NoError(t, err)
			require.Equal(t, tc.n, n)
			dt, err := ds.Datatype()
			require.NoError(t, err)
			require.Equal(t, tc.class, dt.Class)
			if tc.class != core.DatatypeCompound {
				require.Equal(t, tc.size, dt.Size)
			}
			require.Equal(t, tc.path, ds.Path())
		})
	}
}

func TestDimensionScales_H5py(t *testing.T) {
	f, err := Open("testdata/dimscales/h5py_dimscales.h5")
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	data := datasetAt(t, f, "/data")
	timeDS := datasetAt(t, f, "/time")
	freq := datasetAt(t, f, "/freq")
	alt := datasetAt(t, f, "/freq_alt")
	v := datasetAt(t, f, "/grp/v")

	require.False(t, data.IsDimensionScale())
	require.True(t, timeDS.IsDimensionScale())
	require.True(t, alt.IsDimensionScale())

	name, err := timeDS.DimensionScaleName()
	require.NoError(t, err)
	require.Equal(t, "time", name)
	name, err = freq.DimensionScaleName()
	require.NoError(t, err)
	require.Equal(t, "frequency", name)
	name, err = alt.DimensionScaleName()
	require.NoError(t, err)
	require.Empty(t, name)
	name, err = data.DimensionScaleName()
	require.NoError(t, err)
	require.Empty(t, name)

	// DIMENSION_LIST: raw references and resolved scales.
	lists, err := data.DimensionList()
	require.NoError(t, err)
	require.Equal(t, [][]ObjectRef{
		{timeDS.Reference()},
		{freq.Reference(), alt.Reference()},
	}, lists)

	scales, err := data.AttachedScales(0)
	require.NoError(t, err)
	require.Equal(t, []string{"/time"}, scalePaths(t, scales))
	scales, err = data.AttachedScales(1)
	require.NoError(t, err)
	require.Equal(t, []string{"/freq", "/freq_alt"}, scalePaths(t, scales))
	_, err = data.AttachedScales(2)
	require.Error(t, err)

	scales, err = v.AttachedScales(0)
	require.NoError(t, err)
	require.Equal(t, []string{"/time"}, scalePaths(t, scales))

	// A dataset without scales.
	lists, err = timeDS.DimensionList()
	require.NoError(t, err)
	require.Nil(t, lists)
	scales, err = timeDS.AttachedScales(0)
	require.NoError(t, err)
	require.Empty(t, scales)

	// REFERENCE_LIST back references.
	refs, err := timeDS.ReferenceList()
	require.NoError(t, err)
	require.ElementsMatch(t, []DimensionReference{
		{Dataset: data.Reference(), Index: 0},
		{Dataset: v.Reference(), Index: 0},
	}, refs)
	refs, err = alt.ReferenceList()
	require.NoError(t, err)
	require.Equal(t, []DimensionReference{{Dataset: data.Reference(), Index: 1}}, refs)
	refs, err = data.ReferenceList()
	require.NoError(t, err)
	require.Nil(t, refs)

	p, ok := f.ObjectPath(v.Reference())
	require.True(t, ok)
	require.Equal(t, "/grp/v", p)
	_, ok = f.ObjectPath(ObjectRef(1))
	require.False(t, ok)
	_, err = f.Dereference(ObjectRef(1))
	require.Error(t, err)
}

func TestReferenceAttributes_H5py(t *testing.T) {
	f, err := Open("testdata/dimscales/h5py_dimscales.h5")
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	data := datasetAt(t, f, "/data")
	timeDS := datasetAt(t, f, "/time")
	v := datasetAt(t, f, "/grp/v")

	ref, err := data.ReadAttribute("ref")
	require.NoError(t, err)
	require.Equal(t, timeDS.Reference(), ref)

	refs, err := data.ReadAttribute("refs")
	require.NoError(t, err)
	require.Equal(t, []ObjectRef{timeDS.Reference(), v.Reference()}, refs)

	dl, err := data.ReadAttribute("DIMENSION_LIST")
	require.NoError(t, err)
	require.IsType(t, [][]ObjectRef{}, dl)

	// Compound attribute with a reference member (generic path).
	rl, err := timeDS.ReadAttribute("REFERENCE_LIST")
	require.NoError(t, err)
	vals, ok := rl.([]core.CompoundValue)
	require.True(t, ok, "got %T", rl)
	require.Len(t, vals, 2)
	for _, cv := range vals {
		obj, err := f.Dereference(cv["dataset"].(ObjectRef))
		require.NoError(t, err)
		require.IsType(t, &Dataset{}, obj)
		require.EqualValues(t, 0, cv["dimension"])
	}

	// Compound dataset with a reference member.
	rc := datasetAt(t, f, "/refcompound")
	rows, err := rc.ReadCompound()
	require.NoError(t, err)
	require.Len(t, rows, 2)
	require.Equal(t, timeDS.Reference(), rows[0]["obj"])
	require.EqualValues(t, 7, rows[0]["n"])
	obj, err := f.Dereference(rows[1]["obj"].(ObjectRef))
	require.NoError(t, err)
	g, ok := obj.(*Group)
	require.True(t, ok, "got %T", obj)
	require.Equal(t, "grp", g.Name())
	p, ok := f.ObjectPath(rows[1]["obj"].(ObjectRef))
	require.True(t, ok)
	require.Equal(t, "/grp", p)
}

func TestDimensionScales_NetCDF4(t *testing.T) {
	f, err := Open("testdata/dimscales/netcdf4_dimscales.nc")
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	ir := datasetAt(t, f, "/Data.IR")
	shape, err := ir.Shape()
	require.NoError(t, err)
	require.Equal(t, []uint64{3, 2, 4}, shape)

	got := make([]string, 0, len(shape))
	for dim := range shape {
		scales, err := ir.AttachedScales(dim)
		require.NoError(t, err)
		require.Len(t, scales, 1)
		got = append(got, scales[0].Path())
	}
	require.Equal(t, []string{"/M", "/R", "/N"}, got)

	n := datasetAt(t, f, "/N")
	require.True(t, n.IsDimensionScale())
	refs, err := n.ReferenceList()
	require.NoError(t, err)
	require.Equal(t, []DimensionReference{{Dataset: ir.Reference(), Index: 2}}, refs)
	p, ok := f.ObjectPath(refs[0].Dataset)
	require.True(t, ok)
	require.Equal(t, "/Data.IR", p)
}

// TestDimensionScales_GoWritten reads back scales attached by this library.
func TestDimensionScales_GoWritten(t *testing.T) {
	path := t.TempDir() + "/ds.h5"
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	x, err := fw.CreateDataset("/x", Float64, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, x.Write([]float64{1, 2}))
	require.NoError(t, x.SetDimensionScale("x"))
	d, err := fw.CreateDataset("/d", Float64, []uint64{2, 2})
	require.NoError(t, err)
	require.NoError(t, d.Write([]float64{1, 2, 3, 4}))
	require.NoError(t, d.AttachDimensionScale(1, x))
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	ds := datasetAt(t, f, "/d")
	scales, err := ds.AttachedScales(1)
	require.NoError(t, err)
	require.Equal(t, []string{"/x"}, scalePaths(t, scales))
	scales, err = ds.AttachedScales(0)
	require.NoError(t, err)
	require.Empty(t, scales)
	xs := datasetAt(t, f, "/x")
	name, err := xs.DimensionScaleName()
	require.NoError(t, err)
	require.Equal(t, "x", name)
	refs, err := xs.ReferenceList()
	require.NoError(t, err)
	require.Equal(t, []DimensionReference{{Dataset: ds.Reference(), Index: 1}}, refs)
}
