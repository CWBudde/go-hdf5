package hdf5

import (
	"fmt"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/cwbudde/go-hdf5/internal/core"
)

// Fixtures from testdata/dense/generate.py (HDF5 C library via h5py and
// netCDF-C).
const denseLongSuffix = "_a_rather_long_link_name_to_fill_the_fractal_heap_quickly"

func childNames(g *Group) []string {
	names := make([]string, 0, len(g.Children()))
	for _, c := range g.Children() {
		names = append(names, c.Name())
	}
	sort.Strings(names)
	return names
}

func attrNames(t *testing.T, attrs []*core.Attribute) map[string]*core.Attribute {
	t.Helper()
	m := make(map[string]*core.Attribute, len(attrs))
	for _, a := range attrs {
		m[a.Name] = a
	}
	return m
}

// TestReadDenseLinksBlockEnd reads a netCDF-C file whose link heap has
// checksummed direct blocks and a link message ending exactly at the end of
// the first block (a link was silently lost when the checksum was taken for
// a block trailer).
func TestReadDenseLinksBlockEnd(t *testing.T) {
	f, err := Open("testdata/dense/netcdf4_block_end.nc")
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	names := childNames(f.Root())
	require.Len(t, names, 30)
	require.Contains(t, names, "ReceiverUp")
}

// TestReadDenseDepth1H5py reads link and attribute name indexes of depth >= 1
// written by the HDF5 C library.
func TestReadDenseDepth1H5py(t *testing.T) {
	f, err := Open("testdata/dense/h5py_many.h5")
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	names := childNames(f.Root())
	require.Len(t, names, 401)
	for i := 0; i < 400; i++ {
		require.Contains(t, names, fmt.Sprintf("l%03d%s", i, denseLongSuffix))
	}

	attrs, err := f.Root().Attributes()
	require.NoError(t, err)
	byName := attrNames(t, attrs)
	require.Len(t, byName, 300)
	v, err := byName["attr299"].ReadValue()
	require.NoError(t, err)
	require.Equal(t, "value 299", v)

	var data *Dataset
	for _, c := range f.Root().Children() {
		if c.Name() == "data" {
			data = c.(*Dataset)
		}
	}
	require.NotNil(t, data)
	dattrs, err := data.Attributes()
	require.NoError(t, err)
	dByName := attrNames(t, dattrs)
	require.Len(t, dByName, 300)
	v, err = dByName["dattr123"].ReadValue()
	require.NoError(t, err)
	require.InDelta(t, 123.0, v, 0)
}

// TestReadDenseDepth1NetCDF reads a netCDF-C file with 150 variables, 300
// global attributes and a variable with 200 attributes.
func TestReadDenseDepth1NetCDF(t *testing.T) {
	f, err := Open("testdata/dense/netcdf4_many.nc")
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	names := childNames(f.Root())
	require.Len(t, names, 151) // variables + dimension "n"
	require.Contains(t, names, "var149"+denseLongSuffix)

	attrs, err := f.Root().Attributes()
	require.NoError(t, err)
	byName := attrNames(t, attrs)
	require.Len(t, byName, 301) // + _NCProperties
	v, err := byName["gattr150"].ReadValue()
	require.NoError(t, err)
	require.Equal(t, "global 150", v)

	for _, c := range f.Root().Children() {
		if c.Name() != "var000"+denseLongSuffix {
			continue
		}
		vattrs, err := c.(*Dataset).Attributes()
		require.NoError(t, err)
		vByName := attrNames(t, vattrs)
		require.Contains(t, vByName, "vattr199")
		require.GreaterOrEqual(t, len(vByName), 200)
		return
	}
	t.Fatal("variable var000 not found")
}
