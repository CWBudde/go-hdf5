package hdf5

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// writeTextAttributeFile writes string attributes the way SOFA/netCDF
// writers do: root attributes (compact and, with dense=true, dense), group
// and dataset attributes.
func writeTextAttributeFile(t *testing.T, path string, dense bool) {
	t.Helper()
	opts := []interface{}{
		WithRootAttribute("Conventions", "SOFA"),
		WithRootAttribute("Title", "héllo wörld"), // UTF-8
		WithRootAttribute("Empty", ""),
	}
	if dense {
		for _, n := range []string{"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"} {
			opts = append(opts, WithRootAttribute(n, "value "+n))
		}
	}
	fw, err := CreateForWrite(path, CreateTruncate, opts...)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/x", Float64, []uint64{2}, WithAttribute("units", "meter"))
	require.NoError(t, err)
	require.NoError(t, ds.Write([]float64{1, 2}))
	require.NoError(t, ds.WriteAttribute("long_name", "the x"))
	root, err := fw.RootGroup()
	require.NoError(t, err)
	require.NoError(t, root.WriteAttribute("Comment", "added later"))
	require.NoError(t, fw.Close())
}

// TestStringAttributesAreScalarText checks that a Go string attribute is a
// scalar fixed-length string (netCDF NC_CHAR text) and reads back unchanged.
func TestStringAttributesAreScalarText(t *testing.T) {
	for _, dense := range []bool{false, true} {
		path := filepath.Join(t.TempDir(), "text.h5")
		writeTextAttributeFile(t, path, dense)

		f, err := Open(path)
		require.NoError(t, err)
		attrs, err := f.Root().Attributes()
		require.NoError(t, err)
		byName := map[string]*core.Attribute{}
		for _, a := range attrs {
			byName[a.Name] = a
		}
		for name, want := range map[string]string{
			"Conventions": "SOFA", "Title": "héllo wörld", "Empty": "", "Comment": "added later",
		} {
			a := byName[name]
			require.NotNil(t, a, name)
			require.Equal(t, core.DataspaceScalar, a.Dataspace.Type, "%s must have a scalar dataspace", name)
			require.Equal(t, uint32(max(len(want), 1)), a.Datatype.Size, name)
			// ASCII character set even for "héllo wörld", as netCDF-C writes
			// text: libmysofa rejects UTF-8 attributes in dense storage.
			require.Zero(t, a.Datatype.ClassBitField&0xF0, "%s must use the ASCII character set", name)
			v, err := a.ReadValue()
			require.NoError(t, err)
			require.Equal(t, want, v, name)
		}
		require.NoError(t, f.Close())
	}
}

// TestReadOneElementStringAttribute keeps reading the form written before:
// a NUL-terminated string in a one-element simple dataspace.
func TestReadOneElementStringAttribute(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	root, err := fw.RootGroup()
	require.NoError(t, err)
	require.NoError(t, root.WriteAttribute("Old", &encodedAttributeValue{
		datatype:  &core.DatatypeMessage{Class: core.DatatypeString, Size: 4},
		dataspace: &core.DataspaceMessage{Dimensions: []uint64{1}},
		data:      []byte("abc\x00"),
	}))
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	v, err := f.Root().ReadAttribute("Old")
	require.NoError(t, err)
	require.Equal(t, "abc", v)
}

const netcdfTextAttrScript = `
import json, sys
import netCDF4
d = netCDF4.Dataset(sys.argv[1])
out = {"root": {k: [type(d.getncattr(k)).__name__, d.getncattr(k)] for k in d.ncattrs()}}
x = d.variables["x"]
out["x"] = {k: [type(x.getncattr(k)).__name__, str(x.getncattr(k))] for k in x.ncattrs()}
print(json.dumps(out))
`

// TestNetCDFSeesTextAttributes checks with netCDF-C that string attributes
// are NC_CHAR text, not NC_STRING arrays: ncdump -h prints no "string"
// prefix, netCDF4-python returns str.
func TestNetCDFSeesTextAttributes(t *testing.T) {
	for _, dense := range []bool{false, true} {
		path := filepath.Join(t.TempDir(), "text.nc")
		writeTextAttributeFile(t, path, dense)

		if ncdump, err := exec.LookPath("ncdump"); err == nil {
			out, err := exec.Command(ncdump, "-h", path).CombinedOutput()
			require.NoError(t, err, "ncdump failed:\n%s", out)
			var stringLines []string
			for _, line := range strings.Split(string(out), "\n") {
				if strings.Contains(line, "string ") {
					stringLines = append(stringLines, strings.TrimSpace(line))
				}
			}
			require.Empty(t, stringLines, "ncdump -h:\n%s", out)
			require.Contains(t, string(out), `:Conventions = "SOFA" ;`)
			require.Contains(t, string(out), `x:units = "meter" ;`)
		}

		python, err := exec.LookPath("python3")
		if err != nil || exec.Command(python, "-c", "import netCDF4").Run() != nil {
			continue
		}
		out, err := exec.Command(python, "-c", netcdfTextAttrScript, path).CombinedOutput()
		require.NoError(t, err, "netCDF4 failed:\n%s", out)
		s := string(out)
		require.Contains(t, s, `"Conventions": ["str", "SOFA"]`)
		require.Contains(t, s, `"Comment": ["str", "added later"]`)
		require.Contains(t, s, `"Title": ["str", "h\u00e9llo w\u00f6rld"]`)
		require.Contains(t, s, `"long_name": ["str", "the x"]`)
		if dense {
			require.Contains(t, s, `"a9": ["str", "value a9"]`)
		}
	}
}
