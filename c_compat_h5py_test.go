package hdf5

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// h5pyDumpScript opens a file with h5py (the HDF5 C library), reads every
// dataset and attribute and prints a JSON summary. When netCDF4 is installed
// it also opens the file with netCDF-C and reads all variables.
const h5pyDumpScript = `
import json, sys
import numpy as np
import h5py

def conv(v):
    v = np.asarray(v)
    if v.dtype.kind in "SO":
        return [x.decode() if isinstance(x, bytes) else str(x) for x in v.ravel().tolist()]
    return v.ravel().tolist()

out = {}
with h5py.File(sys.argv[1], "r") as f:
    def visit(name, obj):
        entry = {"attrs": {k: conv(v) for k, v in obj.attrs.items()}}
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            entry["shape"] = list(data.shape)
            if data.dtype.names:  # compound: flatten fields as "a.b"
                fields = {}
                def flatten(prefix, arr):
                    for n in arr.dtype.names:
                        sub = arr[n]
                        if sub.dtype.names:
                            flatten(prefix + n + ".", sub)
                        else:
                            fields[prefix + n] = conv(sub)
                flatten("", data)
                entry["fields"] = fields
            elif data.dtype.kind == "O":  # variable-length sequences
                entry["sum"] = float(sum(np.sum(np.asarray(x, dtype=np.float64)) for x in data.ravel()))
            else:
                entry["sum"] = float(np.sum(data))
        out["/" + name] = entry
    out["/"] = {"attrs": {k: conv(v) for k, v in f.attrs.items()}}
    f.visititems(visit)

try:
    import netCDF4
except ImportError:
    netCDF4 = None
if len(sys.argv) > 2 and sys.argv[2] == "no-netcdf":
    netCDF4 = None  # e.g. LZF is an h5py-only filter
if netCDF4 is not None:
    def walk(g):
        for v in g.variables.values():
            v[:]
            v.ncattrs()
        for sg in g.groups.values():
            walk(sg)
    d = netCDF4.Dataset(sys.argv[1])
    walk(d)
    d.ncattrs()
    d.close()
    out["_netcdf4"] = {"attrs": {}}

print(json.dumps(out, default=str))  # object references as text
`

type h5pyEntry struct {
	Attrs  map[string][]interface{} `json:"attrs"`
	Shape  []int                    `json:"shape"`
	Sum    float64                  `json:"sum"`
	Fields map[string][]interface{} `json:"fields"`
}

// TestInteropH5py opens files written by this library with the HDF5 C library
// through h5py (and netCDF-C through netCDF4-python when available).
// Skipped when python3 or h5py are not installed.
func TestInteropH5py(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	if err := exec.Command(python, "-c", "import h5py").Run(); err != nil {
		t.Skip("h5py not available")
	}

	for _, sc := range compatScenarios() {
		t.Run(sc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), sc.name+".h5")
			sc.write(t, path)

			args := []string{"-c", h5pyDumpScript, path}
			// LZF is an h5py-only filter; netCDF-C aborts on names longer
			// than NC_MAX_NAME (256), also in files written by libhdf5.
			if strings.HasPrefix(sc.name, "lzf") || sc.name == "root_links_long_names" {
				args = append(args, "no-netcdf")
			}
			out, err := exec.Command(python, args...).CombinedOutput()
			require.NoError(t, err, "h5py/netCDF4 failed to read %s:\n%s", sc.name, out)

			var got map[string]h5pyEntry
			require.NoError(t, json.Unmarshal(out, &got), "output: %s", out)

			switch sc.name {
			case "root_attrs_compact":
				require.Equal(t, []interface{}{"SOFA"}, got["/"].Attrs["Conventions"])
				require.Equal(t, []interface{}{float64(7)}, got["/"].Attrs["N"])
			case "root_attrs_dense":
				require.Len(t, got["/"].Attrs, 12)
				require.Equal(t, []interface{}{"value 11"}, got["/"].Attrs["attr11"])
			case "groups_and_datasets":
				require.InDelta(t, 8.0, got["/g1/sub/x"].Sum, 1e-9)
				require.InDelta(t, 12.5, got["/c1"].Sum, 1e-9)
				require.Equal(t, []interface{}{"meter"}, got["/c1"].Attrs["units"])
				require.Equal(t, []interface{}{float64(5)}, got["/c1"].Attrs["n"])
				require.Equal(t, []int{2, 3}, got["/c2"].Shape)
				require.InDelta(t, 18.0, got["/c2"].Sum, 1e-9)
				require.Equal(t, []int{2, 3, 4}, got["/c3"].Shape)
				require.InDelta(t, 288.0, got["/c3"].Sum, 1e-9)
				require.InDelta(t, 50.0, got["/k1"].Sum, 1e-9)
				require.Equal(t, []interface{}{"chunked"}, got["/k1"].Attrs["long_name"])
				require.Equal(t, []int{50, 40}, got["/k2"].Shape)
				require.InDelta(t, 2000000.0, got["/k2"].Sum, 1e-6)
				require.InDelta(t, 1800.0, got["/k3"].Sum, 1e-9)
				require.InDelta(t, 0.0, got["/unwritten"].Sum, 1e-9)
			case "attributes_after_creation":
				require.Equal(t, []interface{}{"Pa"}, got["/a"].Attrs["units"])
				require.Equal(t, []interface{}{2.5}, got["/a"].Attrs["scale"])
				require.InDelta(t, 8.0, got["/b"].Sum, 1e-9)
				require.Len(t, got["/b"].Attrs, 12)
				require.Equal(t, []interface{}{"hello"}, got["/grp"].Attrs["title"])
			case "large_initial_header":
				require.Equal(t, []interface{}{strings.Repeat("n", 200)}, got["/wide"].Attrs["NAME"])
				require.Equal(t, []interface{}{float64(4)}, got["/wide"].Attrs["_Netcdf4Dimid"])
				require.Equal(t, []interface{}{"added"}, got["/wide"].Attrs["later"])
				require.InDelta(t, 6.0, got["/wide"].Sum, 1e-9)
			case "chunked_large_header":
				require.InDelta(t, 50.0, got["/kc"].Sum, 1e-9)
				require.Len(t, got["/kc"].Attrs, 4)
			case "filter_orders":
				require.InDelta(t, 800.0, got["/fg"].Sum, 1e-9)
				require.InDelta(t, 800.0, got["/gf"].Sum, 1e-9)
			case "lzf_long_runs":
				require.InDelta(t, 5995.0, got["/lz"].Sum, 1e-9)
			case "compact_attrs_continuation":
				require.Len(t, got["/c"].Attrs, 8)
				require.Equal(t, []interface{}{strings.Repeat("c", 60) + "0"}, got["/c"].Attrs["k0"])
				require.Equal(t, []interface{}{float64(4)}, got["/c"].Attrs["k7"])
			case "dense_attrs_growing_heap":
				require.Len(t, got["/"].Attrs, 200)
				require.Len(t, got["/v"].Attrs, 200)
				require.Equal(t, []interface{}{strings.Repeat("x", 300) + "-199"}, got["/v"].Attrs["a199"])
				require.Equal(t, []interface{}{strings.Repeat("x", 300) + "-0"}, got["/"].Attrs["root000"])
			case "dense_group":
				for i := 0; i < 12; i++ {
					require.InDelta(t, 2.0, got[fmt.Sprintf("/dense/l%02d", i)].Sum, 1e-9)
				}
			case "many_links":
				for i := 0; i < 20; i++ {
					require.Contains(t, got, "/d"+string(rune('0'+i/10))+string(rune('0'+i%10)))
				}
			case "links_100", "links_1000":
				n := 100
				if sc.name == "links_1000" {
					n = 1000
				}
				_, nc := got["_netcdf4"]
				want := n + n/2 + 2 // datasets + "/" + "/sub"
				if nc {
					want++
				}
				require.Equal(t, want, len(got))
				for i := 0; i < n; i++ {
					require.InDelta(t, float64(i), got["/"+manyLinkName(i)].Sum, 0)
					if i%2 == 0 {
						require.InDelta(t, float64(i), got["/sub/"+manyLinkName(i)].Sum, 0)
					}
				}
			case "vlen_dataset":
				require.Equal(t, []int{3}, got["/v"].Shape)
				require.InDelta(t, 21.0, got["/v"].Sum, 1e-9)
			case "compound_dataset":
				requireCompoundInteropFields(t, got["/cmp"])
			case "root_links_compact", "root_links_dense", "root_links_dense_attrs", "root_links_reopen":
				requireRootLinkScenario(t, sc.name, got)
			case "superblock_v0":
				require.Equal(t, []interface{}{"SOFA"}, got["/"].Attrs["Conventions"])
				require.InDelta(t, 4.5, got["/x09"].Sum, 1e-9)
			}
		})
	}
}

// requireH5py returns the python3 executable, skipping the test when python3
// or h5py are not installed.
func requireH5py(t *testing.T) string {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	if err := exec.Command(python, "-c", "import h5py").Run(); err != nil {
		t.Skip("h5py not available")
	}
	return python
}

// TestInteropH5pyCompoundRead reads compound datasets written by h5py with
// the oldest (compound datatype version 1) and newest datatype encodings.
func TestInteropH5pyCompoundRead(t *testing.T) {
	python := requireH5py(t)
	const script = `
import sys
import numpy as np
import h5py
pt = np.dtype([("x", "<f4"), ("y", "<f4")])
dt = np.dtype([("id", "<i4"), ("value", "<f8"), ("small", "<i2"), ("name", "S6"), ("pt", pt)])
a = np.array([(1, 1.5, -3, b"alpha", (0.25, -0.5)), (2, -2.25, 300, b"beta", (1.5, 2.5)),
              (-7, 1e10, 32767, b"gamma!", (-8, 16))], dtype=dt)
with h5py.File(sys.argv[1], "w", libver=sys.argv[2]) as f:
    f["cmp"] = a
`
	for _, libver := range []string{"earliest", "latest"} {
		t.Run(libver, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "cmp.h5")
			out, err := exec.Command(python, "-c", script, path, libver).CombinedOutput()
			require.NoError(t, err, "%s", out)

			f, err := Open(path)
			require.NoError(t, err)
			defer f.Close()
			ds, ok := findDatasetByName(f, "cmp")
			require.True(t, ok)
			values, err := ds.ReadCompound()
			require.NoError(t, err)
			require.Len(t, values, len(compoundInteropRecords))
			for i, r := range compoundInteropRecords {
				require.Equal(t, r.id, values[i]["id"])
				require.Equal(t, r.value, values[i]["value"])
				require.Equal(t, r.small, values[i]["small"])
				require.Equal(t, r.name, values[i]["name"])
				pt, ok := values[i]["pt"].(core.CompoundValue)
				require.True(t, ok)
				require.Equal(t, r.x, pt["x"])
				require.Equal(t, r.y, pt["y"])
			}
		})
	}
}

// requireCompoundInteropFields checks the h5py view of the dataset written by
// writeCompoundInteropFile.
func requireCompoundInteropFields(t *testing.T, e h5pyEntry) {
	t.Helper()
	require.Equal(t, []int{len(compoundInteropRecords)}, e.Shape)
	require.Len(t, e.Fields, 6, "fields: %v", e.Fields)
	for i, r := range compoundInteropRecords {
		require.InDelta(t, float64(r.id), e.Fields["id"][i], 0, "id[%d]", i)
		require.InDelta(t, r.value, e.Fields["value"][i], 0, "value[%d]", i)
		require.InDelta(t, float64(r.small), e.Fields["small"][i], 0, "small[%d]", i)
		require.Equal(t, r.name, e.Fields["name"][i], "name[%d]", i)
		require.InDelta(t, float64(r.x), e.Fields["pt.x"][i], 0, "pt.x[%d]", i)
		require.InDelta(t, float64(r.y), e.Fields["pt.y"][i], 0, "pt.y[%d]", i)
	}
}

// requireRootLinkScenario checks what h5py read from the new-style root
// scenarios of compatScenarios.
func requireRootLinkScenario(t *testing.T, name string, got map[string]h5pyEntry) {
	t.Helper()
	switch name {
	case "root_links_compact":
		require.Equal(t, []interface{}{"SOFA"}, got["/"].Attrs["Conventions"])
		for i := 0; i < 3; i++ {
			require.InDelta(t, float64(i), got["/"+rootLinkName(i)].Sum, 0)
		}
	case "root_links_dense":
		require.Equal(t, []interface{}{"SimpleFreeFieldHRIR"}, got["/"].Attrs["SOFAConventions"])
		require.Equal(t, []int{2, 2, 4}, got["/Data.IR"].Shape)
		require.InDelta(t, 3.0, got["/Data.IR"].Sum, 1e-9)
		require.Equal(t, []int{2, 3, 1}, got["/ReceiverPosition"].Shape)
		require.Contains(t, got["/Data.IR"].Attrs, "DIMENSION_LIST")
		require.Contains(t, got["/M"].Attrs, "REFERENCE_LIST")
	case "root_links_dense_attrs":
		require.Len(t, got["/"].Attrs, 12)
		for i := 0; i < 30; i++ {
			require.InDelta(t, float64(i), got["/"+rootLinkName(i)].Sum, 0)
		}
	case "root_links_reopen":
		for i := 0; i < 15; i++ {
			require.InDelta(t, float64(i), got["/"+rootLinkName(i)].Sum, 0)
		}
	case "root_links_long_names":
		for i := 0; i < 10; i++ {
			require.InDelta(t, float64(i), got["/"+longRootLinkName(i)].Sum, 0)
		}
	}
}
