package hdf5

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

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
            entry["sum"] = float(np.sum(data))
        out["/" + name] = entry
    out["/"] = {"attrs": {k: conv(v) for k, v in f.attrs.items()}}
    f.visititems(visit)

try:
    import netCDF4
except ImportError:
    netCDF4 = None
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

print(json.dumps(out))
`

type h5pyEntry struct {
	Attrs map[string][]interface{} `json:"attrs"`
	Shape []int                    `json:"shape"`
	Sum   float64                  `json:"sum"`
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

			out, err := exec.Command(python, "-c", h5pyDumpScript, path).CombinedOutput()
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
			case "superblock_v0":
				require.Equal(t, []interface{}{"SOFA"}, got["/"].Attrs["Conventions"])
				require.InDelta(t, 4.5, got["/x09"].Sum, 1e-9)
			}
		})
	}
}
