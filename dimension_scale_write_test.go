package hdf5

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// writeDimensionScaleFile writes a small netCDF-4 style file: two
// "dimension without variable" scales (M, N), a coordinate-like scale with
// data (T), and variables attached to them.
func writeDimensionScaleFile(t *testing.T, path string, partial bool) {
	t.Helper()

	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)

	scale := func(name string, n uint64, dimid int32, ncName string) *DatasetWriter {
		ds, err := fw.CreateDataset("/"+name, Float32, []uint64{n})
		require.NoError(t, err)
		require.NoError(t, ds.Write(make([]float32, n)))
		require.NoError(t, ds.SetDimensionScale(ncName))
		require.NoError(t, ds.WriteAttribute("_Netcdf4Dimid", dimid))
		return ds
	}
	m := scale("M", 3, 0, fmt.Sprintf("This is a netCDF dimension but not a netCDF variable.%10d", 3))
	n := scale("N", 4, 1, fmt.Sprintf("This is a netCDF dimension but not a netCDF variable.%10d", 4))

	// Attach before the data is written: attachments are deferred to Close.
	data, err := fw.CreateDataset("/Data.IR", Float64, []uint64{3, 4})
	require.NoError(t, err)
	require.NoError(t, data.AttachDimensionScale(0, m))
	require.NoError(t, data.AttachDimensionScale(1, n))
	require.NoError(t, data.AttachDimensionScale(1, n)) // duplicate: no-op
	vals := make([]float64, 12)
	for i := range vals {
		vals[i] = float64(i)
	}
	require.NoError(t, data.Write(vals))

	// A variable with many attributes, so DIMENSION_LIST lands in dense storage.
	dense, err := fw.CreateDataset("/Dense", Float64, []uint64{4})
	require.NoError(t, err)
	require.NoError(t, dense.Write([]float64{1, 2, 3, 4}))
	for i := 0; i < 9; i++ {
		require.NoError(t, dense.WriteAttribute(fmt.Sprintf("a%d", i), int32(i)))
	}
	require.NoError(t, dense.AttachDimensionScale(0, n))

	// Only the first dimension has a scale; the second stays empty. netCDF-C
	// (4.9) cannot open such files, even when written by libhdf5, so this
	// variable is only checked with h5py.
	if partial {
		p, err := fw.CreateDataset("/Partial", Int32, []uint64{3, 2})
		require.NoError(t, err)
		require.NoError(t, p.Write([]int32{1, 2, 3, 4, 5, 6}))
		require.NoError(t, p.AttachDimensionScale(0, m))
		require.NoError(t, p.WriteAttribute("scale_ref", m.Reference()))
		require.NoError(t, p.WriteAttribute("scale_refs", []ObjectRef{n.Reference(), m.Reference()}))
	}

	require.NoError(t, fw.Close())
}

func TestAttachDimensionScaleErrors(t *testing.T) {
	fw, err := CreateForWrite(filepath.Join(t.TempDir(), "e.h5"), CreateTruncate)
	require.NoError(t, err)
	defer func() { require.NoError(t, fw.Close()) }()

	a, err := fw.CreateDataset("/a", Float32, []uint64{2})
	require.NoError(t, err)
	b, err := fw.CreateDataset("/b", Float32, []uint64{2, 2})
	require.NoError(t, err)

	require.Error(t, b.AttachDimensionScale(0, nil))
	require.Error(t, b.AttachDimensionScale(2, a))
	require.Error(t, b.AttachDimensionScale(-1, a))
	require.Error(t, a.AttachDimensionScale(0, a))
	require.NotZero(t, a.Address())
	require.Equal(t, ObjectRef(a.Address()), a.Reference())
	require.Error(t, a.WriteAttribute("x", []ObjectRef{}))
	require.Error(t, a.WriteAttribute("x", [][]ObjectRef{}))
	require.Error(t, a.WriteAttribute("x", []DimensionReference{}))
}

func TestAttachDimensionScaleReadBack(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dims.h5")
	writeDimensionScaleFile(t, path, true)

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	var names []string
	f.Walk(func(p string, obj Object) {
		if ds, ok := obj.(*Dataset); ok {
			names = append(names, p)
			attrs, err := ds.Attributes()
			require.NoError(t, err, p)
			require.NotEmpty(t, attrs, p)
		}
	})
	require.ElementsMatch(t, []string{"/M", "/N", "/Data.IR", "/Dense", "/Partial"}, names)

	// REFERENCE_LIST's "dimension" member is H5T_STD_U32LE, as libhdf5
	// writes it; the "dataset" reference must not swallow it.
	f.Walk(func(p string, obj Object) {
		ds, ok := obj.(*Dataset)
		if !ok || p != "/M" {
			return
		}
		attrs, err := ds.Attributes()
		require.NoError(t, err)
		for _, a := range attrs {
			if a.Name != referenceListAttr {
				continue
			}
			ct, err := core.ParseCompoundType(a.Datatype)
			require.NoError(t, err)
			require.Len(t, ct.Members, 2)
			require.Equal(t, "dimension", ct.Members[1].Name)
			require.Equal(t, uint32(0), ct.Members[1].Type.ClassBitField&0x08, "dimension must be unsigned")
			return
		}
		t.Fatalf("/M has no %s", referenceListAttr)
	})
}

func TestDeterministicDimensionScales(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.h5"), filepath.Join(dir, "b.h5")
	writeDimensionScaleFile(t, a, true)
	writeDimensionScaleFile(t, b, true)
	wantBytes, err := os.ReadFile(a)
	require.NoError(t, err)
	gotBytes, err := os.ReadFile(b)
	require.NoError(t, err)
	require.Equal(t, wantBytes, gotBytes)
}

const dimScaleCheckScript = `
import json, sys
import h5py
out = {"h5py": {}, "nc": {}}
with h5py.File(sys.argv[1], "r") as f:
    for name in ("Data.IR", "Dense", "Partial"):
        ds = f[name]
        out["h5py"][name] = [[s.name for s in d.values()] for d in ds.dims]
    p = f["Partial"]
    scalar = p.attrs["scale_ref"]
    shape = getattr(scalar, "shape", ())
    out["h5py"]["scalar_shape"] = list(shape) if shape != () else None
    refs = ([scalar] if shape == () else list(scalar)) + list(p.attrs["scale_refs"])
    out["h5py"]["refs"] = [f[r].name for r in refs]
    for name in ("M", "N"):
        out["h5py"][name + ".reflist"] = sorted(
            [f[r["dataset"]].name, int(r["dimension"])] for r in f[name].attrs["REFERENCE_LIST"])
try:
    import netCDF4
except ImportError:
    netCDF4 = None
if netCDF4 is not None:
    d = netCDF4.Dataset(sys.argv[2])
    out["nc"]["dims"] = {k: len(v) for k, v in d.dimensions.items()}
    out["nc"]["vars"] = {k: list(v.dimensions) for k, v in d.variables.items()}
    out["nc"]["sum"] = float(d.variables["Data.IR"][:].sum())
    d.close()
print(json.dumps(out))
`

// TestInteropDimensionScales checks with the HDF5 C library (h5py) and
// netCDF-C (netCDF4-python) that attached scales resolve and that netCDF
// sees named dimensions instead of phony_dim_*.
func TestInteropDimensionScales(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	if err := exec.Command(python, "-c", "import h5py").Run(); err != nil {
		t.Skip("h5py not available")
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "dims.h5")
	writeDimensionScaleFile(t, path, true)
	ncPath := filepath.Join(dir, "dims_nc.h5")
	writeDimensionScaleFile(t, ncPath, false)

	out, err := exec.Command(python, "-c", dimScaleCheckScript, path, ncPath).CombinedOutput()
	require.NoError(t, err, "python failed:\n%s", out)

	var got struct {
		H5py map[string]interface{} `json:"h5py"`
		NC   struct {
			Dims map[string]int      `json:"dims"`
			Vars map[string][]string `json:"vars"`
			Sum  float64             `json:"sum"`
		} `json:"nc"`
	}
	require.NoError(t, json.Unmarshal(out, &got), "output: %s", out)

	require.Equal(t, []interface{}{[]interface{}{"/M"}, []interface{}{"/N"}}, got.H5py["Data.IR"])
	require.Equal(t, []interface{}{[]interface{}{"/N"}}, got.H5py["Dense"])
	require.Equal(t, []interface{}{[]interface{}{"/M"}, []interface{}{}}, got.H5py["Partial"])
	require.Equal(t, []interface{}{"/M", "/N", "/M"}, got.H5py["refs"])
	require.Nil(t, got.H5py["scalar_shape"], "scalar ObjectRef must use a scalar dataspace")
	require.Equal(t, []interface{}{
		[]interface{}{"/Data.IR", float64(0)}, []interface{}{"/Partial", float64(0)},
	}, got.H5py["M.reflist"])
	require.Equal(t, []interface{}{
		[]interface{}{"/Data.IR", float64(1)}, []interface{}{"/Dense", float64(0)},
	}, got.H5py["N.reflist"])

	if got.NC.Dims == nil {
		t.Log("netCDF4 not available; skipping netCDF checks")
		return
	}
	require.Equal(t, 3, got.NC.Dims["M"])
	require.Equal(t, 4, got.NC.Dims["N"])
	require.Equal(t, []string{"M", "N"}, got.NC.Vars["Data.IR"])
	require.Equal(t, []string{"N"}, got.NC.Vars["Dense"])
	require.NotContains(t, got.NC.Vars, "Partial")
	require.NotContains(t, got.NC.Vars, "M")
	require.NotContains(t, got.NC.Vars, "N")
	require.InDelta(t, 66.0, got.NC.Sum, 1e-9)
}

const dimScaleExistingScript = `
import sys
import h5py, numpy as np
with h5py.File(sys.argv[1], "w", libver="latest") as f:
    x = f.create_dataset("X", data=np.zeros(3, "f4")); x.make_scale("X")
    y = f.create_dataset("Y", data=np.zeros(4, "f4")); y.make_scale("Y")
    f.create_dataset("Z", data=np.zeros(4, "f4")).make_scale("Z")
    a = f.create_dataset("A", data=np.zeros((3, 4)))
    a.dims[0].attach_scale(x)
    b = f.create_dataset("B", data=np.zeros(3))
    b.dims[0].attach_scale(x)
`

const dimScaleListScript = `
import json, sys
import h5py
out = {}
with h5py.File(sys.argv[1], "r") as f:
    for name in ("A", "B", "C"):
        if name in f:
            out[name] = [sorted(s.name for s in d.values()) for d in f[name].dims]
    for name in ("X", "Y", "Z"):
        if "REFERENCE_LIST" in f[name].attrs:
            out[name + ".reflist"] = sorted(
                [f[r["dataset"]].name, int(r["dimension"])] for r in f[name].attrs["REFERENCE_LIST"])
print(json.dumps(out))
`

// TestAttachDimensionScaleKeepsExisting reopens files that already have
// dimension scale attachments (written by libhdf5 and by this library),
// attaches more scales and checks that no prior attachment is lost.
func TestAttachDimensionScaleKeepsExisting(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	if err := exec.Command(python, "-c", "import h5py").Run(); err != nil {
		t.Skip("h5py not available")
	}

	writeGo := func(t *testing.T, path string) {
		t.Helper()
		fw, err := CreateForWrite(path, CreateTruncate)
		require.NoError(t, err)
		mk := func(name string, dims []uint64) *DatasetWriter {
			ds, err := fw.CreateDataset("/"+name, Float32, dims)
			require.NoError(t, err)
			n := uint64(1)
			for _, d := range dims {
				n *= d
			}
			require.NoError(t, ds.Write(make([]float32, n)))
			return ds
		}
		x, y := mk("X", []uint64{3}), mk("Y", []uint64{4})
		require.NoError(t, x.SetDimensionScale("X"))
		require.NoError(t, y.SetDimensionScale("Y"))
		z := mk("Z", []uint64{4})
		require.NoError(t, z.SetDimensionScale("Z"))
		a, b := mk("A", []uint64{3, 4}), mk("B", []uint64{3})
		require.NoError(t, a.AttachDimensionScale(0, x))
		require.NoError(t, b.AttachDimensionScale(0, x))
		require.NoError(t, fw.Close())
	}
	writeLib := func(t *testing.T, path string) {
		t.Helper()
		out, err := exec.Command(python, "-c", dimScaleExistingScript, path).CombinedOutput()
		require.NoError(t, err, "%s", out)
	}

	for name, write := range map[string]func(*testing.T, string){"go-hdf5": writeGo, "libhdf5": writeLib} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "existing.h5")
			write(t, path)

			fw, err := OpenForWrite(path, OpenReadWrite)
			require.NoError(t, err)
			a, err := fw.OpenDataset("/A")
			require.NoError(t, err)
			x, err := fw.OpenDataset("/X")
			require.NoError(t, err)
			y, err := fw.OpenDataset("/Y")
			require.NoError(t, err)
			z, err := fw.OpenDataset("/Z")
			require.NoError(t, err)
			require.NoError(t, a.AttachDimensionScale(1, y))
			require.NoError(t, a.AttachDimensionScale(1, z)) // second scale on the same dim
			require.NoError(t, a.AttachDimensionScale(0, x)) // already attached: no-op
			require.NoError(t, fw.Close())

			out, err := exec.Command(python, "-c", dimScaleListScript, path).CombinedOutput()
			require.NoError(t, err, "%s", out)
			var got map[string]interface{}
			require.NoError(t, json.Unmarshal(out, &got), "%s", out)

			require.Equal(t, []interface{}{[]interface{}{"/X"}, []interface{}{"/Y", "/Z"}}, got["A"])
			require.Equal(t, []interface{}{[]interface{}{"/X"}}, got["B"])
			require.Equal(t, []interface{}{
				[]interface{}{"/A", float64(0)}, []interface{}{"/B", float64(0)},
			}, got["X.reflist"])
			require.Equal(t, []interface{}{[]interface{}{"/A", float64(1)}}, got["Y.reflist"])
			require.Equal(t, []interface{}{[]interface{}{"/A", float64(1)}}, got["Z.reflist"])
		})
	}
}
