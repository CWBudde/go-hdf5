package hdf5

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestCreateDataset_WithAttribute_RoundTrip verifies that attributes
// supplied via WithAttribute() are written into a dataset's object
// header and can be read back through the standard reader.
func TestCreateDataset_WithAttribute_RoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "with_attr.h5")

	fw, err := CreateForWrite(path, CreateTruncate)
	if err != nil {
		t.Fatalf("CreateForWrite: %v", err)
	}

	// Contiguous dataset with two compact attributes.
	contig, err := fw.CreateDataset("/contig", Float64, []uint64{4},
		WithAttribute("CLASS", "DIMENSION_SCALE"),
		WithAttribute("NAME", "M"))
	if err != nil {
		t.Fatalf("CreateDataset(contig): %v", err)
	}
	if err := contig.Write([]float64{1, 2, 3, 4}); err != nil {
		t.Fatalf("Write contig: %v", err)
	}

	// Chunked dataset (parallel path) with one attribute.
	chunked, err := fw.CreateDataset("/chunked", Float64, []uint64{4},
		WithChunkDims([]uint64{2}),
		WithAttribute("CLASS", "DIMENSION_SCALE"))
	if err != nil {
		t.Fatalf("CreateDataset(chunked): %v", err)
	}
	if err := chunked.Write([]float64{10, 20, 30, 40}); err != nil {
		t.Fatalf("Write chunked: %v", err)
	}

	if err := fw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Read back.
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = f.Close() })

	for _, tc := range []struct {
		path     string
		expected map[string]string
	}{
		{
			path:     "contig",
			expected: map[string]string{"CLASS": "DIMENSION_SCALE", "NAME": "M"},
		},
		{
			path:     "chunked",
			expected: map[string]string{"CLASS": "DIMENSION_SCALE"},
		},
	} {
		t.Run(tc.path, func(t *testing.T) {
			ds, ok := childDataset(f.Root().Children(), tc.path)
			if !ok {
				t.Fatalf("dataset /%s not found", tc.path)
			}
			for name, want := range tc.expected {
				got, err := ds.ReadAttribute(name)
				if err != nil {
					t.Errorf("ReadAttribute(%q): %v", name, err)
					continue
				}
				gotStr, ok := got.(string)
				if !ok {
					t.Errorf("attribute %q: got %T, want string", name, got)
					continue
				}
				if gotStr != want {
					t.Errorf("attribute %q = %q, want %q", name, gotStr, want)
				}
			}
		})
	}

	if t.Failed() {
		st, _ := os.Stat(path)
		t.Logf("test artefact: %s (%d bytes)", path, st.Size())
	}
}

// TestCreateDataset_WithAttribute_Dense creates contiguous and chunked
// datasets with more than MaxCompactDatasetAttributes attributes (dense
// storage), attaches a dimension scale afterwards (DIMENSION_LIST joins the
// dense storage) and reads everything back, with h5py too when available.
func TestCreateDataset_WithAttribute_Dense(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dense.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)

	const n = MaxCompactDatasetAttributes + 4
	want := map[string]interface{}{}
	var opts []DatasetOption
	for i := 0; i < n; i++ {
		name := stringDigit("attr_", i)
		var v interface{} = int32(i)
		if i%3 == 0 {
			v = stringDigit("value_", i)
		}
		want[name] = v
		opts = append(opts, WithAttribute(name, v))
	}

	scale, err := fw.CreateDataset("/x", Float64, []uint64{3})
	require.NoError(t, err)
	require.NoError(t, scale.Write([]float64{1, 2, 3}))
	require.NoError(t, scale.SetDimensionScale("x"))
	for _, name := range []string{"contiguous", "chunked"} {
		dsOpts := opts
		if name == "chunked" {
			dsOpts = append(append([]DatasetOption(nil), opts...), WithChunkDims([]uint64{2}))
		}
		dw, err := fw.CreateDataset("/"+name, Float64, []uint64{3}, dsOpts...)
		require.NoError(t, err, name)
		require.NoError(t, dw.Write([]float64{4, 5, 6}), name)
		require.NoError(t, dw.AttachDimensionScale(0, scale), name)
	}
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer f.Close()
	for _, name := range []string{"contiguous", "chunked"} {
		ds, ok := findDatasetByName(f, name)
		require.True(t, ok, name)
		for attr, v := range want {
			got, err := ds.ReadAttribute(attr)
			require.NoError(t, err, "%s: %s", name, attr)
			switch v := v.(type) {
			case int32:
				require.EqualValues(t, v, got, "%s: %s", name, attr)
			default:
				require.Equal(t, v, got, "%s: %s", name, attr)
			}
		}
		scales, err := ds.AttachedScales(0)
		require.NoError(t, err, name)
		require.Len(t, scales, 1, name)
		data, err := ds.Read()
		require.NoError(t, err, name)
		require.Equal(t, []float64{4, 5, 6}, data, name)
	}

	python := requireH5py(t)
	const script = `
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    for name in ("contiguous", "chunked"):
        d = f[name]
        attrs = sorted(k for k in d.attrs if k.startswith("attr_"))
        assert len(attrs) == int(sys.argv[2]), (name, attrs)
        assert d.attrs["attr_1"] == 1, d.attrs["attr_1"]
        assert d.dims[0][0].name == "/x", d.dims[0].keys()
        assert list(d[:]) == [4, 5, 6]
print("ok")
`
	out, err := exec.Command(python, "-c", script, path, strconv.Itoa(n)).CombinedOutput()
	require.NoError(t, err, "%s", out)
}

func childDataset(children []Object, name string) (*Dataset, bool) {
	for _, c := range children {
		if c.Name() == name {
			ds, ok := c.(*Dataset)
			return ds, ok
		}
	}
	return nil, false
}

func stringDigit(prefix string, i int) string {
	const digits = "0123456789"
	if i < 10 {
		return prefix + string(digits[i])
	}
	return prefix + string(digits[i/10]) + string(digits[i%10])
}
