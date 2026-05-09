package hdf5

import (
	"os"
	"path/filepath"
	"testing"
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

// TestCreateDataset_WithAttribute_TooMany ensures the compact-storage
// limit is enforced rather than silently producing an invalid file.
func TestCreateDataset_WithAttribute_TooMany(t *testing.T) {
	path := filepath.Join(t.TempDir(), "too_many.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	if err != nil {
		t.Fatalf("CreateForWrite: %v", err)
	}
	t.Cleanup(func() { _ = fw.Close() })

	opts := []DatasetOption{}
	for i := 0; i < MaxCompactDatasetAttributes+1; i++ {
		opts = append(opts, WithAttribute(stringDigit("attr_", i), int32(i)))
	}
	if _, err := fw.CreateDataset("/over", Int32, []uint64{1}, opts...); err == nil {
		t.Fatalf("CreateDataset accepted %d attributes; want error", MaxCompactDatasetAttributes+1)
	}
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
