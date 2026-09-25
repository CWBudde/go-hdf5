package hdf5

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestDenseAttributeNamesH5Dump writes enough root attributes for dense
// storage, with names of every length class the name hash distinguishes
// (including multiples of 12, which the B-tree v2 name index used to hash
// wrongly), and checks that h5dump (libhdf5) can open every one of them.
func TestDenseAttributeNamesH5Dump(t *testing.T) {
	h5dump := findH5Dump()
	if h5dump == "" {
		t.Skip("h5dump not available")
	}
	names := []string{
		"A", "Title", "Conventions", // < 12 bytes
		"DateModified", "Organization", // 12 bytes
		"DateCreated1", "ApplicationName", // 12, 15
		"abcdefghijklmnopqrstuvwx", // 24
		"SOFAConventionsVersion",   // 22
	}
	path := filepath.Join(t.TempDir(), "dense_names.h5")
	opts := make([]interface{}, 0, len(names))
	for _, n := range names {
		opts = append(opts, WithRootAttribute(n, "value of "+n))
	}
	fw, err := CreateForWrite(path, CreateTruncate, opts...)
	if err != nil {
		t.Fatalf("CreateForWrite: %v", err)
	}
	if err := fw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	out, err := runH5Dump(h5dump, "-A", path)
	if err != nil || strings.Contains(out, "error") {
		t.Fatalf("h5dump -A failed (%v):\n%s", err, out)
	}
	for _, n := range names {
		if !strings.Contains(out, `"value of `+n+`"`) {
			t.Errorf("h5dump output lacks the value of attribute %q", n)
		}
	}
}
