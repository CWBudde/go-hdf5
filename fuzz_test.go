package hdf5

import (
	"os"
	"path/filepath"
	"runtime/debug"
	"testing"
	"time"

	"github.com/cwbudde/go-hdf5/internal/core"
)

// fuzzSeedFiles are small, varied HDF5 files used to seed FuzzOpen.
var fuzzSeedFiles = []string{
	"testdata/minimal.h5",
	"testdata/simple.h5",
	"testdata/simple_contiguous.h5",
	"testdata/simple_float64.h5",
	"testdata/matrix_2x3.h5",
	"testdata/multiple_datasets.h5",
	"testdata/compound_test.h5",
	"testdata/string_test.h5",
	"testdata/vlen_strings.h5",
	"testdata/gzip_test.h5",
	"testdata/test_3d_chunked.h5",
	"testdata/with_attributes.h5",
	"testdata/with_groups.h5",
	"testdata/v0.h5",
	"testdata/v2.h5",
	"testdata/v3.h5",
	"testdata/various_types.h5",
	"testdata/test_attributes.h5",
}

// exerciseFile opens path and touches every object, dataset and attribute
// reachable from the root group. Errors are ignored: the goal is that
// malformed input never panics or exhausts memory.
func exerciseFile(path string) {
	f, err := Open(path)
	if err != nil {
		return
	}
	defer func() { _ = f.Close() }()

	f.Walk(func(_ string, obj Object) {
		switch o := obj.(type) {
		case *Group:
			readAllAttributes(o.Attributes())
		case *Dataset:
			exerciseDataset(o)
		}
	})
}

func readAllAttributes(attrs []*core.Attribute, err error) {
	if err != nil {
		return
	}
	for _, a := range attrs {
		if a != nil {
			_, _ = a.ReadValue()
		}
	}
}

func exerciseDataset(d *Dataset) {
	_, _ = d.Info()
	_, _ = d.Read()
	_, _ = d.ReadStrings()
	_, _ = d.ReadCompound()
	readAllAttributes(d.Attributes())
	if names, err := d.ListAttributes(); err == nil {
		for _, n := range names {
			_, _ = d.ReadAttribute(n)
		}
	}
	_, _ = d.ReadSlice([]uint64{0}, []uint64{1})
	if it, err := d.ChunkIterator(); err == nil {
		for i := 0; i < 64 && it.Next(); i++ {
			_, _ = it.Chunk()
		}
	}
}

func FuzzOpen(f *testing.F) {
	for _, p := range fuzzSeedFiles {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		f.Add(b)
	}
	extra, _ := filepath.Glob("testdata/c-library-corpus/*/*.h5")
	crashers, _ := filepath.Glob("testdata/fuzz/crashers/*.h5")
	for _, p := range append(extra, crashers...) {
		if b, err := os.ReadFile(p); err == nil && len(b) <= 64*1024 {
			f.Add(b)
		}
	}

	var watchdog time.Duration
	if v := os.Getenv("HDF5_FUZZ_WATCHDOG"); v != "" {
		watchdog, _ = time.ParseDuration(v)
		debug.SetTraceback("all")
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if watchdog > 0 {
			// Turn hangs into crashes so the fuzzer records the input.
			timer := time.AfterFunc(watchdog, func() { panic("fuzz input exceeded watchdog timeout") })
			defer timer.Stop()
		}
		p := filepath.Join(t.TempDir(), "fuzz.h5")
		if err := os.WriteFile(p, data, 0o600); err != nil {
			t.Fatal(err)
		}
		exerciseFile(p)
	})
}

// TestFuzzCrashers replays inputs that previously caused out-of-memory
// crashes (huge allocations driven by corrupted on-disk sizes).
func TestFuzzCrashers(t *testing.T) {
	files, err := filepath.Glob("testdata/fuzz/crashers/*.h5")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Skip("no crasher files")
	}
	for _, p := range files {
		t.Run(filepath.Base(p), func(_ *testing.T) {
			exerciseFile(p)
		})
	}
}
