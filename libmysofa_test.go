package hdf5

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestLibmysofaLoad loads written SOFA-shaped files with libmysofa, whose
// HDF5 reader only understands netCDF-C's layout (new-style groups, a
// global heap below 64 KiB, few continuation chunks, compact variable
// attributes). $LIBMYSOFA_LOAD names the harness built by
// scripts/libmysofa/build.sh; the test is skipped without it.
func TestLibmysofaLoad(t *testing.T) {
	harness := os.Getenv("LIBMYSOFA_LOAD")
	if harness == "" {
		t.Skip("LIBMYSOFA_LOAD not set (see scripts/libmysofa/build.sh)")
	}

	for _, tc := range []struct {
		name  string
		shape sofaShape
	}{
		{"minimal", sofaShape{M: 2, N: 4}},
		{"large", largeSOFAShape},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), tc.name+".sofa")
			writeSOFA(t, path, tc.shape)
			s := readRootLinkStorage(t, path)
			require.True(t, s.dense, "the SOFA file should exercise dense link storage")

			out, err := exec.Command(harness, path).CombinedOutput()
			require.NoError(t, err, "libmysofa failed to load the file:\n%s", out)
			require.Equal(t, path+": check 0", strings.TrimSpace(string(out)))
		})
	}
}
