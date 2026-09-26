package hdf5

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestLibmysofaLoad loads a written SOFA-shaped file with libmysofa, whose
// HDF5 reader only understands new-style groups. $LIBMYSOFA_LOAD names the
// harness built by scripts/libmysofa/build.sh; the test is skipped without it.
func TestLibmysofaLoad(t *testing.T) {
	harness := os.Getenv("LIBMYSOFA_LOAD")
	if harness == "" {
		t.Skip("LIBMYSOFA_LOAD not set (see scripts/libmysofa/build.sh)")
	}

	path := filepath.Join(t.TempDir(), "minimal.sofa")
	writeMinimalSOFA(t, path)
	s := readRootLinkStorage(t, path)
	require.True(t, s.dense, "the SOFA file should exercise dense link storage")

	out, err := exec.Command(harness, path).CombinedOutput()
	require.NoError(t, err, "libmysofa failed to load the file:\n%s", out)
	require.Equal(t, path+": check 0", strings.TrimSpace(string(out)))
}
