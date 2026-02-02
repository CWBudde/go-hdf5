package utils

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestJenkinsChecksum_RealHDF5Superblock verifies the Jenkins lookup3 implementation
// against checksums stored in real HDF5 files produced by the official C library.
func TestJenkinsChecksum_RealHDF5Superblock(t *testing.T) {
	files := []string{
		"../../testdata/minimal.h5",
		"../../testdata/matrix_2x3.h5",
		"../../testdata/compound_test.h5",
		"../../testdata/gzip_test.h5",
	}

	tested := 0
	for _, fname := range files {
		f, err := os.Open(fname)
		if err != nil {
			t.Logf("skipping %s: %v", fname, err)
			continue
		}

		buf := make([]byte, 48)
		n, err := f.ReadAt(buf, 0)
		f.Close()
		if n < 48 {
			t.Logf("skipping %s: too short", fname)
			continue
		}

		// Check HDF5 signature
		if string(buf[0:8]) != "\x89HDF\r\n\x1a\n" {
			t.Logf("skipping %s: not HDF5", fname)
			continue
		}

		version := buf[8]
		if version < 2 {
			t.Logf("skipping %s: superblock v%d (no checksum)", fname, version)
			continue
		}

		storedChecksum := binary.LittleEndian.Uint32(buf[44:48])
		computed := JenkinsChecksum(buf[0:44])

		t.Logf("%s: v%d stored=0x%08X computed=0x%08X", fname, version, storedChecksum, computed)
		assert.Equal(t, storedChecksum, computed, "checksum mismatch for %s", fname)
		tested++
	}

	require.Greater(t, tested, 0, "no v2+ superblock files found to test against")
}
