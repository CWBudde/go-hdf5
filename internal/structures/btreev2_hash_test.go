package structures

import (
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/utils"
)

// TestJenkinsHashMatchesLookup3 checks the B-tree v2 name hash against
// utils.JenkinsLookup3, the lookup3 port whose checksums libhdf5 verifies
// on every object header we write. Names whose length is a multiple of 12
// ("DateModified", "Organization") used to hash differently, so libhdf5
// could not find those attributes in dense storage.
func TestJenkinsHashMatchesLookup3(t *testing.T) {
	if got := jenkinsHash(""); got != 0xdeadbeef {
		t.Errorf("jenkinsHash(\"\") = %#08x, want lookup3's seed 0xdeadbeef", got)
	}
	for n := 1; n <= 48; n++ {
		name := strings.Repeat("abcdefghijklmnopqrstuvwxyz", 2)[:n]
		if got, want := jenkinsHash(name), utils.JenkinsLookup3([]byte(name), 0); got != want {
			t.Errorf("jenkinsHash(%q) (len %d) = %#08x, want %#08x", name, n, got, want)
		}
	}
	for _, name := range []string{"DateModified", "Organization", "DIMENSION_LIST", "_NCProperties"} {
		if got, want := jenkinsHash(name), utils.JenkinsLookup3([]byte(name), 0); got != want {
			t.Errorf("jenkinsHash(%q) = %#08x, want %#08x", name, got, want)
		}
	}
}

// TestJenkinsLookup3ReferenceVector pins the lookup3 reference value from
// Bob Jenkins' lookup3.c driver, so both hashes are tied to the original.
func TestJenkinsLookup3ReferenceVector(t *testing.T) {
	const s = "Four score and seven years ago"
	if got := utils.JenkinsLookup3([]byte(s), 0); got != 0x17770551 {
		t.Errorf("JenkinsLookup3(%q, 0) = %#08x, want 0x17770551", s, got)
	}
	if got := jenkinsHash(s); got != 0x17770551 {
		t.Errorf("jenkinsHash(%q) = %#08x, want 0x17770551", s, got)
	}
}
