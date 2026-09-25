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

// TestLegacyJenkinsHash pins legacyJenkinsHash to values computed with the
// jenkinsHash of go-hdf5 v0.16.0.
func TestLegacyJenkinsHash(t *testing.T) {
	for name, want := range map[string]uint32{
		"":                                     0x31b8a510,
		"A":                                    0x01014ba1,
		"Title":                                0x5a50032f,
		"DateModified":                         0x8cecf938,
		"Organization":                         0xdf13a3ea,
		"abcdefghijklmnopqrstuvwx":             0x197b8c64,
		"SOFAConventionsVersion":               0xd6ef483f,
		"abcdefghijklmnopqrstuvwxyzabcdefghij": 0xef45e1ae,
	} {
		if got := legacyJenkinsHash(name); got != want {
			t.Errorf("legacyJenkinsHash(%q) = %#08x, want %#08x", name, got, want)
		}
		if differs := len(name)%12 == 0; (legacyJenkinsHash(name) != jenkinsHash(name)) != differs {
			t.Errorf("%q (len %d): legacy and current hash differ = %v, want %v",
				name, len(name), !differs, differs)
		}
	}
}

// legacyTree returns a tree holding "Title" and "DateModified" as go-hdf5
// v0.16.0 wrote them: the 12-byte name under its legacy hash.
func legacyTree(t *testing.T) *WritableBTreeV2 {
	t.Helper()
	bt := NewWritableBTreeV2(4096)
	if err := bt.InsertRecord("Title", 1); err != nil {
		t.Fatalf("InsertRecord: %v", err)
	}
	if err := bt.InsertRecord("DateModified", 2); err != nil {
		t.Fatalf("InsertRecord: %v", err)
	}
	for i := range bt.records {
		if bt.records[i].NameHash != jenkinsHash("DateModified") {
			continue
		}
		rec := bt.records[i]
		rec.NameHash = legacyJenkinsHash("DateModified")
		bt.records = insertRecordSorted(append(bt.records[:i], bt.records[i+1:]...), rec)
		bt.leaf.Records = bt.records
		break
	}
	return bt
}

// TestLegacyHashRecordsAreFoundAndMigrated checks that a record written with
// the legacy hash is still found by search, update and delete, and that the
// tree then stores the corrected hash, sorted, without a duplicate.
func TestLegacyHashRecordsAreFoundAndMigrated(t *testing.T) {
	assertMigrated := func(t *testing.T, bt *WritableBTreeV2, wantRecords int) {
		t.Helper()
		if len(bt.records) != wantRecords {
			t.Fatalf("tree has %d records, want %d", len(bt.records), wantRecords)
		}
		for i, r := range bt.records {
			if r.NameHash == legacyJenkinsHash("DateModified") {
				t.Errorf("record %d still carries the legacy hash", i)
			}
			if i > 0 && bt.records[i-1].NameHash > r.NameHash {
				t.Errorf("records not sorted by hash at %d", i)
			}
		}
	}

	t.Run("search", func(t *testing.T) {
		bt := legacyTree(t)
		heapID, ok := bt.SearchRecord("DateModified")
		if !ok || heapID[0] != 2 {
			t.Fatalf("SearchRecord = %v, %v; want heap ID 2, true", heapID, ok)
		}
		assertMigrated(t, bt, 2)
		if !bt.HasKey("DateModified") || !bt.HasKey("Title") {
			t.Error("HasKey lost a record after migration")
		}
	})
	t.Run("update", func(t *testing.T) {
		bt := legacyTree(t)
		if err := bt.UpdateRecord("DateModified", 7); err != nil {
			t.Fatalf("UpdateRecord: %v", err)
		}
		assertMigrated(t, bt, 2)
		if heapID, _ := bt.SearchRecord("DateModified"); heapID[0] != 7 {
			t.Errorf("heap ID after update = %d, want 7", heapID[0])
		}
	})
	t.Run("delete", func(t *testing.T) {
		bt := legacyTree(t)
		if err := bt.DeleteRecord("DateModified"); err != nil {
			t.Fatalf("DeleteRecord: %v", err)
		}
		assertMigrated(t, bt, 1)
		if bt.HasKey("DateModified") {
			t.Error("DateModified still present after delete")
		}
	})
	t.Run("absent", func(t *testing.T) {
		bt := legacyTree(t)
		if bt.HasKey("Organization") {
			t.Error("HasKey(Organization) = true for a name never inserted")
		}
	})
}
