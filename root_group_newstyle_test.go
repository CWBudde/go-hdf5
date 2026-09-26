package hdf5

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/structures"
	"github.com/stretchr/testify/require"
)

// rootLinkStorage summarizes how the root group of a file stores its links.
type rootLinkStorage struct {
	symbolTable  bool // old-style group (Symbol Table message)
	linkInfo     bool // new-style group (Link Info message)
	groupInfo    bool
	compactLinks int  // Link messages in the root object header
	dense        bool // Link Info references a fractal heap and name index
}

func readRootLinkStorage(t *testing.T, path string) rootLinkStorage {
	t.Helper()
	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	sb := f.Superblock()
	oh, err := core.ReadObjectHeader(f.Reader(), sb.RootGroup, sb)
	require.NoError(t, err)

	var s rootLinkStorage
	for _, m := range oh.Messages {
		switch m.Type {
		case core.MsgSymbolTable:
			s.symbolTable = true
		case core.MsgLinkInfo:
			s.linkInfo = true
			li, err := core.ParseLinkInfoMessage(m.Data, sb)
			require.NoError(t, err)
			s.dense = li.FractalHeapAddress != undefAddr && li.FractalHeapAddress != 0
			if s.dense {
				require.NotEqual(t, undefAddr, li.NameBTreeAddress, "dense links need a name index")
			}
		case core.MsgGroupInfo:
			s.groupInfo = true
		case core.MsgLinkMessage:
			s.compactLinks++
		}
	}
	return s
}

// rootLinkName is the name of the i-th dataset written by writeRootLinks.
func rootLinkName(i int) string { return fmt.Sprintf("d%02d", i) }

// writeRootLinks creates n one-element datasets /d<first>, ... holding i.
func writeRootLinks(t *testing.T, fw *FileWriter, first, n int) {
	t.Helper()
	for i := first; i < first+n; i++ {
		ds, err := fw.CreateDataset("/"+rootLinkName(i), Float64, []uint64{1})
		require.NoError(t, err)
		require.NoError(t, ds.Write([]float64{float64(i)}))
	}
}

// requireRootLinks checks that the root of the file at path holds exactly
// the datasets 0..n-1 written by writeRootLinks, plus extra names.
func requireRootLinks(t *testing.T, path string, n int, extra ...string) {
	t.Helper()
	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	want := append([]string{}, extra...)
	for i := 0; i < n; i++ {
		want = append(want, rootLinkName(i))
	}

	children := f.Root().Children()
	got := make([]string, 0, len(children))
	for _, c := range children {
		got = append(got, strings.TrimPrefix(c.Name(), "/"))
	}
	require.ElementsMatch(t, want, got)

	for i := 0; i < n; i++ {
		vals, err := datasetAt(t, f, "/"+rootLinkName(i)).Read()
		require.NoError(t, err)
		require.Equal(t, []float64{float64(i)}, vals)
	}
}

// TestRootGroupNewStyle checks that superblock v2 files get a new-style root
// group, like netCDF-C writes and libmysofa requires: Link Info + Group Info,
// compact Link messages up to 8 links and dense link storage above.
func TestRootGroupNewStyle(t *testing.T) {
	for _, n := range []int{0, 1, 8, 9, 30} {
		t.Run(fmt.Sprintf("links_%d", n), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "root.h5")
			fw, err := CreateForWrite(path, CreateTruncate)
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, n)
			require.NoError(t, fw.Close())

			s := readRootLinkStorage(t, path)
			require.False(t, s.symbolTable, "root must not be a symbol table group")
			require.True(t, s.linkInfo, "root must carry a Link Info message")
			require.True(t, s.groupInfo, "root must carry a Group Info message")
			if n <= 8 {
				require.Equal(t, n, s.compactLinks)
				require.False(t, s.dense)
			} else {
				require.Zero(t, s.compactLinks, "dense groups keep no Link messages in the header")
				require.True(t, s.dense)
			}
			requireRootLinks(t, path, n)
		})
	}
}

func TestRootGroupNewStyleWithRootAttributes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "root.h5")
	opts := make([]interface{}, 0, 12)
	for i := 0; i < 12; i++ {
		opts = append(opts, WithRootAttribute(fmt.Sprintf("attr%02d", i), fmt.Sprintf("value %d", i)))
	}
	fw, err := CreateForWrite(path, CreateTruncate, opts...)
	require.NoError(t, err)
	writeRootLinks(t, fw, 0, 30)
	root, err := fw.RootGroup()
	require.NoError(t, err)
	require.NoError(t, root.WriteAttribute("later", "added"))
	require.NoError(t, fw.Close())

	s := readRootLinkStorage(t, path)
	require.True(t, s.linkInfo)
	require.True(t, s.dense)
	requireRootLinks(t, path, 30)

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	attrs, err := f.Root().Attributes()
	require.NoError(t, err)
	require.Len(t, attrs, 13)
	v, err := f.Root().ReadAttribute("attr11")
	require.NoError(t, err)
	require.Equal(t, "value 11", v)
}

func TestRootGroupSuperblockV0KeepsSymbolTable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "v0.h5")
	fw, err := CreateForWrite(path, CreateTruncate, WithSuperblockVersion(SuperblockV0))
	require.NoError(t, err)
	writeRootLinks(t, fw, 0, 12)
	require.NoError(t, fw.Close())

	s := readRootLinkStorage(t, path)
	require.True(t, s.symbolTable)
	require.False(t, s.linkInfo)
	requireRootLinks(t, path, 12)
}

func TestRootGroupNewStyleDuplicateLink(t *testing.T) {
	for _, n := range []int{3, 12} {
		t.Run(fmt.Sprintf("links_%d", n), func(t *testing.T) {
			fw, err := CreateForWrite(filepath.Join(t.TempDir(), "dup.h5"), CreateTruncate)
			require.NoError(t, err)
			defer func() { _ = fw.Close() }()
			writeRootLinks(t, fw, 0, n)
			_, err = fw.CreateDataset("/"+rootLinkName(1), Float64, []uint64{1})
			require.ErrorContains(t, err, "already exists")
		})
	}
}

// TestRootGroupNewStyleReopen adds links to a new-style root after
// reopening the file, crossing the compact → dense threshold in that session.
func TestRootGroupNewStyleReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reopen.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	writeRootLinks(t, fw, 0, 5)
	require.NoError(t, fw.Close())

	fw, err = OpenForWrite(path, OpenReadWrite)
	require.NoError(t, err)
	writeRootLinks(t, fw, 5, 10)
	require.NoError(t, fw.Close())

	s := readRootLinkStorage(t, path)
	require.True(t, s.dense)
	requireRootLinks(t, path, 15)
}

func TestRootGroupNewStyleHardLink(t *testing.T) {
	for _, n := range []int{2, 12} {
		t.Run(fmt.Sprintf("links_%d", n), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "hard.h5")
			fw, err := CreateForWrite(path, CreateTruncate)
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, n)
			require.NoError(t, fw.CreateHardLink("/alias", "/"+rootLinkName(1)))
			require.NoError(t, fw.Close())

			requireRootLinks(t, path, n, "alias")
			f, err := Open(path)
			require.NoError(t, err)
			defer func() { _ = f.Close() }()
			vals, err := datasetAt(t, f, "/alias").Read()
			require.NoError(t, err)
			require.Equal(t, []float64{1}, vals)
		})
	}
}

// TestRootGroupNewStyleH5Dump reads compact and dense new-style roots with
// the HDF5 C library's h5dump. Skipped when h5dump is not installed.
func TestRootGroupNewStyleH5Dump(t *testing.T) {
	h5dump := findH5Dump()
	if h5dump == "" {
		t.Skip("h5dump not available")
	}
	for _, n := range []int{3, 30} {
		t.Run(fmt.Sprintf("links_%d", n), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "root.h5")
			fw, err := CreateForWrite(path, CreateTruncate, WithRootAttribute("Conventions", "SOFA"))
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, n)
			require.NoError(t, fw.Close())

			out, err := runH5Dump(h5dump, path)
			require.NoError(t, err, "h5dump failed:\n%s", out)
			require.Contains(t, out, `ATTRIBUTE "Conventions"`)
			for i := 0; i < n; i++ {
				require.Contains(t, out, fmt.Sprintf("DATASET %q", rootLinkName(i)))
				require.Contains(t, out, fmt.Sprintf("(0): %d\n", i))
			}
		})
	}
}

// readRootLinkOrders returns the creation order of every root link and the
// Link Info's maximum creation order (the next order to assign).
func readRootLinkOrders(t *testing.T, path string) (orders map[string]int64, maxOrder int64) {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	sb := f.Superblock()
	oh, err := core.ReadObjectHeader(f.Reader(), sb.RootGroup, sb)
	require.NoError(t, err)

	orders = map[string]int64{}
	add := func(msg []byte) {
		lm, err := structures.ParseLinkMessage(msg, sb)
		require.NoError(t, err)
		require.True(t, lm.CreationOrderValid, "link %q has no creation order", lm.Name)
		orders[lm.Name] = lm.CreationOrder
	}
	var li *core.LinkInfoMessage
	for _, m := range oh.Messages {
		switch m.Type {
		case core.MsgLinkInfo:
			li, err = core.ParseLinkInfoMessage(m.Data, sb)
			require.NoError(t, err)
		case core.MsgLinkMessage:
			add(m.Data)
		}
	}
	require.NotNil(t, li)
	require.True(t, li.HasCreationOrderTracking(), "root must track link creation order")
	if li.FractalHeapAddress != undefAddr {
		r := bytes.NewReader(data)
		fh, err := structures.OpenFractalHeap(r, li.FractalHeapAddress, sb.LengthSize, sb.OffsetSize, sb.Endianness)
		require.NoError(t, err)
		ids, err := structures.ReadBTreeV2LinkNameHeapIDs(r, li.NameBTreeAddress, sb)
		require.NoError(t, err)
		for _, id := range ids {
			obj, err := fh.ReadObjectSpecCompliant(id)
			require.NoError(t, err)
			add(obj)
		}
	}
	return orders, li.MaxCreationOrder
}

// TestRootGroupNewStyleCreationOrder checks that root links carry their
// creation order, like netCDF-C writes them (libmysofa relies on it for
// dense links), and that it continues after reopening the file.
func TestRootGroupNewStyleCreationOrder(t *testing.T) {
	for _, n := range []int{3, 8, 9, 30} {
		t.Run(fmt.Sprintf("links_%d", n), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "order.h5")
			fw, err := CreateForWrite(path, CreateTruncate)
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, n)
			require.NoError(t, fw.Close())

			orders, maxOrder := readRootLinkOrders(t, path)
			require.Len(t, orders, n)
			for i := 0; i < n; i++ {
				require.Equal(t, int64(i), orders[rootLinkName(i)], "creation order of %s", rootLinkName(i))
			}
			require.Equal(t, int64(n), maxOrder)
		})
	}

	t.Run("reopen", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "order.h5")
		fw, err := CreateForWrite(path, CreateTruncate)
		require.NoError(t, err)
		writeRootLinks(t, fw, 0, 5)
		require.NoError(t, fw.Close())
		fw, err = OpenForWrite(path, OpenReadWrite)
		require.NoError(t, err)
		writeRootLinks(t, fw, 5, 10)
		require.NoError(t, fw.Close())

		orders, maxOrder := readRootLinkOrders(t, path)
		for i := 0; i < 15; i++ {
			require.Equal(t, int64(i), orders[rootLinkName(i)])
		}
		require.Equal(t, int64(15), maxOrder)
	})
}
