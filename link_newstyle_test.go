package hdf5

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/structures"
	"github.com/stretchr/testify/require"
)

// rootLinkMessages returns the root group's Link messages by name, whether
// they are stored compactly in its object header or densely in its fractal
// heap.
func rootLinkMessages(t *testing.T, path string) map[string]*core.LinkMessage {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	sb := f.Superblock()
	oh, err := core.ReadObjectHeader(f.Reader(), sb.RootGroup, sb)
	require.NoError(t, err)

	links := make(map[string]*core.LinkMessage)
	add := func(msg []byte) {
		lm, err := core.ParseLinkMessage(msg, sb)
		require.NoError(t, err)
		require.NotContains(t, links, lm.Name, "duplicate link")
		links[lm.Name] = lm
	}
	for _, m := range oh.Messages {
		switch m.Type {
		case core.MsgLinkMessage:
			add(m.Data)
		case core.MsgLinkInfo:
			li, err := core.ParseLinkInfoMessage(m.Data, sb)
			require.NoError(t, err)
			if !li.HasFractalHeap() {
				continue
			}
			r := bytes.NewReader(data)
			fh, err := structures.OpenFractalHeap(r, li.FractalHeapAddress, sb.LengthSize, sb.OffsetSize, sb.Endianness)
			require.NoError(t, err)
			_, records, err := core.ReadBTreeV2Records(r, li.NameBTreeAddress, sb)
			require.NoError(t, err)
			for _, rec := range records {
				obj, err := fh.ReadObjectSpecCompliant(rec[4:11]) // hash (4) + heap ID (7)
				require.NoError(t, err)
				add(obj)
			}
		}
	}
	return links
}

// writeExternalTarget writes dir/other.h5 holding the dataset /x = [7, 8].
func writeExternalTarget(t *testing.T, dir string) {
	t.Helper()
	fw, err := CreateForWrite(filepath.Join(dir, "other.h5"), CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/x", Float64, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]float64{7, 8}))
	require.NoError(t, fw.Close())
}

// requireSoftAndExternalLinks checks the Link messages of the soft link
// /s → /d00 and the external link /e → other.h5:/x.
func requireSoftAndExternalLinks(t *testing.T, links map[string]*core.LinkMessage) {
	t.Helper()
	s, e := links["s"], links["e"]
	require.NotNil(t, s, "root has no link s")
	require.NotNil(t, e, "root has no link e")

	require.Equal(t, core.LinkTypeSoft, s.Type)
	target, err := s.GetSoftLinkPath()
	require.NoError(t, err)
	require.Equal(t, "/d00", target)

	require.Equal(t, core.LinkTypeExternal, e.Type)
	require.Equal(t, []byte("\x00other.h5\x00/x\x00"), e.LinkValue,
		"external link value: version/flags, file name and object path, NUL-terminated")
	file, obj, err := e.GetExternalLinkInfo()
	require.NoError(t, err)
	require.Equal(t, "other.h5", file)
	require.Equal(t, "/x", obj)
}

const h5pyLinkScript = `
import sys, json, h5py
path = sys.argv[1]
out = {}
with h5py.File(path, "r") as f:
    s = f.get("s", getlink=True)
    e = f.get("e", getlink=True)
    out["soft"] = [type(s).__name__, s.path]
    out["external"] = [type(e).__name__, e.filename, e.path]
    out["soft_value"] = f["s"][()].tolist()
    out["external_value"] = f["e"][()].tolist()
with h5py.File(path, "r+") as f:
    f.create_dataset("py", data=[1.0])
with h5py.File(path, "r") as f:
    names = []
    f.id.links.iterate(lambda n: names.append(n.decode()), idx_type=h5py.h5.INDEX_NAME)
    out["names"] = names
print(json.dumps(out))
`

// TestNewStyleRootSoftAndExternalLinks checks that soft and external links
// under a new-style root are Link messages in the root (compact or dense),
// in the format libhdf5 writes, with no object header of their own.
func TestNewStyleRootSoftAndExternalLinks(t *testing.T) {
	// 2: all compact; 7: the soft link is the 8th link and the external
	// link moves the root to dense storage; 12: dense before either.
	for _, n := range []int{2, 7, 12} {
		t.Run(fmt.Sprintf("datasets_%d", n), func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "links.h5")
			writeExternalTarget(t, dir)

			fw, err := CreateForWrite(path, CreateTruncate)
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, n)
			require.NoError(t, fw.CreateSoftLink("/s", "/d00"))
			require.NoError(t, fw.CreateExternalLink("/e", "other.h5", "/x"))
			writeRootLinks(t, fw, n, 1) // a link after the external one
			require.NoError(t, fw.Close())

			fw, err = OpenForWrite(path, OpenReadWrite)
			require.NoError(t, err)
			writeRootLinks(t, fw, n+1, 1)
			require.NoError(t, fw.Close())

			links := rootLinkMessages(t, path)
			require.Len(t, links, n+4)
			requireSoftAndExternalLinks(t, links)
			for name, lm := range links {
				require.True(t, lm.HasCreationOrder(), "link %q has no creation order", name)
			}
			require.Equal(t, uint64(n), links["s"].CreationOrder)
			require.Equal(t, uint64(n+1), links["e"].CreationOrder)
			require.Equal(t, uint64(n+2), links[rootLinkName(n)].CreationOrder)
			require.Equal(t, uint64(n+3), links[rootLinkName(n+1)].CreationOrder)

			s := readRootLinkStorage(t, path)
			require.Equal(t, n+4 > 8, s.dense)
			// Soft and external links are not objects.
			requireRootLinks(t, path, n+2)

			if h5dump := findH5Dump(); h5dump != "" {
				cmd := exec.Command(h5dump, "-H", "links.h5")
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				require.NoError(t, err, "h5dump failed:\n%s", out)
				for _, want := range []string{
					`SOFTLINK "s" {`, `LINKTARGET "/d00"`,
					`EXTERNAL_LINK "e" {`, `TARGETFILE "other.h5"`, `TARGETPATH "/x"`,
				} {
					require.Contains(t, string(out), want)
				}
				cmd = exec.Command(h5dump, "-d", "/e", "links.h5")
				cmd.Dir = dir
				out, err = cmd.CombinedOutput()
				require.NoError(t, err, "h5dump failed:\n%s", out)
				require.Contains(t, string(out), "(0): 7, 8", "the external link must resolve to other.h5:/x")
			}

			python := requireH5py(t)
			cmd := exec.Command(python, "-c", h5pyLinkScript, path)
			cmd.Dir = dir
			out, err := cmd.CombinedOutput()
			require.NoError(t, err, "h5py failed:\n%s", out)
			var res struct {
				Soft          []string  `json:"soft"`
				External      []string  `json:"external"`
				SoftValue     []float64 `json:"soft_value"`
				ExternalValue []float64 `json:"external_value"`
				Names         []string  `json:"names"`
			}
			require.NoError(t, json.Unmarshal(out, &res), "output: %s", out)
			require.Equal(t, []string{"SoftLink", "/d00"}, res.Soft)
			require.Equal(t, []string{"ExternalLink", "other.h5", "/x"}, res.External)
			require.Equal(t, []float64{0}, res.SoftValue)
			require.Equal(t, []float64{7, 8}, res.ExternalValue)
			require.Len(t, res.Names, n+5)
			require.Contains(t, res.Names, "s")
			require.Contains(t, res.Names, "e")
			require.Contains(t, res.Names, "py")
		})
	}
}

// TestOpenForWriteKeepsLibhdf5Links adds a dataset to the root of files
// h5py wrote with soft and external links (compact and dense).
func TestOpenForWriteKeepsLibhdf5Links(t *testing.T) {
	python := requireH5py(t)
	const create = `
import sys, h5py
with h5py.File(sys.argv[1], "w", libver="latest") as f:
    for i in range(int(sys.argv[2])):
        f.create_dataset("d%02d" % i, data=[float(i)])
    f["s"] = h5py.SoftLink("/d00")
    f["e"] = h5py.ExternalLink("other.h5", "/x")
`
	const check = `
import sys, json, h5py
with h5py.File(sys.argv[1], "r") as f:
    s = f.get("s", getlink=True)
    e = f.get("e", getlink=True)
    print(json.dumps({"names": sorted(f.keys()), "soft": s.path, "external": [e.filename, e.path],
                      "added": f[sys.argv[2]][()].tolist()}))
`
	for _, n := range []int{2, 10} {
		t.Run(fmt.Sprintf("datasets_%d", n), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "h5py_links.h5")
			out, err := exec.Command(python, "-c", create, path, strconv.Itoa(n)).CombinedOutput()
			require.NoError(t, err, "h5py failed:\n%s", out)

			requireSoftAndExternalLinks(t, rootLinkMessages(t, path))

			fw, err := OpenForWrite(path, OpenReadWrite)
			require.NoError(t, err)
			writeRootLinks(t, fw, n, 1)
			require.NoError(t, fw.Close())

			links := rootLinkMessages(t, path)
			require.Len(t, links, n+3)
			requireSoftAndExternalLinks(t, links)

			out, err = exec.Command(python, "-c", check, path, rootLinkName(n)).CombinedOutput()
			require.NoError(t, err, "h5py failed:\n%s", out)
			var res struct {
				Names    []string  `json:"names"`
				Soft     string    `json:"soft"`
				External []string  `json:"external"`
				Added    []float64 `json:"added"`
			}
			require.NoError(t, json.Unmarshal(out, &res), "output: %s", out)
			require.Len(t, res.Names, n+3)
			require.Equal(t, "/d00", res.Soft)
			require.Equal(t, []string{"other.h5", "/x"}, res.External)
			require.Equal(t, []float64{float64(n)}, res.Added)
		})
	}
}
