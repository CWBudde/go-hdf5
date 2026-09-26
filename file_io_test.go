package hdf5

import (
	"bytes"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOpenReader_BytesReader(t *testing.T) {
	data, err := os.ReadFile("testdata/dimscales/h5py_dimscales.h5")
	require.NoError(t, err)

	f, err := OpenReader(bytes.NewReader(data), int64(len(data)))
	require.NoError(t, err)
	ds := datasetAt(t, f, "/grp/v")
	vals, err := ds.Read()
	require.NoError(t, err)
	require.Equal(t, []float64{1, 1, 1}, vals)
	scales, err := ds.AttachedScales(0)
	require.NoError(t, err)
	require.Equal(t, []string{"/time"}, scalePaths(t, scales))
	require.NoError(t, f.Close())
	require.NoError(t, f.Close())
}

func TestOpenReader_Errors(t *testing.T) {
	data, err := os.ReadFile("testdata/dimscales/h5py_dimscales.h5")
	require.NoError(t, err)

	_, err = OpenReader(nil, 0)
	require.Error(t, err)
	_, err = OpenReader(bytes.NewReader(data), -1)
	require.Error(t, err)
	_, err = OpenReader(bytes.NewReader([]byte("not hdf5 at all")), 15)
	require.Error(t, err)

	// The size bounds every read: a truncated view must fail cleanly even
	// though the underlying reader holds the whole file.
	for _, size := range []int64{8, 100, 1000, int64(len(data)) / 2} {
		f, err := OpenReader(bytes.NewReader(data), size)
		if err != nil {
			continue
		}
		f.Walk(func(_ string, obj Object) {
			if ds, ok := obj.(*Dataset); ok {
				_, _ = ds.Read()
				_, _ = ds.Shape()
			}
		})
	}
}

// failingWriter fails every write.
type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }

// writeSampleFile writes groups, datasets, attributes (compact, dense and
// variable-length) and dimension scales through fw.
func writeSampleFile(t *testing.T, fw *FileWriter) {
	t.Helper()
	x, err := fw.CreateDataset("/x", Float64, []uint64{3})
	require.NoError(t, err)
	require.NoError(t, x.Write([]float64{1, 2, 3}))
	require.NoError(t, x.SetDimensionScale("x"))

	_, err = fw.CreateGroup("/g")
	require.NoError(t, err)
	d, err := fw.CreateDataset("/g/d", Float64, []uint64{3, 2}, WithChunkDims([]uint64{1, 2}), WithGZIPCompression(4))
	require.NoError(t, err)
	require.NoError(t, d.Write([]float64{1, 2, 3, 4, 5, 6}))
	require.NoError(t, d.WriteAttribute("units", "m"))
	for i := range 10 {
		require.NoError(t, d.WriteAttribute(string(rune('a'+i))+"_attr", int32(i)))
	}
	require.NoError(t, d.AttachDimensionScale(0, x))

	// Allocated but never written: Close must extend the file to its EOA.
	_, err = fw.CreateDataset("/unwritten", Float64, []uint64{4})
	require.NoError(t, err)
}

func TestCreateForWriteTo_RoundTrip(t *testing.T) {
	var buf bytes.Buffer
	fw, err := CreateForWriteTo(&buf)
	require.NoError(t, err)
	writeSampleFile(t, fw)
	require.Zero(t, buf.Len(), "nothing is written before Close")
	require.NoError(t, fw.Close())
	require.NoError(t, fw.Close())
	require.NotZero(t, buf.Len())

	// Same bytes as writing to a file.
	path := filepath.Join(t.TempDir(), "file.h5")
	fw2, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	writeSampleFile(t, fw2)
	require.NoError(t, fw2.Close())
	onDisk, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, onDisk, buf.Bytes())

	f, err := OpenReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	require.NoError(t, err)
	d := datasetAt(t, f, "/g/d")
	vals, err := d.Read()
	require.NoError(t, err)
	require.Equal(t, []float64{1, 2, 3, 4, 5, 6}, vals)
	shape, err := d.Shape()
	require.NoError(t, err)
	require.Equal(t, []uint64{3, 2}, shape)
	scales, err := d.AttachedScales(0)
	require.NoError(t, err)
	require.Equal(t, []string{"/x"}, scalePaths(t, scales))
	attrs, err := d.ListAttributes()
	require.NoError(t, err)
	require.Len(t, attrs, 12) // units, 10 ints, DIMENSION_LIST
	unwritten, err := datasetAt(t, f, "/unwritten").Read()
	require.NoError(t, err)
	require.Equal(t, []float64{0, 0, 0, 0}, unwritten)

	// The HDF5 C library reads the bytes too (not netCDF-C: the scale
	// lives in another group than the dataset it labels).
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	if err := exec.Command(python, "-c", "import h5py").Run(); err != nil {
		t.Skip("h5py not available")
	}
	memPath := filepath.Join(t.TempDir(), "mem.h5")
	require.NoError(t, os.WriteFile(memPath, buf.Bytes(), 0o600))
	const script = `
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    d = f["/g/d"]
    assert d.shape == (3, 2), d.shape
    assert float(np.sum(d[()])) == 21.0
    assert d.attrs["units"] == b"m", d.attrs["units"]
    assert d.dims[0][0].name == "/x", d.dims[0][0].name
    assert f["/x"].attrs["CLASS"] == b"DIMENSION_SCALE"
    assert float(np.sum(f["/unwritten"][()])) == 0.0
print("ok")
`
	out, err := exec.Command(python, "-c", script, memPath).CombinedOutput()
	require.NoError(t, err, "h5py failed to read in-memory file:\n%s", out)
}

func TestCreateForWriteTo_OffsetWriterAndErrors(t *testing.T) {
	_, err := CreateForWriteTo(nil)
	require.Error(t, err)
	_, err = CreateForWriteTo(io.Discard, 42)
	require.Error(t, err)

	// An io.WriterAt target via io.NewOffsetWriter, at a non-zero offset.
	path := filepath.Join(t.TempDir(), "embedded.bin")
	osf, err := os.Create(path)
	require.NoError(t, err)
	const off = 512
	fw, err := CreateForWriteTo(io.NewOffsetWriter(osf, off), WithSuperblockVersion(SuperblockV0))
	require.NoError(t, err)
	x, err := fw.CreateDataset("/x", Int32, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, x.Write([]int32{7, 9}))
	require.NoError(t, fw.Close())
	st, err := osf.Stat()
	require.NoError(t, err)
	require.NoError(t, osf.Close())

	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	f, err := OpenReader(io.NewSectionReader(bytes.NewReader(raw), off, st.Size()-off), st.Size()-off)
	require.NoError(t, err)
	require.EqualValues(t, 0, f.SuperblockVersion())
	vals, err := datasetAt(t, f, "/x").Read()
	require.NoError(t, err)
	require.Equal(t, []float64{7, 9}, vals)

	// Destination errors surface from Close.
	fw, err = CreateForWriteTo(failingWriter{})
	require.NoError(t, err)
	err = fw.Close()
	require.ErrorContains(t, err, "disk full")
}
