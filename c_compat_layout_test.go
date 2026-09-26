package hdf5

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/cwbudde/go-hdf5/internal/structures"
	"github.com/cwbudde/go-hdf5/internal/utils"
	"github.com/cwbudde/go-hdf5/internal/writer"
	"github.com/stretchr/testify/require"
)

// This file checks that files produced by the writer satisfy the structural
// invariants the reference HDF5 C library enforces when opening a file.
// Every violation checked here made h5py / netCDF-C reject files written by
// this library (e.g. "actual len exceeds EOA", "incorrect metadata checksum",
// "object 'x' doesn't exist", "bad dimensions for chunked storage").

const undefAddr = ^uint64(0)

// writeScenario creates one test file.
type writeScenario struct {
	name  string
	write func(t *testing.T, path string)
}

func seqFloat64(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = float64(i) + 0.5
	}
	return v
}

//nolint:gocognit // one closure per scenario
func compatScenarios() []writeScenario {
	closeOK := func(t *testing.T, fw *FileWriter) {
		t.Helper()
		require.NoError(t, fw.Close())
	}
	return []writeScenario{
		{"empty", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			closeOK(t, fw)
		}},
		{"root_attrs_compact", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate,
				WithRootAttribute("Conventions", "SOFA"),
				WithRootAttribute("Version", "1.0"),
				WithRootAttribute("N", int32(7)))
			require.NoError(t, err)
			closeOK(t, fw)
		}},
		{"root_attrs_dense", func(t *testing.T, p string) {
			opts := make([]interface{}, 0, 12)
			for i := 0; i < 12; i++ {
				opts = append(opts, WithRootAttribute(fmt.Sprintf("attr%02d", i), fmt.Sprintf("value %d", i)))
			}
			fw, err := CreateForWrite(p, CreateTruncate, opts...)
			require.NoError(t, err)
			closeOK(t, fw)
		}},
		{"groups_and_datasets", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate, WithRootAttribute("Conventions", "SOFA"))
			require.NoError(t, err)
			_, err = fw.CreateGroup("/g1")
			require.NoError(t, err)
			_, err = fw.CreateGroup("/g1/sub")
			require.NoError(t, err)
			ds, err := fw.CreateDataset("/g1/sub/x", Float64, []uint64{4})
			require.NoError(t, err)
			require.NoError(t, ds.Write(seqFloat64(4)))

			c1, err := fw.CreateDataset("/c1", Float64, []uint64{5},
				WithAttribute("units", "meter"), WithAttribute("n", int32(5)))
			require.NoError(t, err)
			require.NoError(t, c1.Write(seqFloat64(5)))
			c2, err := fw.CreateDataset("/c2", Float64, []uint64{2, 3})
			require.NoError(t, err)
			require.NoError(t, c2.Write(seqFloat64(6)))
			c3, err := fw.CreateDataset("/c3", Float64, []uint64{2, 3, 4})
			require.NoError(t, err)
			require.NoError(t, c3.Write(seqFloat64(24)))

			k1, err := fw.CreateDataset("/k1", Float64, []uint64{10}, WithChunkDims([]uint64{4}),
				WithAttribute("long_name", "chunked"))
			require.NoError(t, err)
			require.NoError(t, k1.Write(seqFloat64(10)))
			k2, err := fw.CreateDataset("/k2", Float64, []uint64{50, 40}, WithChunkDims([]uint64{3, 4}))
			require.NoError(t, err)
			require.NoError(t, k2.Write(seqFloat64(2000))) // 221 chunks: multi-level chunk B-tree
			k3, err := fw.CreateDataset("/k3", Float64, []uint64{3, 4, 5}, WithChunkDims([]uint64{2, 2, 5}),
				WithGZIPCompression(6))
			require.NoError(t, err)
			require.NoError(t, k3.Write(seqFloat64(60)))

			// Never written: allocated space must still be inside the file.
			_, err = fw.CreateDataset("/unwritten", Float64, []uint64{100})
			require.NoError(t, err)
			closeOK(t, fw)
		}},
		{"attributes_after_creation", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			a, err := fw.CreateDataset("/a", Float64, []uint64{3})
			require.NoError(t, err)
			require.NoError(t, a.Write(seqFloat64(3)))
			require.NoError(t, a.WriteAttribute("units", "Pa"))
			require.NoError(t, a.WriteAttribute("scale", 2.5))
			// Written after the attributes grew /a's object header.
			b, err := fw.CreateDataset("/b", Float64, []uint64{2, 2})
			require.NoError(t, err)
			require.NoError(t, b.Write(seqFloat64(4)))
			// Compact -> dense transition on a dataset.
			for i := 0; i < 12; i++ {
				require.NoError(t, b.WriteAttribute(fmt.Sprintf("a%02d", i), int32(i)))
			}
			g, err := fw.CreateGroup("/grp")
			require.NoError(t, err)
			require.NoError(t, g.WriteAttribute("title", "hello"))
			closeOK(t, fw)
		}},
		{"large_initial_header", func(t *testing.T, p string) {
			// Object header chunk > 255 bytes at creation (2-byte size field).
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			ds, err := fw.CreateDataset("/wide", Float32, []uint64{3},
				WithAttribute("CLASS", "DIMENSION_SCALE"),
				WithAttribute("NAME", strings.Repeat("n", 200)),
				WithAttribute("_Netcdf4Dimid", int32(4)))
			require.NoError(t, err)
			require.NoError(t, ds.Write([]float32{1, 2, 3}))
			require.NoError(t, ds.WriteAttribute("later", "added"))
			closeOK(t, fw)
		}},
		{"chunked_large_header", func(t *testing.T, p string) {
			// Chunked dataset whose initial header exceeds 255 bytes: the
			// chunk-size field is 2 bytes wide, which shifts the layout
			// message's chunk index address patched by Write.
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			opts := []DatasetOption{WithChunkDims([]uint64{4})}
			for i := 0; i < 4; i++ {
				opts = append(opts, WithAttribute(fmt.Sprintf("a%d", i), strings.Repeat("z", 100)))
			}
			ds, err := fw.CreateDataset("/kc", Float64, []uint64{10}, opts...)
			require.NoError(t, err)
			require.NoError(t, ds.Write(seqFloat64(10)))
			closeOK(t, fw)
		}},
		{"filter_orders", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			a, err := fw.CreateDataset("/fg", Float64, []uint64{40}, WithChunkDims([]uint64{16}),
				WithFletcher32(), WithGZIPCompression(6))
			require.NoError(t, err)
			require.NoError(t, a.Write(seqFloat64(40)))
			b, err := fw.CreateDataset("/gf", Float64, []uint64{40}, WithChunkDims([]uint64{16}),
				WithGZIPCompression(6), WithFletcher32())
			require.NoError(t, err)
			require.NoError(t, b.Write(seqFloat64(40)))
			closeOK(t, fw)
		}},
		{"lzf_long_runs", func(t *testing.T, p string) {
			// LZF (filter 32000, named) with long back-references, checked
			// against h5py's liblzf-based decoder.
			withLZF := func(cfg *datasetConfig) {
				if cfg.pipeline == nil {
					cfg.pipeline = writer.NewFilterPipeline()
				}
				cfg.pipeline.AddFilter(writer.NewLZFFilter())
			}
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			ds, err := fw.CreateDataset("/lz", Float64, []uint64{2000}, WithChunkDims([]uint64{500}), withLZF)
			require.NoError(t, err)
			v := make([]float64, 2000)
			for i := range v {
				v[i] = float64(i % 7)
			}
			require.NoError(t, ds.Write(v))
			closeOK(t, fw)
		}},
		{"compact_attrs_continuation", func(t *testing.T, p string) {
			// 8 attributes that overflow the header's first chunk stay
			// compact (continuation chunk); repeated rewrites reuse it.
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			ds, err := fw.CreateDataset("/c", Float64, []uint64{2})
			require.NoError(t, err)
			require.NoError(t, ds.Write([]float64{1, 2}))
			for i := 0; i < 8; i++ {
				require.NoError(t, ds.WriteAttribute(fmt.Sprintf("k%d", i), strings.Repeat("c", 60)+fmt.Sprint(i)))
			}
			for i := 0; i < 5; i++ {
				require.NoError(t, ds.WriteAttribute("k7", int32(i)))
			}
			closeOK(t, fw)
		}},
		{"dense_attrs_growing_heap", func(t *testing.T, p string) {
			long := strings.Repeat("x", 300)
			opts := make([]interface{}, 0, 200)
			for i := 0; i < 200; i++ {
				opts = append(opts, WithRootAttribute(fmt.Sprintf("root%03d", i), fmt.Sprintf("%s-%d", long, i)))
			}
			fw, err := CreateForWrite(p, CreateTruncate, opts...)
			require.NoError(t, err)
			ds, err := fw.CreateDataset("/v", Float64, []uint64{2})
			require.NoError(t, err)
			require.NoError(t, ds.Write(seqFloat64(2)))
			// Added one by one after creation: the dense heap's direct block
			// has to grow (and move) several times.
			for i := 0; i < 200; i++ {
				require.NoError(t, ds.WriteAttribute(fmt.Sprintf("a%03d", i), fmt.Sprintf("%s-%d", long, i)))
			}
			closeOK(t, fw)
		}},
		{"dense_group", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			links := map[string]string{}
			for i := 0; i < 12; i++ {
				ds, err := fw.CreateDataset(fmt.Sprintf("/d%02d", i), Float64, []uint64{2})
				require.NoError(t, err)
				require.NoError(t, ds.Write(seqFloat64(2)))
				links[fmt.Sprintf("l%02d", i)] = fmt.Sprintf("/d%02d", i)
			}
			require.NoError(t, fw.CreateDenseGroup("/dense", links))
			closeOK(t, fw)
		}},
		{"many_links", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			for i := 19; i >= 0; i-- { // reverse order: exercises sorting
				ds, err := fw.CreateDataset(fmt.Sprintf("/d%02d", i), Float64, []uint64{2})
				require.NoError(t, err)
				require.NoError(t, ds.Write(seqFloat64(2)))
			}
			closeOK(t, fw)
		}},
		{"links_100", func(t *testing.T, p string) { writeManyLinks(t, p, 100) }},
		{"links_1000", func(t *testing.T, p string) { writeManyLinks(t, p, 1000) }},
		{"vlen_dataset", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			ds, err := fw.CreateDataset("/v", VLenInt32, []uint64{3})
			require.NoError(t, err)
			require.NoError(t, ds.Write([][]int32{{1, 2, 3}, {4}, {5, 6}}))
			closeOK(t, fw)
		}},
		{"compound_dataset", writeCompoundInteropFile},
		{"superblock_v0", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate, WithSuperblockVersion(SuperblockV0),
				WithRootAttribute("Conventions", "SOFA"))
			require.NoError(t, err)
			for i := 0; i < 10; i++ {
				ds, err := fw.CreateDataset(fmt.Sprintf("/x%02d", i), Float64, []uint64{3},
					WithAttribute("units", "m"))
				require.NoError(t, err)
				require.NoError(t, ds.Write(seqFloat64(3)))
			}
			closeOK(t, fw)
		}},
		{"root_links_compact", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate, WithRootAttribute("Conventions", "SOFA"))
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, 3)
			closeOK(t, fw)
		}},
		{"root_links_dense", writeMinimalSOFA},
		{"root_links_dense_attrs", func(t *testing.T, p string) {
			opts := make([]interface{}, 0, 12)
			for i := 0; i < 12; i++ {
				opts = append(opts, WithRootAttribute(fmt.Sprintf("attr%02d", i), fmt.Sprintf("value %d", i)))
			}
			fw, err := CreateForWrite(p, CreateTruncate, opts...)
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, 30)
			closeOK(t, fw)
		}},
		{"root_links_reopen", func(t *testing.T, p string) {
			fw, err := CreateForWrite(p, CreateTruncate)
			require.NoError(t, err)
			writeRootLinks(t, fw, 0, 5)
			closeOK(t, fw)
			fw, err = OpenForWrite(p, OpenReadWrite)
			require.NoError(t, err)
			writeRootLinks(t, fw, 5, 10)
			closeOK(t, fw)
		}},
	}
}

// writeMinimalSOFA writes a small valid SimpleFreeFieldHRIR file the way
// netCDF-C lays out SOFA files: global attributes, one dimension scale per
// SOFA dimension (M=2, R=2, E=1, N=4, C=3, I=1) named like netCDF's phony
// dimensions, and float64 variables with their Type/Units attributes. With
// 15 datasets the root group uses dense link storage.
func writeMinimalSOFA(t *testing.T, path string) {
	t.Helper()
	fw, err := CreateForWrite(path, CreateTruncate,
		WithRootAttribute("Conventions", "SOFA"),
		WithRootAttribute("Version", "2.1"),
		WithRootAttribute("SOFAConventions", "SimpleFreeFieldHRIR"),
		WithRootAttribute("SOFAConventionsVersion", "1.0"),
		WithRootAttribute("DataType", "FIR"),
		WithRootAttribute("RoomType", "free field"))
	require.NoError(t, err)

	dims := map[string]uint64{"M": 2, "R": 2, "E": 1, "N": 4, "C": 3, "I": 1}
	scales := map[string]*DatasetWriter{}
	for _, name := range []string{"C", "E", "I", "M", "N", "R"} {
		n := dims[name]
		ds, err := fw.CreateDataset("/"+name, Float64, []uint64{n})
		require.NoError(t, err)
		require.NoError(t, ds.Write(make([]float64, n)))
		require.NoError(t, ds.SetDimensionScale(
			fmt.Sprintf("This is a netCDF dimension but not a netCDF variable.%10d", n)))
		scales[name] = ds
	}

	variable := func(name string, dimNames string, values []float64, attrs ...string) {
		shape := make([]uint64, len(dimNames))
		for i, d := range dimNames {
			shape[i] = dims[string(d)]
		}
		ds, err := fw.CreateDataset("/"+name, Float64, shape)
		require.NoError(t, err)
		require.NoError(t, ds.Write(values))
		for i, d := range dimNames {
			require.NoError(t, ds.AttachDimensionScale(i, scales[string(d)]))
		}
		for i := 0; i+1 < len(attrs); i += 2 {
			require.NoError(t, ds.WriteAttribute(attrs[i], attrs[i+1]))
		}
	}
	cartesian := []string{"Type", "cartesian", "Units", "metre"} //nolint:misspell // SOFA unit names
	variable("ListenerPosition", "IC", []float64{0, 0, 0}, cartesian...)
	variable("ListenerUp", "IC", []float64{0, 0, 1}, cartesian...)
	variable("ListenerView", "IC", []float64{1, 0, 0}, cartesian...)
	variable("ReceiverPosition", "RCI", []float64{0, 0.09, 0, 0, -0.09, 0}, cartesian...)
	variable("SourcePosition", "MC", []float64{0, 0, 1.2, 90, 0, 1.2},
		"Type", "spherical", "Units", "degree, degree, metre") //nolint:misspell // SOFA unit names
	variable("EmitterPosition", "ECI", []float64{0, 0, 0}, cartesian...)
	variable("Data.IR", "MRN", []float64{
		1, 0, 0, 0, 0.5, 0, 0, 0,
		0, 1, 0, 0, 0, 0.5, 0, 0,
	})
	variable("Data.SamplingRate", "I", []float64{48000}, "Units", "hertz")
	variable("Data.Delay", "IR", []float64{0, 0})
	require.NoError(t, fw.Close())
}

// manyLinkName is the name of link i written by writeManyLinks: long names
// fill the local heap quickly.
func manyLinkName(i int) string {
	return fmt.Sprintf("variable_with_a_descriptive_name_%04d", i)
}

// writeManyLinks writes n datasets at the root and n/2 in a subgroup (in
// reverse order, so every insertion reshuffles the symbol table nodes). The
// local heaps outgrow their initial 256 bytes and the group B-trees their
// single leaf of 32 symbol table nodes (256 links) when n is large.
func writeManyLinks(t *testing.T, p string, n int) {
	t.Helper()
	fw, err := CreateForWrite(p, CreateTruncate)
	require.NoError(t, err)
	_, err = fw.CreateGroup("/sub")
	require.NoError(t, err)
	for i := n - 1; i >= 0; i-- {
		ds, err := fw.CreateDataset("/"+manyLinkName(i), Float64, []uint64{1})
		require.NoError(t, err)
		require.NoError(t, ds.Write([]float64{float64(i)}))
		if i%2 == 0 {
			ds, err := fw.CreateDataset("/sub/"+manyLinkName(i), Float64, []uint64{1})
			require.NoError(t, err)
			require.NoError(t, ds.Write([]float64{float64(i)}))
		}
	}
	require.NoError(t, fw.Close())
}

// TestWrittenFilesSatisfyCLibraryInvariants validates the on-disk structure of
// files produced by the writer.
func TestWrittenFilesSatisfyCLibraryInvariants(t *testing.T) {
	for _, sc := range compatScenarios() {
		t.Run(sc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), sc.name+".h5")
			sc.write(t, path)
			data, err := os.ReadFile(path)
			require.NoError(t, err)
			newLayoutChecker(t, data).checkFile()

			// The library itself must still read the file.
			f, err := Open(path)
			require.NoError(t, err)
			require.NoError(t, f.Close())
		})
	}
}

// TestDenseAttributeStorageIsCompact guards against oversized dense storage:
// a handful of short root attributes used to allocate a 64 KiB fractal heap
// direct block (a ~75 KB file of mostly zeros).
func TestDenseAttributeStorageIsCompact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dense_small.h5")
	opts := make([]interface{}, 0, 11)
	for i := 0; i < 11; i++ {
		opts = append(opts, WithRootAttribute(fmt.Sprintf("attr%02d", i), fmt.Sprintf("value %d", i)))
	}
	fw, err := CreateForWrite(path, CreateTruncate, opts...)
	require.NoError(t, err)
	require.NoError(t, fw.Close())

	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Less(t, info.Size(), int64(12*1024), "11 short dense attributes should need only a few KiB")
}

// TestSuperblockEOAMatchesFileSize is the minimal regression test for the
// original bug: the end-of-file address in the superblock must equal the file
// size after Close, also when objects are added after creation.
func TestSuperblockEOAMatchesFileSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eoa.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/x", Float64, []uint64{16})
	require.NoError(t, err)
	require.NoError(t, ds.Write(seqFloat64(16)))
	require.NoError(t, fw.Close())

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, uint64(len(data)), binary.LittleEndian.Uint64(data[28:36]), "superblock EOA")
	require.Equal(t, binary.LittleEndian.Uint32(data[44:48]), utils.JenkinsChecksum(data[:44]), "superblock checksum")
}

type layoutChecker struct {
	t       *testing.T
	d       []byte
	eoa     uint64
	visited map[uint64]bool
}

func newLayoutChecker(t *testing.T, d []byte) *layoutChecker {
	return &layoutChecker{t: t, d: d, visited: map[uint64]bool{}}
}

func (c *layoutChecker) u64(off uint64) uint64 { return binary.LittleEndian.Uint64(c.d[off : off+8]) }

func (c *layoutChecker) u32(off uint64) uint32 { return binary.LittleEndian.Uint32(c.d[off : off+4]) }

func (c *layoutChecker) u16(off uint64) uint16 { return binary.LittleEndian.Uint16(c.d[off : off+2]) }

// inside asserts that [addr, addr+size) lies below the EOA.
func (c *layoutChecker) inside(what string, addr, size uint64) {
	c.t.Helper()
	require.NotEqual(c.t, undefAddr, addr, "%s: undefined address", what)
	require.LessOrEqual(c.t, addr+size, c.eoa, "%s at %d (+%d bytes) extends past EOA %d", what, addr, size, c.eoa)
}

func (c *layoutChecker) checkFile() {
	t := c.t
	require.Equal(t, "\x89HDF\r\n\x1a\n", string(c.d[0:8]))
	var root uint64
	switch c.d[8] {
	case 0:
		c.eoa = c.u64(40)
		root = c.u64(64)
		require.Equal(t, uint16(4), c.u16(16), "group leaf node K")
		require.Equal(t, uint16(16), c.u16(18), "group internal node K")
	case 2, 3:
		c.eoa = c.u64(28)
		root = c.u64(36)
		require.Equal(t, c.u32(44), utils.JenkinsChecksum(c.d[:44]), "superblock checksum")
	default:
		t.Fatalf("unexpected superblock version %d", c.d[8])
	}
	require.Equal(t, uint64(len(c.d)), c.eoa, "superblock EOA must equal the file size")
	c.checkObjectHeader(root)
}

type ohMessage struct {
	typ  uint8
	data []byte
}

func (c *layoutChecker) checkObjectHeader(addr uint64) {
	t := c.t
	if c.visited[addr] {
		return
	}
	c.visited[addr] = true

	var msgs []ohMessage
	if string(c.d[addr:addr+4]) == "OHDR" {
		require.Equal(t, byte(2), c.d[addr+4], "OHDR version")
		flags := c.d[addr+5]
		pos := addr + 6
		if flags&0x20 != 0 {
			pos += 16
		}
		if flags&0x10 != 0 {
			pos += 4
		}
		width := uint64(1) << (flags & 0x03)
		var chunk uint64
		for i := uint64(0); i < width; i++ {
			chunk |= uint64(c.d[pos+i]) << (8 * i)
		}
		pos += width
		end := pos + chunk
		// The C library reads prefix + chunk + 4-byte checksum.
		c.inside("object header", addr, end+4-addr)
		require.Equal(t, c.u32(end), utils.JenkinsChecksum(c.d[addr:end]), "object header checksum at %d", addr)
		msgs = c.parseV2Messages(pos, end, flags)
	} else {
		require.Equal(t, byte(1), c.d[addr], "object header v1 version at %d", addr)
		n := int(c.u16(addr + 2))
		size := uint64(c.u32(addr + 8))
		c.inside("object header v1", addr, 16+size)
		pos := addr + 16
		for i := 0; i < n; i++ {
			typ := c.u16(pos)
			sz := uint64(c.u16(pos + 2))
			require.Zero(t, sz%8, "v1 message sizes must be 8-byte aligned")
			msgs = append(msgs, ohMessage{typ: uint8(typ), data: c.d[pos+8 : pos+8+sz]})
			pos += 8 + sz
		}
		require.Equal(t, addr+16+size, pos, "v1 header size field must cover all messages")
	}

	rank := -1
	linkInfo, groupInfo := false, false
	for _, m := range msgs {
		switch m.typ {
		case 0x01: // dataspace
			rank = int(m.data[1])
		case 0x02:
			linkInfo = true
		case 0x0A:
			groupInfo = true
		}
	}
	// libhdf5 needs the Group Info message to add links to a new-style group.
	require.Equal(t, linkInfo, groupInfo, "new-style group at %d needs Link Info and Group Info messages", addr)
	for _, m := range msgs {
		switch m.typ {
		case 0x02:
			c.checkLinkInfo(m.data, msgs)
		case 0x03:
			c.checkDatatype(m.data)
		case 0x06:
			c.checkLink(m.data)
		case 0x08:
			c.checkLayout(m.data, rank)
		case 0x0A:
			_, err := core.ParseGroupInfoMessage(m.data)
			require.NoError(t, err, "group info message at %d", addr)
		case 0x11:
			c.checkSymbolTableGroup(binary.LittleEndian.Uint64(m.data[0:8]), binary.LittleEndian.Uint64(m.data[8:16]))
		case 0x15:
			c.checkDenseAttributes(m.data)
		}
	}
}

var layoutSuperblock = &core.Superblock{OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}

// checkLink validates a Link message and the object a hard link points to.
func (c *layoutChecker) checkLink(data []byte) {
	lm, err := structures.ParseLinkMessage(data, layoutSuperblock)
	require.NoError(c.t, err, "link message")
	require.NotEmpty(c.t, lm.Name, "link name")
	if lm.IsHardLink() {
		c.checkObjectHeader(lm.ObjectAddress)
	}
}

// checkLinkInfo validates the link storage of a new-style group: compact
// (Link messages in the header, no heap or name index) or dense (fractal
// heap + type 5 name index, no Link messages).
func (c *layoutChecker) checkLinkInfo(data []byte, msgs []ohMessage) {
	t := c.t
	li, err := core.ParseLinkInfoMessage(data, layoutSuperblock)
	require.NoError(t, err, "link info message")
	compact := 0
	for _, m := range msgs {
		if m.typ == 0x06 {
			compact++
		}
	}
	if li.FractalHeapAddress == undefAddr {
		require.Equal(t, undefAddr, li.NameBTreeAddress, "compact link storage has no name index")
		require.LessOrEqual(t, compact, 8, "compact groups hold at most max_compact (8) links")
		return
	}
	require.Zero(t, compact, "dense groups keep no Link messages in the header")

	heapAddr, btreeAddr := li.FractalHeapAddress, li.NameBTreeAddress
	require.Equal(t, "FRHP", string(c.d[heapAddr:heapAddr+4]))
	hdrLen := uint64(22 + 12*8 + 3*8)
	c.inside("fractal heap header", heapAddr, hdrLen+4)
	require.Equal(t, c.u32(heapAddr+hdrLen), utils.JenkinsChecksum(c.d[heapAddr:heapAddr+hdrLen]), "fractal heap header checksum")
	require.Equal(t, uint16(7), c.u16(heapAddr+5), "link heap IDs are 7 bytes (H5G_DENSE_FHEAP_ID_LEN)")

	require.Equal(t, "BTHD", string(c.d[btreeAddr:btreeAddr+4]))
	require.Equal(t, byte(5), c.d[btreeAddr+5], "link name index must be a type 5 B-tree")
	require.Equal(t, uint16(11), c.u16(btreeAddr+10), "type 5 record size")
	require.Equal(t, c.u32(btreeAddr+34), utils.JenkinsChecksum(c.d[btreeAddr:btreeAddr+34]), "B-tree v2 header checksum")
	c.inside("B-tree v2 leaf", c.u64(btreeAddr+16), uint64(c.u32(btreeAddr+6)))

	r := bytes.NewReader(c.d)
	fh, err := structures.OpenFractalHeap(r, heapAddr, 8, 8, binary.LittleEndian)
	require.NoError(t, err, "open link heap")
	ids, err := structures.ReadBTreeV2LinkNameHeapIDs(r, btreeAddr, layoutSuperblock)
	require.NoError(t, err, "read link name index")
	require.Equal(t, fh.Header.ManagedObjCount, uint64(len(ids)), "every heap object is indexed")
	for _, id := range ids {
		obj, err := fh.ReadObjectSpecCompliant(id)
		require.NoError(t, err, "read link heap object")
		c.checkLink(obj)
	}
}

func (c *layoutChecker) parseV2Messages(pos, end uint64, flags byte) []ohMessage {
	hdr := uint64(4)
	if flags&0x04 != 0 {
		hdr = 6
	}
	var msgs []ohMessage
	for pos+hdr <= end {
		typ := c.d[pos]
		sz := uint64(c.u16(pos + 1))
		require.LessOrEqual(c.t, pos+hdr+sz, end, "message overruns object header chunk")
		data := c.d[pos+hdr : pos+hdr+sz]
		if typ == 0x10 { // continuation
			caddr, clen := binary.LittleEndian.Uint64(data[0:8]), binary.LittleEndian.Uint64(data[8:16])
			c.inside("continuation chunk", caddr, clen)
			require.Equal(c.t, "OCHK", string(c.d[caddr:caddr+4]))
			require.Equal(c.t, c.u32(caddr+clen-4), utils.JenkinsChecksum(c.d[caddr:caddr+clen-4]), "OCHK checksum")
			msgs = append(msgs, c.parseV2Messages(caddr+4, caddr+clen-4, flags)...)
		} else {
			msgs = append(msgs, ohMessage{typ: typ, data: data})
		}
		pos += hdr + sz
	}
	return msgs
}

func (c *layoutChecker) checkDatatype(dt []byte) {
	class := dt[0] & 0x0F
	size := binary.LittleEndian.Uint32(dt[4:8])
	precision := binary.LittleEndian.Uint16(dt[10:12])
	switch class {
	case 0: // fixed point: bit offset 0, precision = 8*size
		require.Equal(c.t, uint16(size*8), precision, "integer precision")
	case 1: // IEEE float
		require.Equal(c.t, uint16(size*8), precision, "float precision")
		require.Equal(c.t, byte(0x20), dt[1]&0x30, "float mantissa normalization must be 'implied'")
		require.Equal(c.t, byte(size*8-1), dt[2], "float sign location")
		epos, esize, msize := dt[12], dt[13], dt[15]
		require.LessOrEqual(c.t, int(epos)+int(esize), int(precision), "exponent range")
		require.Equal(c.t, int(epos), int(msize), "mantissa directly below exponent")
		bias := binary.LittleEndian.Uint32(dt[16:20])
		require.Equal(c.t, uint32(math.Pow(2, float64(esize-1))-1), bias, "exponent bias")
	}
}

func (c *layoutChecker) checkLayout(l []byte, rank int) {
	require.Equal(c.t, byte(3), l[0], "layout message version")
	switch l[1] {
	case 1: // contiguous
		addr, size := binary.LittleEndian.Uint64(l[2:10]), binary.LittleEndian.Uint64(l[10:18])
		if addr != undefAddr {
			c.inside("contiguous data", addr, size)
		}
	case 2: // chunked
		ndims := int(l[2])
		require.Equal(c.t, rank+1, ndims, "chunked layout dimensionality must be rank+1")
		btree := binary.LittleEndian.Uint64(l[3:11])
		if btree == undefAddr {
			return
		}
		dims := make([]uint64, ndims)
		for i := range dims {
			dims[i] = uint64(binary.LittleEndian.Uint32(l[11+4*i:]))
		}
		c.checkChunkBTree(btree, dims)
	}
}

// checkChunkBTree validates a v1 raw-data chunk B-tree (node K = 32).
func (c *layoutChecker) checkChunkBTree(addr uint64, dims []uint64) {
	t := c.t
	nd := uint64(len(dims))
	keySize := 8 + 8*nd
	nodeSize := 24 + 64*8 + 65*keySize
	c.inside("chunk B-tree node", addr, nodeSize)
	require.Equal(t, "TREE", string(c.d[addr:addr+4]))
	require.Equal(t, byte(1), c.d[addr+4], "chunk B-tree node type")
	level := c.d[addr+5]
	n := uint64(c.u16(addr + 6))
	require.LessOrEqual(t, n, uint64(64), "entries per chunk B-tree node")
	pos := addr + 24
	for i := uint64(0); i <= n; i++ {
		for j := uint64(0); j < nd; j++ {
			off := c.u64(pos + 8 + 8*j)
			require.Zero(t, off%dims[j], "chunk key offset must be a multiple of the chunk dimension")
		}
		require.Zero(t, c.u64(pos+8+8*(nd-1)), "datatype dimension offset must be 0")
		pos += keySize
		if i < n {
			child := c.u64(pos)
			if level > 0 {
				c.checkChunkBTree(child, dims)
			} else {
				c.inside("chunk", child, uint64(c.u32(pos-keySize)))
			}
			pos += 8
		}
	}
}

// checkSymbolTableGroup validates an old-style (symbol table) group: keys of
// the group B-tree, sorted symbol table nodes of at most 2*leafK entries and
// the empty name at heap offset 0.
func (c *layoutChecker) checkSymbolTableGroup(btree, heap uint64) {
	t := c.t
	require.Equal(t, "HEAP", string(c.d[heap:heap+4]))
	heapSize := c.u64(heap + 8)
	heapData := c.u64(heap + 24)
	c.inside("local heap data", heapData, heapSize)
	name := func(off uint64) string {
		s := heapData + off
		e := s
		for c.d[e] != 0 {
			e++
		}
		return string(c.d[s:e])
	}
	require.Equal(t, byte(0), c.d[heapData], "heap offset 0 must hold the empty string")

	prev := ""
	c.checkGroupBTreeNode(btree, -1, name, &prev)
}

// checkGroupBTreeNode validates a group B-tree node and its subtree: levels
// decrease by one, left keys continue the previous right key, right keys are
// the largest name below their child and entries are sorted across nodes.
func (c *layoutChecker) checkGroupBTreeNode(btree uint64, wantLevel int, name func(uint64) string, prev *string) {
	t := c.t
	c.inside("group B-tree node", btree, 24+33*8+32*8)
	require.Equal(t, "TREE", string(c.d[btree:btree+4]))
	require.Equal(t, byte(0), c.d[btree+4], "group B-tree node type")
	level := int(c.d[btree+5])
	if wantLevel >= 0 {
		require.Equal(t, wantLevel, level, "group B-tree level")
	}
	n := uint64(c.u16(btree + 6))
	require.LessOrEqual(t, n, uint64(32), "group B-tree node holds at most 2*internalK children")
	require.Positive(t, n, "group B-tree node must not be empty")
	pos := btree + 24
	require.Equal(t, *prev, name(c.u64(pos)), "left key must be the previous right key")
	pos += 8
	for i := uint64(0); i < n; i++ {
		child := c.u64(pos)
		rightKey := name(c.u64(pos + 8))
		pos += 16

		if level > 0 {
			c.checkGroupBTreeNode(child, level-1, name, prev)
			require.Equal(t, *prev, rightKey, "B-tree right key must be the largest name in its child")
			continue
		}
		snod := child
		c.inside("symbol table node", snod, 8+8*40)
		require.Equal(t, "SNOD", string(c.d[snod:snod+4]))
		cnt := uint64(c.u16(snod + 6))
		require.LessOrEqual(t, cnt, uint64(8), "symbol table node holds at most 2*leafK entries")
		for e := uint64(0); e < cnt; e++ {
			ent := snod + 8 + e*40
			nm := name(c.u64(ent))
			require.Greater(t, nm, *prev, "symbol table entries must be sorted and within B-tree keys")
			*prev = nm
			c.checkObjectHeader(c.u64(ent + 8))
		}
		require.Equal(t, *prev, rightKey, "B-tree right key must be the largest name in its child")
	}
}

// checkDenseAttributes validates the fractal heap and the type 8 name index.
func (c *layoutChecker) checkDenseAttributes(ai []byte) {
	t := c.t
	pos := 2
	if ai[1]&0x01 != 0 {
		pos += 2
	}
	heapAddr := binary.LittleEndian.Uint64(ai[pos:])
	btreeAddr := binary.LittleEndian.Uint64(ai[pos+8:])

	require.Equal(t, "FRHP", string(c.d[heapAddr:heapAddr+4]))
	hdrLen := uint64(22 + 12*8 + 3*8)
	c.inside("fractal heap header", heapAddr, hdrLen+4)
	require.Equal(t, c.u32(heapAddr+hdrLen), utils.JenkinsChecksum(c.d[heapAddr:heapAddr+hdrLen]), "fractal heap header checksum")

	require.Equal(t, "BTHD", string(c.d[btreeAddr:btreeAddr+4]))
	require.Equal(t, byte(8), c.d[btreeAddr+5], "attribute name index must be a type 8 B-tree")
	require.Equal(t, uint16(17), c.u16(btreeAddr+10), "type 8 record size")
	require.Equal(t, c.u32(btreeAddr+34), utils.JenkinsChecksum(c.d[btreeAddr:btreeAddr+34]), "B-tree v2 header checksum")
	root := c.u64(btreeAddr + 16)
	c.inside("B-tree v2 leaf", root, uint64(c.u32(btreeAddr+6)))
}
