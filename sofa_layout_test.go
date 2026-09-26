package hdf5

import (
	"encoding/binary"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSOFALayoutForLibmysofa checks, without libmysofa, the layout rules
// its HDF5 reader depends on (TestLibmysofaLoad runs the reader itself), on
// a file shaped like the SOFA files go-sofa writes:
//   - libmysofa follows at most 25 continuation messages per file, so
//     DIMENSION_LIST, written at Close, must fit the first chunk of the
//     variables' object headers;
//   - it keeps global heap collection addresses in 16 bits, so the
//     collection holding the DIMENSION_LIST references must end below 64 KiB;
//   - its dense attribute reader only reads scalar strings, so variables
//     keep all their attributes compact.
func TestSOFALayoutForLibmysofa(t *testing.T) {
	path := filepath.Join(t.TempDir(), "large.sofa")
	writeSOFA(t, path, largeSOFAShape)

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	sb := f.Superblock()

	scales := map[string]bool{"C": true, "E": true, "I": true, "M": true, "N": true, "R": true}
	continuations := 0
	datasets := 0
	f.Walk(func(p string, obj Object) {
		d, ok := obj.(*Dataset)
		if !ok {
			return
		}
		datasets++
		oh, err := core.ReadObjectHeader(f.Reader(), d.Address(), sb)
		require.NoError(t, err)
		conts, attrs := 0, 0
		for _, m := range oh.Messages {
			switch m.Type {
			case core.MsgContinuation:
				conts++
			case core.MsgAttributeInfo:
				t.Errorf("%s: attributes must stay compact", p)
			case core.MsgAttribute:
				attrs++
			}
		}
		continuations += conts
		name := strings.TrimPrefix(p, "/")
		if !scales[name] {
			assert.Zero(t, conts, "%s: DIMENSION_LIST must fit the first header chunk", p)
		}
		if name == "Annotated" {
			assert.Equal(t, 11, attrs, "%s: 10 attributes plus DIMENSION_LIST", p)
		}
	})
	require.Equal(t, 30, datasets)
	assert.Less(t, continuations, 25, "libmysofa follows at most 25 continuation messages")
	assertDimensionListHeapBelow64KiB(t, f)
}

// TestDimensionListHeapReservedForReferences checks that variable-length
// data written before Close does not take the global heap collection
// reserved for DIMENSION_LIST: its references would otherwise go to a
// later collection beyond libmysofa's 64 KiB.
func TestDimensionListHeapReservedForReferences(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vlen.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	x, err := fw.CreateDataset("/x", Float64, []uint64{4})
	require.NoError(t, err)
	require.NoError(t, x.Write(seqFloat64(4)))
	require.NoError(t, x.SetDimensionScale("x"))

	// 160 KiB of data, then 10 KiB of strings: more than the reserved 4 KiB,
	// so further collections are allocated beyond 64 KiB.
	big, err := fw.CreateDataset("/big", Float64, []uint64{20000})
	require.NoError(t, err)
	require.NoError(t, big.Write(make([]float64, 20000)))
	texts := make([]string, 50)
	for i := range texts {
		texts[i] = strings.Repeat(string(rune('a'+i%26)), 200)
	}
	s, err := fw.CreateDataset("/s", VLenString, []uint64{uint64(len(texts))})
	require.NoError(t, err)
	require.NoError(t, s.Write(texts))
	// A user attribute of reference lists (5 KiB in the global heap) does
	// not take the reserved collection either.
	lists := make([][]ObjectRef, 200)
	for i := range lists {
		lists[i] = []ObjectRef{ObjectRef(big.Address())}
	}
	require.NoError(t, s.WriteAttribute("refs", lists))

	v, err := fw.CreateDataset("/v", Float64, []uint64{4})
	require.NoError(t, err)
	require.NoError(t, v.Write(seqFloat64(4)))
	require.NoError(t, v.AttachDimensionScale(0, x))
	require.NoError(t, fw.Close())

	f, err := Open(path)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	assertDimensionListHeapBelow64KiB(t, f)
}

// TestDimensionListHeapKeptByOpenForWrite checks that DIMENSION_LIST
// attributes rewritten in an OpenForWrite session keep their references in
// the collection below 64 KiB instead of a new one at the end of the file:
// the collection the existing references are in, or the empty one reserved
// at creation.
func TestDimensionListHeapKeptByOpenForWrite(t *testing.T) {
	for _, attached := range []bool{true, false} {
		t.Run(fmt.Sprintf("attached=%v", attached), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "rmw.h5")
			fw, err := CreateForWrite(path, CreateTruncate)
			require.NoError(t, err)
			x, err := fw.CreateDataset("/x", Float64, []uint64{4})
			require.NoError(t, err)
			require.NoError(t, x.Write(seqFloat64(4)))
			require.NoError(t, x.SetDimensionScale("x"))
			v, err := fw.CreateDataset("/v", Float64, []uint64{4})
			require.NoError(t, err)
			require.NoError(t, v.Write(seqFloat64(4)))
			if attached {
				require.NoError(t, v.AttachDimensionScale(0, x))
			}
			// 160 KiB of data: what the session allocates lies beyond 64 KiB.
			big, err := fw.CreateDataset("/big", Float64, []uint64{20000})
			require.NoError(t, err)
			require.NoError(t, big.Write(make([]float64, 20000)))
			require.NoError(t, fw.Close())

			fw, err = OpenForWrite(path, OpenReadWrite)
			require.NoError(t, err)
			x, err = fw.OpenDataset("/x")
			require.NoError(t, err)
			v, err = fw.OpenDataset("/v")
			require.NoError(t, err)
			w, err := fw.CreateDataset("/w", Float64, []uint64{4})
			require.NoError(t, err)
			require.NoError(t, w.Write(seqFloat64(4)))
			require.NoError(t, w.AttachDimensionScale(0, x))
			require.NoError(t, v.AttachDimensionScale(0, x))
			require.NoError(t, fw.Close())

			f, err := Open(path)
			require.NoError(t, err)
			defer func() { _ = f.Close() }()
			assertDimensionListHeapBelow64KiB(t, f)
			var xAddr uint64
			lists := map[string]interface{}{}
			f.Walk(func(p string, obj Object) {
				d, ok := obj.(*Dataset)
				if !ok {
					return
				}
				if p == "/x" {
					xAddr = d.Address()
				}
				if p == "/v" || p == "/w" {
					lists[p], err = d.ReadAttribute(dimensionListAttr)
					require.NoError(t, err, p)
				}
			})
			want := [][]ObjectRef{{ObjectRef(xAddr)}}
			assert.Equal(t, map[string]interface{}{"/v": want, "/w": want}, lists)
		})
	}
}

// assertDimensionListHeapBelow64KiB checks that the DIMENSION_LIST
// attributes of all datasets of f reference one global heap collection
// that ends below 64 KiB, which libmysofa needs.
func assertDimensionListHeapBelow64KiB(t *testing.T, f *File) {
	t.Helper()
	sb := f.Superblock()
	collections := map[uint64]bool{}
	f.Walk(func(_ string, obj Object) {
		d, ok := obj.(*Dataset)
		if !ok {
			return
		}
		oh, err := core.ReadObjectHeader(f.Reader(), d.Address(), sb)
		require.NoError(t, err)
		for _, m := range oh.Messages {
			if m.Type != core.MsgAttribute {
				continue
			}
			a, err := core.ParseAttributeMessage(m.Data, sb.Endianness)
			require.NoError(t, err)
			if a.Name != dimensionListAttr {
				continue
			}
			for i := 0; i+vlenElementSize <= len(a.Data); i += vlenElementSize {
				collections[binary.LittleEndian.Uint64(a.Data[i+4:])] = true
			}
		}
	})

	assert.Len(t, collections, 1, "all DIMENSION_LIST references in one global heap collection")
	for addr := range collections {
		size := make([]byte, 8)
		_, err := f.Reader().ReadAt(size, int64(addr)+8)
		require.NoError(t, err)
		assert.LessOrEqual(t, addr+binary.LittleEndian.Uint64(size), uint64(0x10000),
			"global heap collection at %#x must end below 64 KiB", addr)
	}
}
