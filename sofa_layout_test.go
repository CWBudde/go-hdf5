package hdf5

import (
	"encoding/binary"
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
	collections := map[uint64]bool{}
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
				a, err := core.ParseAttributeMessage(m.Data, sb.Endianness)
				require.NoError(t, err)
				if a.Name != dimensionListAttr {
					continue
				}
				for i := 0; i+vlenElementSize <= len(a.Data); i += vlenElementSize {
					collections[binary.LittleEndian.Uint64(a.Data[i+4:])] = true
				}
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

	assert.Len(t, collections, 1, "all DIMENSION_LIST references in one global heap collection")
	for addr := range collections {
		size := make([]byte, 8)
		_, err := f.Reader().ReadAt(size, int64(addr)+8)
		require.NoError(t, err)
		assert.LessOrEqual(t, addr+binary.LittleEndian.Uint64(size), uint64(0x10000),
			"global heap collection at %#x must end below 64 KiB", addr)
	}
}
