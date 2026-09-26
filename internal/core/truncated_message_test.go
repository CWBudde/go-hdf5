package core

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

// messageParser parses one header message; it must return an error, never
// panic, whatever the input.
type messageParser struct {
	name  string
	parse func(data []byte) error
	valid [][]byte // well-formed messages to truncate and corrupt
}

// TestParsersRejectTruncatedMessages feeds every parser of this package
// each prefix of well-formed messages, and each prefix of copies with one
// byte set to 0xFF (flags that announce optional fields, huge sizes). A
// parser that checks len(data) < k and then reads beyond k panics here.
func TestParsersRejectTruncatedMessages(t *testing.T) {
	for _, p := range messageParsers(t) {
		t.Run(p.name, func(t *testing.T) {
			require.NotEmpty(t, p.valid)
			for vi, valid := range p.valid {
				require.NoError(t, p.parse(valid), "message %d must parse", vi)
				for pos := -1; pos < len(valid); pos++ {
					msg := append([]byte(nil), valid...)
					if pos >= 0 {
						msg[pos] = 0xFF
					}
					for n := 0; n <= len(msg); n++ {
						input := msg[:n]
						require.NotPanics(t, func() { _ = p.parse(input) },
							"message %d, byte %d = 0xFF, %d of %d bytes: % x", vi, pos, n, len(msg), input)
					}
				}
			}
		})
	}
}

func messageParsers(t *testing.T) []messageParser {
	t.Helper()
	sb := testSB()
	must := func(b []byte, err error) []byte {
		t.Helper()
		require.NoError(t, err)
		return b
	}

	f64 := &DatatypeMessage{Class: DatatypeFloat, Version: 1, Size: 8, ClassBitField: 0x20, Properties: []byte{0, 0, 64, 0, 52, 11, 0, 0, 255, 3, 0, 0}}
	i32 := &DatatypeMessage{Class: DatatypeFixed, Version: 1, Size: 4, ClassBitField: 0x08, Properties: []byte{0, 0, 32, 0}}
	str := &DatatypeMessage{Class: DatatypeString, Version: 1, Size: 5}
	datatypes := [][]byte{
		must(EncodeDatatypeMessage(f64)),
		must(EncodeDatatypeMessage(i32)),
		must(EncodeDatatypeMessage(str)),
		must(EncodeArrayDatatypeMessage(must(EncodeDatatypeMessage(i32)), []uint64{2, 3}, 24)),
		must(EncodeEnumDatatypeMessage(must(EncodeDatatypeMessage(i32)), []string{"a", "bc"}, []byte{0, 0, 0, 0, 1, 0, 0, 0}, 4)),
		must(EncodeCompoundDatatypeV3(12, []CompoundFieldDef{{Name: "x", Offset: 0, Type: f64}, {Name: "y", Offset: 8, Type: i32}})),
	}

	scalar := &DataspaceMessage{Type: DataspaceScalar}
	simple := &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{2}}
	attributes := [][]byte{
		must(EncodeAttributeMessage("units", str, scalar, []byte("meter"))),
		must(EncodeAttributeMessage("scale", f64, simple, make([]byte, 16))),
	}

	hardLink := make([]byte, 8)
	binary.LittleEndian.PutUint64(hardLink, 0x1234)
	links := [][]byte{
		must(EncodeLinkMessage(&LinkMessage{Version: 1, Type: LinkTypeHard, Name: "x", LinkValue: hardLink}, sb)),
		must(EncodeLinkMessage(&LinkMessage{Version: 1, Flags: LinkFlagCreationOrderBit, Type: LinkTypeHard, CreationOrder: 7, Name: "Data.IR", LinkValue: hardLink}, sb)),
		must(EncodeLinkMessage(&LinkMessage{Version: 1, Flags: LinkFlagLinkTypeFieldBit, Type: LinkTypeSoft, Name: "s", LinkValue: []byte("\x04\x00/abc")}, sb)),
		must(EncodeLinkMessage(&LinkMessage{Version: 1, Flags: LinkFlagLinkTypeFieldBit, Type: LinkTypeExternal, Name: "e", LinkValue: []byte("\x04\x00f.h5\x02\x00/x")}, sb)),
	}

	// Filter pipeline version 1 (deflate with a name and one client value)
	// and version 2 (deflate, level 6).
	pipelineV1 := []byte{
		1, 1, 0, 0, 0, 0, 0, 0, // version, filters, reserved
		1, 0, 8, 0, 0, 0, 1, 0, // deflate, name length, flags, 1 value
		'd', 'e', 'f', 'l', 'a', 't', 'e', 0,
		6, 0, 0, 0, 0, 0, 0, 0, // level 6, padding
	}
	pipelineV2 := []byte{2, 1, 1, 0, 0, 0, 1, 0, 6, 0, 0, 0}

	continuation := make([]byte, 16)
	binary.LittleEndian.PutUint64(continuation, 0x100)
	binary.LittleEndian.PutUint64(continuation[8:], 0x40)
	heapRef := append(append([]byte(nil), continuation[:8]...), 1, 0, 0, 0)

	return []messageParser{
		{
			name: "dataspace", parse: func(d []byte) error { _, err := ParseDataspaceMessage(d); return err },
			valid: [][]byte{
				must(EncodeDataspaceMessage([]uint64{3, 4}, []uint64{10, 20})),
				{2, 0, 0, 0}, // scalar
				{1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0},
			},
		},
		{name: "datatype", parse: func(d []byte) error {
			dt, err := ParseDatatypeMessage(d)
			if err == nil && dt.Class == DatatypeCompound {
				_, err = ParseCompoundType(dt)
			}
			return err
		}, valid: datatypes},
		{
			name: "attribute", parse: func(d []byte) error { _, err := ParseAttributeMessage(d, binary.LittleEndian); return err },
			valid: attributes,
		},
		{
			name: "attribute info", parse: func(d []byte) error { _, err := ParseAttributeInfoMessage(d, sb); return err },
			valid: [][]byte{
				must(EncodeAttributeInfoMessage(&AttributeInfoMessage{FractalHeapAddr: 0x100, BTreeNameIndexAddr: 0x200}, sb)),
				must(EncodeAttributeInfoMessage(&AttributeInfoMessage{Flags: 3, MaxCreationIndex: 4, FractalHeapAddr: 0x100, BTreeNameIndexAddr: 0x200, BTreeOrderIndexAddr: 0x300}, sb)),
			},
		},
		{
			name: "data layout", parse: func(d []byte) error { _, err := ParseDataLayoutMessage(d, sb); return err },
			valid: [][]byte{
				must(EncodeLayoutMessage(LayoutContiguous, 64, 0x800, sb, nil)),
				must(EncodeLayoutMessage(LayoutChunked, 64, 0x800, sb, []uint64{4, 2})),
				{3, byte(LayoutCompact), 4, 0, 1, 2, 3, 4},
			},
		},
		{
			name: "filter pipeline", parse: func(d []byte) error { _, err := ParseFilterPipelineMessage(d); return err },
			valid: [][]byte{pipelineV1, pipelineV2},
		},
		{
			name: "group info", parse: func(d []byte) error { _, err := ParseGroupInfoMessage(d); return err },
			valid: [][]byte{
				EncodeGroupInfoMessage(&GroupInfoMessage{}),
				EncodeGroupInfoMessage(&GroupInfoMessage{Flags: 3, MaxCompact: 8, MinDense: 6, EstNumEntries: 4, EstNameLen: 8}),
			},
		},
		{
			name: "link", parse: func(d []byte) error { _, err := ParseLinkMessage(d, sb); return err },
			valid: links,
		},
		{
			name: "link info", parse: func(d []byte) error { _, err := ParseLinkInfoMessage(d, sb); return err },
			valid: [][]byte{
				must(EncodeLinkInfoMessage(&LinkInfoMessage{FractalHeapAddress: 0x100, NameBTreeAddress: 0x200}, sb)),
				must(EncodeLinkInfoMessage(&LinkInfoMessage{Flags: 3, MaxCreationOrder: 9, FractalHeapAddress: 0x100, NameBTreeAddress: 0x200, CreationOrderBTreeAddress: 0x300}, sb)),
			},
		},
		{
			name: "global heap reference", parse: func(d []byte) error { _, err := ParseGlobalHeapReference(d, 8); return err },
			valid: [][]byte{heapRef},
		},
		{
			name: "continuation", parse: func(d []byte) error { _, err := parseContinuationMessage(d, sb); return err },
			valid: [][]byte{continuation},
		},
	}
}
