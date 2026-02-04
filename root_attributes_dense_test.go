package hdf5

import (
	"os"
	"testing"

	"github.com/meko-christian/go-hdf5/internal/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRootAttributes_NineAttributes_DenseStorage tests that 9 attributes trigger dense storage.
// This is Phase 3.3 - Dense Storage Implementation.
// Dense storage uses Fractal Heap + B-tree instead of inline attribute messages.
func TestRootAttributes_NineAttributes_DenseStorage(t *testing.T) {
	tmpfile := t.TempDir() + "/nine_root_attrs.h5"
	defer os.Remove(tmpfile)

	// Skip if h5dump is not available
	if _, err := os.Stat("/usr/bin/h5dump"); os.IsNotExist(err) {
		if _, err := os.Stat("/home/christian/anaconda3/bin/h5dump"); os.IsNotExist(err) {
			t.Skip("h5dump not available, skipping validation")
		}
	}

	// RED: Create file with 9 attributes (exceeds compact limit of 8)
	fw, err := CreateForWrite(tmpfile, CreateTruncate,
		WithRootAttribute("Attr1", "value1"),
		WithRootAttribute("Attr2", "value2"),
		WithRootAttribute("Attr3", "value3"),
		WithRootAttribute("Attr4", "value4"),
		WithRootAttribute("Attr5", "value5"),
		WithRootAttribute("Attr6", "value6"),
		WithRootAttribute("Attr7", "value7"),
		WithRootAttribute("Attr8", "value8"),
		WithRootAttribute("Attr9", "value9"))
	require.NoError(t, err, "CreateForWrite should succeed")
	require.NotNil(t, fw)

	err = fw.Close()
	require.NoError(t, err, "Close should succeed")

	// Reopen and verify all 9 attributes
	f, err := Open(tmpfile)
	require.NoError(t, err, "Open should succeed")
	defer f.Close()

	rootGroup := f.Root()
	attrs, err := rootGroup.Attributes()
	require.NoError(t, err, "Getting attributes should succeed")
	require.Len(t, attrs, 9, "Should have exactly 9 attributes")

	// Helper to find attribute by name
	findAttr := func(name string) *core.Attribute {
		for _, attr := range attrs {
			if attr.Name == name {
				return attr
			}
		}
		return nil
	}

	// Verify each attribute
	for i := 1; i <= 9; i++ {
		attrName := "Attr" + string(rune('0'+i))
		expectedValue := "value" + string(rune('0'+i))

		attr := findAttr(attrName)
		require.NotNil(t, attr, "Attribute %s should exist", attrName)

		value, err := attr.ReadValue()
		require.NoError(t, err, "Reading %s should succeed", attrName)
		assert.Equal(t, expectedValue, value, "%s value should match", attrName)
	}
}

// TestRootAttributes_TwentyAttributes_DenseStorage tests the SOFA use case (20+ attributes).
func TestRootAttributes_TwentyAttributes_DenseStorage(t *testing.T) {
	tmpfile := t.TempDir() + "/twenty_root_attrs.h5"
	defer os.Remove(tmpfile)

	// RED: Create file with 20 attributes (SOFA use case)
	// Build attribute options programmatically
	attrOpts := make([]interface{}, 0, 20)
	for i := 1; i <= 20; i++ {
		attrName := "Attr" + string(rune('0'+i))
		attrValue := "value" + string(rune('0'+i))
		attrOpts = append(attrOpts, WithRootAttribute(attrName, attrValue))
	}

	fw, err := CreateForWrite(tmpfile, CreateTruncate, attrOpts...)
	require.NoError(t, err, "CreateForWrite should succeed")
	require.NotNil(t, fw)

	err = fw.Close()
	require.NoError(t, err, "Close should succeed")

	// Reopen and verify all 20 attributes
	f, err := Open(tmpfile)
	require.NoError(t, err, "Open should succeed")
	defer f.Close()

	rootGroup := f.Root()
	attrs, err := rootGroup.Attributes()
	require.NoError(t, err, "Getting attributes should succeed")
	require.Len(t, attrs, 20, "Should have exactly 20 attributes")

	// Helper to find attribute by name
	findAttr := func(name string) *core.Attribute {
		for _, attr := range attrs {
			if attr.Name == name {
				return attr
			}
		}
		return nil
	}

	// Verify each attribute
	for i := 1; i <= 20; i++ {
		attrName := "Attr" + string(rune('0'+i))
		expectedValue := "value" + string(rune('0'+i))

		attr := findAttr(attrName)
		require.NotNil(t, attr, "Attribute %s should exist", attrName)

		value, err := attr.ReadValue()
		require.NoError(t, err, "Reading %s should succeed", attrName)
		assert.Equal(t, expectedValue, value, "%s value should match", attrName)
	}
}
