package hdf5

import (
	"os"
	"testing"

	"github.com/meko-christian/go-hdf5/internal/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRootAttributes_SingleAttribute_RoundTrip tests creating a file with one root attribute
// and verifying it can be read back correctly.
// This is Phase 3.2 - Compact Storage Implementation.
func TestRootAttributes_SingleAttribute_RoundTrip(t *testing.T) {
	tmpfile := t.TempDir() + "/single_root_attr.h5"
	defer os.Remove(tmpfile)

	// RED: Write the test first, watch it fail
	// Create file with single root attribute
	fw, err := CreateForWrite(tmpfile, CreateTruncate,
		WithRootAttribute("TestAttr", "TestValue"))
	require.NoError(t, err, "CreateForWrite should succeed")
	require.NotNil(t, fw)

	err = fw.Close()
	require.NoError(t, err, "Close should succeed")

	// Reopen and verify the attribute exists and has correct value
	f, err := Open(tmpfile)
	require.NoError(t, err, "Open should succeed")
	defer f.Close()

	rootGroup := f.Root()
	require.NotNil(t, rootGroup, "Root group should exist")

	attrs, err := rootGroup.Attributes()
	require.NoError(t, err, "Getting attributes should succeed")
	require.NotNil(t, attrs, "Attributes should not be nil")
	require.Len(t, attrs, 1, "Should have exactly 1 attribute")

	// Find the TestAttr attribute
	var testAttr *core.Attribute
	for _, attr := range attrs {
		if attr.Name == "TestAttr" {
			testAttr = attr
			break
		}
	}
	require.NotNil(t, testAttr, "TestAttr should exist")

	// Read the attribute value
	value, err := testAttr.ReadValue()
	require.NoError(t, err, "Reading attribute should succeed")
	assert.Equal(t, "TestValue", value, "Attribute value should match")
}

// TestRootAttributes_MultipleAttributes_RoundTrip tests creating a file with multiple root attributes
// and verifying they can all be read back correctly.
func TestRootAttributes_MultipleAttributes_RoundTrip(t *testing.T) {
	tmpfile := t.TempDir() + "/multiple_root_attrs.h5"
	defer os.Remove(tmpfile)

	// RED: Create file with 5 attributes (compact storage)
	fw, err := CreateForWrite(tmpfile, CreateTruncate,
		WithRootAttribute("StringAttr", "test string"),
		WithRootAttribute("Int32Attr", int32(42)),
		WithRootAttribute("Float64Attr", 3.14159),
		WithRootAttribute("Int32ArrayAttr", []int32{1, 2, 3, 4, 5}),
		WithRootAttribute("Conventions", "TestFormat"))
	require.NoError(t, err, "CreateForWrite should succeed")
	require.NotNil(t, fw)

	err = fw.Close()
	require.NoError(t, err, "Close should succeed")

	// Reopen and verify all attributes
	f, err := Open(tmpfile)
	require.NoError(t, err, "Open should succeed")
	defer f.Close()

	rootGroup := f.Root()
	require.NotNil(t, rootGroup, "Root group should exist")

	attrs, err := rootGroup.Attributes()
	require.NoError(t, err, "Getting attributes should succeed")
	require.NotNil(t, attrs, "Attributes should not be nil")
	require.Len(t, attrs, 5, "Should have exactly 5 attributes")

	// Helper to find attribute by name
	findAttr := func(name string) *core.Attribute {
		for _, attr := range attrs {
			if attr.Name == name {
				return attr
			}
		}
		return nil
	}

	// Verify string attribute
	strAttr := findAttr("StringAttr")
	require.NotNil(t, strAttr)
	strVal, err := strAttr.ReadValue()
	require.NoError(t, err)
	assert.Equal(t, "test string", strVal)

	// Verify int32 attribute
	intAttr := findAttr("Int32Attr")
	require.NotNil(t, intAttr)
	intVal, err := intAttr.ReadValue()
	require.NoError(t, err)
	assert.Equal(t, int32(42), intVal)

	// Verify float64 attribute
	floatAttr := findAttr("Float64Attr")
	require.NotNil(t, floatAttr)
	floatVal, err := floatAttr.ReadValue()
	require.NoError(t, err)
	assert.InDelta(t, 3.14159, floatVal.(float64), 0.00001)

	// Verify int32 array attribute
	arrayAttr := findAttr("Int32ArrayAttr")
	require.NotNil(t, arrayAttr)
	arrayVal, err := arrayAttr.ReadValue()
	require.NoError(t, err)
	assert.Equal(t, []int32{1, 2, 3, 4, 5}, arrayVal)

	// Verify Conventions attribute
	convAttr := findAttr("Conventions")
	require.NotNil(t, convAttr)
	convVal, err := convAttr.ReadValue()
	require.NoError(t, err)
	assert.Equal(t, "TestFormat", convVal)
}

// TestRootAttributes_EightAttributes tests the compact storage limit (8 attributes)
func TestRootAttributes_EightAttributes_RoundTrip(t *testing.T) {
	tmpfile := t.TempDir() + "/eight_root_attrs.h5"
	defer os.Remove(tmpfile)

	// RED: Create file with 8 attributes (maximum for compact storage)
	fw, err := CreateForWrite(tmpfile, CreateTruncate,
		WithRootAttribute("Attr1", "value1"),
		WithRootAttribute("Attr2", "value2"),
		WithRootAttribute("Attr3", "value3"),
		WithRootAttribute("Attr4", "value4"),
		WithRootAttribute("Attr5", "value5"),
		WithRootAttribute("Attr6", "value6"),
		WithRootAttribute("Attr7", "value7"),
		WithRootAttribute("Attr8", "value8"))
	require.NoError(t, err, "CreateForWrite should succeed")
	require.NotNil(t, fw)

	err = fw.Close()
	require.NoError(t, err, "Close should succeed")

	// Reopen and verify all 8 attributes
	f, err := Open(tmpfile)
	require.NoError(t, err, "Open should succeed")
	defer f.Close()

	rootGroup := f.Root()
	attrs, err := rootGroup.Attributes()
	require.NoError(t, err, "Getting attributes should succeed")
	require.Len(t, attrs, 8, "Should have exactly 8 attributes")

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
	for i := 1; i <= 8; i++ {
		attrName := "Attr" + string(rune('0'+i))
		expectedValue := "value" + string(rune('0'+i))

		attr := findAttr(attrName)
		require.NotNil(t, attr, "Attribute %s should exist", attrName)

		value, err := attr.ReadValue()
		require.NoError(t, err, "Reading %s should succeed", attrName)
		assert.Equal(t, expectedValue, value, "%s value should match", attrName)
	}
}
