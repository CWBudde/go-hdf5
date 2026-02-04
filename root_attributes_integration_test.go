package hdf5

import (
	"fmt"
	"os"
	"testing"

	"github.com/meko-christian/go-hdf5/internal/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCreateWithRootAttributes is a parametric test for various attribute counts.
// Tests compact storage (≤8) and dense storage (>8) transitions.
func TestCreateWithRootAttributes(t *testing.T) {
	testCases := []struct {
		name       string
		attrCount  int
		expectType string // "compact" or "dense"
	}{
		{"0_attributes", 0, "none"},
		{"1_attribute", 1, "compact"},
		{"5_attributes", 5, "compact"},
		{"8_attributes_limit", 8, "compact"},
		{"9_attributes_dense", 9, "dense"},
		{"20_attributes_sofa", 20, "dense"},
		{"50_attributes_stress", 50, "dense"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tmpfile := t.TempDir() + "/" + tc.name + ".h5"
			defer os.Remove(tmpfile)

			// Create attribute options
			opts := make([]interface{}, 0, tc.attrCount)
			for i := 1; i <= tc.attrCount; i++ {
				attrName := fmt.Sprintf("Attr%d", i)
				attrValue := fmt.Sprintf("value%d", i)
				opts = append(opts, WithRootAttribute(attrName, attrValue))
			}

			// Create file
			fw, err := CreateForWrite(tmpfile, CreateTruncate, opts...)
			require.NoError(t, err, "CreateForWrite should succeed for %d attributes", tc.attrCount)
			require.NotNil(t, fw)

			err = fw.Close()
			require.NoError(t, err, "Close should succeed")

			// Reopen and verify
			f, err := Open(tmpfile)
			require.NoError(t, err, "Open should succeed")
			defer f.Close()

			rootGroup := f.Root()
			attrs, err := rootGroup.Attributes()
			require.NoError(t, err, "Getting attributes should succeed")
			require.Len(t, attrs, tc.attrCount, "Should have exactly %d attributes", tc.attrCount)

			// Verify all attributes
			for i := 1; i <= tc.attrCount; i++ {
				attrName := fmt.Sprintf("Attr%d", i)
				expectedValue := fmt.Sprintf("value%d", i)

				var found bool
				for _, attr := range attrs {
					if attr.Name == attrName {
						found = true
						value, err := attr.ReadValue()
						require.NoError(t, err, "Reading %s should succeed", attrName)
						assert.Equal(t, expectedValue, value, "%s value should match", attrName)
						break
					}
				}
				require.True(t, found, "Attribute %s should exist", attrName)
			}

			t.Logf("✓ Successfully round-tripped %d attributes (%s storage)", tc.attrCount, tc.expectType)
		})
	}
}

// TestRootAttributeTypes tests various attribute datatypes.
func TestRootAttributeTypes(t *testing.T) {
	testCases := []struct {
		name     string
		attrName string
		value    interface{}
		expected interface{}
	}{
		{"string", "StringAttr", "test string", "test string"},
		{"int32", "Int32Attr", int32(42), int32(42)},
		{"int64", "Int64Attr", int64(12345), int64(12345)},
		{"float32", "Float32Attr", float32(3.14), float32(3.14)},
		{"float64", "Float64Attr", float64(2.718281828), float64(2.718281828)},
		// Note: uint8, int16, and other non-standard sizes not yet supported
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tmpfile := t.TempDir() + "/type_" + tc.name + ".h5"
			defer os.Remove(tmpfile)

			// Create file with single attribute of specific type
			fw, err := CreateForWrite(tmpfile, CreateTruncate,
				WithRootAttribute(tc.attrName, tc.value))
			require.NoError(t, err, "CreateForWrite should succeed")
			require.NotNil(t, fw)

			err = fw.Close()
			require.NoError(t, err, "Close should succeed")

			// Reopen and verify type
			f, err := Open(tmpfile)
			require.NoError(t, err, "Open should succeed")
			defer f.Close()

			rootGroup := f.Root()
			attrs, err := rootGroup.Attributes()
			require.NoError(t, err, "Getting attributes should succeed")
			require.Len(t, attrs, 1, "Should have exactly 1 attribute")

			attr := attrs[0]
			require.Equal(t, tc.attrName, attr.Name, "Attribute name should match")

			value, err := attr.ReadValue()
			require.NoError(t, err, "Reading value should succeed")
			assert.Equal(t, tc.expected, value, "Value should match for type %s", tc.name)

			t.Logf("✓ Type %s: wrote %v, read %v", tc.name, tc.value, value)
		})
	}
}

// TestRootAttributeTypes_Arrays tests array attribute types.
func TestRootAttributeTypes_Arrays(t *testing.T) {
	testCases := []struct {
		name     string
		attrName string
		value    interface{}
	}{
		{"int32_array", "Int32Array", []int32{1, 2, 3, 4, 5}},
		{"float64_array", "Float64Array", []float64{1.1, 2.2, 3.3}},
		// Note: uint8 arrays not yet supported
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tmpfile := t.TempDir() + "/array_" + tc.name + ".h5"
			defer os.Remove(tmpfile)

			// Create file with array attribute
			fw, err := CreateForWrite(tmpfile, CreateTruncate,
				WithRootAttribute(tc.attrName, tc.value))
			require.NoError(t, err, "CreateForWrite should succeed")
			require.NotNil(t, fw)

			err = fw.Close()
			require.NoError(t, err, "Close should succeed")

			// Reopen and verify
			f, err := Open(tmpfile)
			require.NoError(t, err, "Open should succeed")
			defer f.Close()

			rootGroup := f.Root()
			attrs, err := rootGroup.Attributes()
			require.NoError(t, err, "Getting attributes should succeed")
			require.Len(t, attrs, 1, "Should have exactly 1 attribute")

			attr := attrs[0]
			require.Equal(t, tc.attrName, attr.Name, "Attribute name should match")

			value, err := attr.ReadValue()
			require.NoError(t, err, "Reading array should succeed")
			assert.Equal(t, tc.value, value, "Array value should match")

			t.Logf("✓ Array type %s round-tripped successfully", tc.name)
		})
	}
}

// TestRootAttributeErrors tests error handling.
func TestRootAttributeErrors(t *testing.T) {
	t.Run("empty_attribute_name", func(t *testing.T) {
		tmpfile := t.TempDir() + "/empty_name.h5"
		defer os.Remove(tmpfile)

		// Empty attribute name should fail at CreateForWrite
		fw, err := CreateForWrite(tmpfile, CreateTruncate,
			WithRootAttribute("", "value"))

		if err != nil {
			// Expected: validation fails early
			require.Error(t, err, "Should fail with empty attribute name")
			require.Contains(t, err.Error(), "attribute name cannot be empty")
		} else {
			// If it succeeded, closing should fail
			require.NotNil(t, fw)
			err = fw.Close()
			require.Error(t, err, "Should fail with empty attribute name")
		}
	})

	t.Run("nil_attribute_value", func(t *testing.T) {
		tmpfile := t.TempDir() + "/nil_value.h5"
		defer os.Remove(tmpfile)

		fw, err := CreateForWrite(tmpfile, CreateTruncate,
			WithRootAttribute("NilAttr", nil))

		// Should fail when trying to infer datatype from nil
		if err == nil {
			// If CreateForWrite succeeded, Close should fail
			err = fw.Close()
			require.Error(t, err, "Should fail with nil value")
		}
	})
}

// TestRootAttributeMixedTypes tests multiple attributes with different types.
func TestRootAttributeMixedTypes(t *testing.T) {
	tmpfile := t.TempDir() + "/mixed_types.h5"
	defer os.Remove(tmpfile)

	fw, err := CreateForWrite(tmpfile, CreateTruncate,
		WithRootAttribute("StringAttr", "metadata"),
		WithRootAttribute("IntAttr", int32(42)),
		WithRootAttribute("FloatAttr", float64(3.14159)),
		WithRootAttribute("ArrayAttr", []int32{1, 2, 3}))
	require.NoError(t, err, "CreateForWrite should succeed")
	require.NotNil(t, fw)

	err = fw.Close()
	require.NoError(t, err, "Close should succeed")

	// Reopen and verify all types
	f, err := Open(tmpfile)
	require.NoError(t, err, "Open should succeed")
	defer f.Close()

	rootGroup := f.Root()
	attrs, err := rootGroup.Attributes()
	require.NoError(t, err, "Getting attributes should succeed")
	require.Len(t, attrs, 4, "Should have 4 attributes")

	// Helper to find attribute
	findAttr := func(name string) *core.Attribute {
		for _, attr := range attrs {
			if attr.Name == name {
				return attr
			}
		}
		return nil
	}

	// Verify string
	stringAttr := findAttr("StringAttr")
	require.NotNil(t, stringAttr)
	val, err := stringAttr.ReadValue()
	require.NoError(t, err)
	assert.Equal(t, "metadata", val)

	// Verify int
	intAttr := findAttr("IntAttr")
	require.NotNil(t, intAttr)
	val, err = intAttr.ReadValue()
	require.NoError(t, err)
	assert.Equal(t, int32(42), val)

	// Verify float
	floatAttr := findAttr("FloatAttr")
	require.NotNil(t, floatAttr)
	val, err = floatAttr.ReadValue()
	require.NoError(t, err)
	assert.Equal(t, float64(3.14159), val)

	// Verify array
	arrayAttr := findAttr("ArrayAttr")
	require.NotNil(t, arrayAttr)
	val, err = arrayAttr.ReadValue()
	require.NoError(t, err)
	assert.Equal(t, []int32{1, 2, 3}, val)

	t.Log("✓ All mixed types verified successfully")
}
