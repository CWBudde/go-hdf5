package hdf5

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestWithRootAttribute_API verifies the WithRootAttribute API compiles and accepts options.
// This is Phase 3.1 - API design only. Actual attribute writing will be implemented in Phase 3.2.
func TestWithRootAttribute_API(t *testing.T) {
	tmpfile := t.TempDir() + "/test_root_attr_api.h5"
	defer os.Remove(tmpfile)

	// Test 1: File creation without root attributes (backward compatibility)
	t.Run("backward_compatibility", func(t *testing.T) {
		fw, err := CreateForWrite(tmpfile, CreateTruncate)
		require.NoError(t, err)
		require.NotNil(t, fw)
		err = fw.Close()
		require.NoError(t, err)
	})

	// Test 2: File creation with single root attribute
	t.Run("single_attribute", func(t *testing.T) {
		fw, err := CreateForWrite(tmpfile, CreateTruncate,
			WithRootAttribute("Conventions", "SOFA"))
		require.NoError(t, err)
		require.NotNil(t, fw)
		err = fw.Close()
		require.NoError(t, err)

		// Note: Verification of actual attribute storage will be added in Phase 3.2
	})

	// Test 3: File creation with multiple root attributes
	t.Run("multiple_attributes", func(t *testing.T) {
		fw, err := CreateForWrite(tmpfile, CreateTruncate,
			WithRootAttribute("Conventions", "SOFA"),
			WithRootAttribute("Version", "1.0"),
			WithRootAttribute("DataType", "FIR"))
		require.NoError(t, err)
		require.NotNil(t, fw)
		err = fw.Close()
		require.NoError(t, err)
	})

	// Test 4: Mix root attributes with other options
	t.Run("mixed_options", func(t *testing.T) {
		fw, err := CreateForWrite(tmpfile, CreateTruncate,
			WithSuperblockVersion(SuperblockV2),
			WithRootAttribute("Conventions", "SOFA"),
			WithBTreeRebalancing(true),
			WithRootAttribute("Version", "1.0"))
		require.NoError(t, err)
		require.NotNil(t, fw)
		err = fw.Close()
		require.NoError(t, err)
	})

	// Test 5: Verify various data types are accepted
	t.Run("various_datatypes", func(t *testing.T) {
		fw, err := CreateForWrite(tmpfile, CreateTruncate,
			WithRootAttribute("string_attr", "test"),
			WithRootAttribute("int32_attr", int32(42)),
			WithRootAttribute("float64_attr", 3.14159),
			WithRootAttribute("int_slice", []int32{1, 2, 3}))
		require.NoError(t, err)
		require.NotNil(t, fw)
		err = fw.Close()
		require.NoError(t, err)
	})
}

// TestWithRootAttribute_Config verifies that WithRootAttribute correctly populates FileWriteConfig.
func TestWithRootAttribute_Config(t *testing.T) {
	cfg := &FileWriteConfig{}

	// Apply WithRootAttribute options
	opt1 := WithRootAttribute("key1", "value1")
	opt2 := WithRootAttribute("key2", int32(42))
	opt3 := WithRootAttribute("key3", 3.14)

	opt1(cfg)
	opt2(cfg)
	opt3(cfg)

	// Verify the RootAttributes map is populated correctly
	require.NotNil(t, cfg.RootAttributes)
	assert.Equal(t, 3, len(cfg.RootAttributes))
	assert.Equal(t, "value1", cfg.RootAttributes["key1"])
	assert.Equal(t, int32(42), cfg.RootAttributes["key2"])
	assert.Equal(t, 3.14, cfg.RootAttributes["key3"])
}

// TestWithRootAttribute_EmptyName verifies behavior with edge cases.
func TestWithRootAttribute_EdgeCases(t *testing.T) {
	t.Run("empty_name", func(t *testing.T) {
		cfg := &FileWriteConfig{}
		opt := WithRootAttribute("", "value")
		opt(cfg)

		// API allows empty names - validation will happen in Phase 3.4
		require.NotNil(t, cfg.RootAttributes)
		assert.Equal(t, "value", cfg.RootAttributes[""])
	})

	t.Run("nil_value", func(t *testing.T) {
		cfg := &FileWriteConfig{}
		opt := WithRootAttribute("test", nil)
		opt(cfg)

		// API allows nil - validation will happen in Phase 3.4
		require.NotNil(t, cfg.RootAttributes)
		assert.Nil(t, cfg.RootAttributes["test"])
	})

	t.Run("overwrite_attribute", func(t *testing.T) {
		cfg := &FileWriteConfig{}
		opt1 := WithRootAttribute("key", "value1")
		opt2 := WithRootAttribute("key", "value2")

		opt1(cfg)
		opt2(cfg)

		// Later value overwrites earlier value
		require.NotNil(t, cfg.RootAttributes)
		assert.Equal(t, "value2", cfg.RootAttributes["key"])
	})
}
