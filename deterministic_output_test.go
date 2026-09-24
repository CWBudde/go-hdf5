package hdf5

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestDeterministicOutput verifies that identical write calls produce
// byte-identical files. Root attributes, dataset attributes and dense group
// links are held in maps internally; map iteration order must not leak into
// the file layout.
func TestDeterministicOutput(t *testing.T) {
	build := func(path string, rootAttrs int) {
		opts := make([]interface{}, 0, rootAttrs+1)
		opts = append(opts, WithRootAttribute("Conventions", "SOFA"))
		for i := 0; i < rootAttrs; i++ {
			opts = append(opts, WithRootAttribute(fmt.Sprintf("attr_%c", 'z'-i), int32(i)))
		}
		fw, err := CreateForWrite(path, CreateTruncate, opts...)
		require.NoError(t, err)

		for i := 0; i < 10; i++ {
			ds, err := fw.CreateDataset(fmt.Sprintf("/ds%d", i), Float64, []uint64{3},
				WithAttribute("units", "m"),
				WithAttribute("long_name", "value"),
				WithAttribute("index", int32(i)))
			require.NoError(t, err)
			require.NoError(t, ds.Write([]float64{1, 2, 3}))
		}

		links := map[string]string{}
		for i := 0; i < 10; i++ {
			links[fmt.Sprintf("link%d", i)] = fmt.Sprintf("/ds%d", i)
		}
		require.NoError(t, fw.CreateDenseGroup("/dense", links))
		require.NoError(t, fw.Close())
	}

	for _, rootAttrs := range []int{5, 20} { // compact and dense root attributes
		t.Run(fmt.Sprintf("root_attrs_%d", rootAttrs), func(t *testing.T) {
			dir := t.TempDir()
			ref := filepath.Join(dir, "ref.h5")
			build(ref, rootAttrs)
			want, err := os.ReadFile(ref)
			require.NoError(t, err)

			for i := 0; i < 5; i++ {
				p := filepath.Join(dir, fmt.Sprintf("run%d.h5", i))
				build(p, rootAttrs)
				got, err := os.ReadFile(p)
				require.NoError(t, err)
				require.Equal(t, want, got, "run %d produced a different file", i)
			}

			if rootAttrs > MaxCompactAttributes {
				return // dense storage is ordered by the name-hash index
			}

			// Compact root attributes are written in WithRootAttribute call order.
			f, err := Open(ref)
			require.NoError(t, err)
			defer func() { _ = f.Close() }()
			attrs, err := f.Root().Attributes()
			require.NoError(t, err)
			require.NotEmpty(t, attrs)
			require.Equal(t, "Conventions", attrs[0].Name)
		})
	}
}
