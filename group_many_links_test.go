package hdf5

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestManyLinksRoundTrip writes groups whose local heap and symbol table
// B-tree must grow (a fixed 256-byte heap used to fail after ~26 links, the
// single B-tree leaf after 256) and reads all links back.
func TestManyLinksRoundTrip(t *testing.T) {
	for _, n := range []int{30, 100, 257, 1000} {
		path := filepath.Join(t.TempDir(), "links.h5")
		writeManyLinks(t, path, n)

		f, err := Open(path)
		require.NoError(t, err)
		root := childNames(f.Root())
		require.Len(t, root, n+1, "n=%d", n)
		for _, c := range f.Root().Children() {
			if c.Name() != "sub" {
				continue
			}
			require.Len(t, c.(*Group).Children(), (n+1)/2, "n=%d", n)
		}
		for _, c := range f.Root().Children() {
			if c.Name() == manyLinkName(n-1) {
				v, err := c.(*Dataset).Read()
				require.NoError(t, err)
				require.Equal(t, []float64{float64(n - 1)}, v)
			}
		}
		require.NoError(t, f.Close())
	}
}

// TestManyLinksDeterministic checks that growing heaps and B-trees keep the
// output byte-for-byte reproducible.
func TestManyLinksDeterministic(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.h5"), filepath.Join(dir, "b.h5")
	writeManyLinks(t, a, 300)
	writeManyLinks(t, b, 300)
	da, err := os.ReadFile(a)
	require.NoError(t, err)
	db, err := os.ReadFile(b)
	require.NoError(t, err)
	require.Equal(t, da, db)
}
