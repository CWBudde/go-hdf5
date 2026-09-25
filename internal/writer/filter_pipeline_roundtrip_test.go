package writer

import (
	"testing"

	"github.com/cwbudde/go-hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// TestPipelineMessageRoundTrip encodes version 2 pipeline messages and
// parses them back: filters without client data (Fletcher32), predefined
// filters (ID < 256, no name) and filters with ID >= 256 (name present, not
// padded) in every position.
func TestPipelineMessageRoundTrip(t *testing.T) {
	pipelines := map[string][]Filter{
		"fletcher_gzip":      {NewFletcher32Filter(), NewGZIPFilter(6)},
		"gzip_fletcher":      {NewGZIPFilter(6), NewFletcher32Filter()},
		"lzf_first":          {NewLZFFilter(), NewShuffleFilter(8), NewFletcher32Filter()},
		"all":                {NewShuffleFilter(4), NewBZIP2Filter(9), NewFletcher32Filter(), NewLZFFilter(), NewGZIPFilter(1)},
		"fletcher_only":      {NewFletcher32Filter()},
		"bzip2_then_deflate": {NewBZIP2Filter(9), NewGZIPFilter(9)},
	}
	for name, filters := range pipelines {
		t.Run(name, func(t *testing.T) {
			fp := NewFilterPipeline()
			for _, f := range filters {
				fp.AddFilter(f)
			}
			msg, err := fp.EncodePipelineMessage()
			require.NoError(t, err)

			parsed, err := core.ParseFilterPipelineMessage(msg)
			require.NoError(t, err)
			require.Equal(t, uint8(2), parsed.Version)
			require.Len(t, parsed.Filters, len(filters))
			for i, f := range filters {
				flags, cd := f.Encode()
				got := parsed.Filters[i]
				require.Equal(t, core.FilterID(f.ID()), got.ID, "filter %d", i)
				require.Equal(t, flags, got.Flags, "filter %d", i)
				if len(cd) == 0 {
					require.Empty(t, got.ClientData, "filter %d", i)
				} else {
					require.Equal(t, cd, got.ClientData, "filter %d", i)
				}
				if f.ID() >= 256 {
					require.Equal(t, f.Name(), got.Name, "filter %d", i)
				} else {
					require.Empty(t, got.Name, "filter %d", i)
				}
			}
		})
	}
}
