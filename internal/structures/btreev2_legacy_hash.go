package structures

// legacyJenkinsHash is the name hash go-hdf5 up to v0.16.0 wrote into
// B-tree v2 name indexes. It differs from jenkinsHash only for names whose
// length is a multiple of 12: it mixed the last full 12-byte block like an
// inner one and then applied lookup3's final mix on top.
//
// It is only used to find records in files written by those versions; see
// WritableBTreeV2.findRecord.
func legacyJenkinsHash(name string) uint32 {
	n := len(name)
	a := uint32(0xdeadbeef) + uint32(n) //nolint:gosec // G115: lookup3 seeds with the length
	b, c := a, a

	word := func(i int) uint32 {
		return uint32(name[i]) | uint32(name[i+1])<<8 | uint32(name[i+2])<<16 | uint32(name[i+3])<<24
	}
	rot := func(x uint32, k uint) uint32 { return x<<k | x>>(32-k) }

	// Every full block, including the last one, went through mix().
	i := 0
	for ; i+12 <= n; i += 12 {
		a += word(i)
		b += word(i + 4)
		c += word(i + 8)
		a -= c
		a ^= rot(c, 4)
		c += b
		b -= a
		b ^= rot(a, 6)
		a += c
		c -= b
		c ^= rot(b, 8)
		b += a
		a -= c
		a ^= rot(c, 16)
		c += b
		b -= a
		b ^= rot(a, 19)
		a += c
		c -= b
		c ^= rot(b, 4)
		b += a
	}

	// Tail bytes, little-endian into a, b, c.
	for k := range n - i {
		v := uint32(name[i+k]) << (uint(k%4) * 8)
		switch k / 4 {
		case 0:
			a += v
		case 1:
			b += v
		default:
			c += v
		}
	}

	// final(), applied even when no tail bytes were left.
	c ^= b
	c -= rot(b, 14)
	a ^= c
	a -= rot(c, 11)
	b ^= a
	b -= rot(a, 25)
	c ^= b
	c -= rot(b, 16)
	a ^= c
	a -= rot(c, 4)
	b ^= a
	b -= rot(a, 14)
	c ^= b
	c -= rot(b, 24)
	return c
}
