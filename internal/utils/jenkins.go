// Package utils provides utility functions for HDF5 format operations.
package utils

// JenkinsLookup3 computes the Jenkins lookup3 hash used by the HDF5 library
// for all metadata checksums (superblock v2+, object headers v2, B-tree v2,
// fractal heap blocks, etc.).
//
// This is a Go port of H5_checksum_lookup3 from the HDF5 C library (H5checksum.c),
// which itself is based on Bob Jenkins' lookup3 hashlittle function.
//
// Reference: http://burtleburtle.net/bob/c/lookup3.c
func JenkinsLookup3(data []byte, initval uint32) uint32 {
	length := len(data)
	if length == 0 {
		return 0
	}

	a := uint32(0xdeadbeef) + uint32(length) + initval
	b := a
	c := a

	k := data

	// Process all but the last block (12 bytes at a time).
	for len(k) > 12 {
		a += uint32(k[0]) | uint32(k[1])<<8 | uint32(k[2])<<16 | uint32(k[3])<<24
		b += uint32(k[4]) | uint32(k[5])<<8 | uint32(k[6])<<16 | uint32(k[7])<<24
		c += uint32(k[8]) | uint32(k[9])<<8 | uint32(k[10])<<16 | uint32(k[11])<<24

		// mix
		a -= c
		a ^= (c << 4) | (c >> 28)
		c += b
		b -= a
		b ^= (a << 6) | (a >> 26)
		a += c
		c -= b
		c ^= (b << 8) | (b >> 24)
		b += a
		a -= c
		a ^= (c << 16) | (c >> 16)
		c += b
		b -= a
		b ^= (a << 19) | (a >> 13)
		a += c
		c -= b
		c ^= (b << 4) | (b >> 28)
		b += a

		k = k[12:]
	}

	// Handle the last (possibly partial) block.
	switch len(k) {
	case 12:
		c += uint32(k[11]) << 24
		fallthrough
	case 11:
		c += uint32(k[10]) << 16
		fallthrough
	case 10:
		c += uint32(k[9]) << 8
		fallthrough
	case 9:
		c += uint32(k[8])
		fallthrough
	case 8:
		b += uint32(k[7]) << 24
		fallthrough
	case 7:
		b += uint32(k[6]) << 16
		fallthrough
	case 6:
		b += uint32(k[5]) << 8
		fallthrough
	case 5:
		b += uint32(k[4])
		fallthrough
	case 4:
		a += uint32(k[3]) << 24
		fallthrough
	case 3:
		a += uint32(k[2]) << 16
		fallthrough
	case 2:
		a += uint32(k[1]) << 8
		fallthrough
	case 1:
		a += uint32(k[0])
	case 0:
		return c
	}

	// final mix
	c ^= b
	c -= (b << 14) | (b >> 18)
	a ^= c
	a -= (c << 11) | (c >> 21)
	b ^= a
	b -= (a << 25) | (a >> 7)
	c ^= b
	c -= (b << 16) | (b >> 16)
	a ^= c
	a -= (c << 4) | (c >> 28)
	b ^= a
	b -= (a << 14) | (a >> 18)
	c ^= b
	c -= (b << 24) | (b >> 8)

	return c
}

// JenkinsChecksum computes the HDF5 metadata checksum with initval=0.
func JenkinsChecksum(data []byte) uint32 {
	return JenkinsLookup3(data, 0)
}
