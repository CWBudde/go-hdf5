package writer

import (
	"errors"
	"io"
	"math"
)

// maxMemoryFileSize bounds the size of an in-memory file so offsets from
// corrupt metadata cannot trigger huge allocations.
const maxMemoryFileSize = math.MaxInt32 * 4 // 8 GiB

// MemoryBuffer is a growable in-memory file implementing io.ReaderAt and
// io.WriterAt. Writes past the end extend the buffer, zero-filling any gap.
// The zero value is an empty buffer ready for use.
type MemoryBuffer struct {
	buf []byte
}

// ReadAt implements io.ReaderAt.
func (m *MemoryBuffer) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("negative offset")
	}
	if off >= int64(len(m.buf)) {
		return 0, io.EOF
	}
	n := copy(p, m.buf[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

// WriteAt implements io.WriterAt.
func (m *MemoryBuffer) WriteAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("negative offset")
	}
	end := off + int64(len(p))
	if end < off || end > maxMemoryFileSize {
		return 0, errors.New("in-memory file size limit exceeded")
	}
	m.grow(end)
	return copy(m.buf[off:], p), nil
}

// Truncate changes the buffer size, zero-filling any extension.
func (m *MemoryBuffer) Truncate(size int64) error {
	if size < 0 || size > maxMemoryFileSize {
		return errors.New("invalid in-memory file size")
	}
	if size <= int64(len(m.buf)) {
		m.buf = m.buf[:size]
		return nil
	}
	m.grow(size)
	return nil
}

// Size returns the buffer length; it lets bounds checks treat the buffer
// like a file of known size.
func (m *MemoryBuffer) Size() int64 {
	return int64(len(m.buf))
}

// Bytes returns the buffer contents. The slice aliases the buffer.
func (m *MemoryBuffer) Bytes() []byte {
	return m.buf
}

func (m *MemoryBuffer) grow(size int64) {
	if size <= int64(len(m.buf)) {
		return
	}
	if size <= int64(cap(m.buf)) {
		old := len(m.buf)
		m.buf = m.buf[:size]
		clear(m.buf[old:])
		return
	}
	newCap := max(size, 2*int64(cap(m.buf)), 4096)
	nb := make([]byte, size, newCap)
	copy(nb, m.buf)
	m.buf = nb
}
