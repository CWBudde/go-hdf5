package core

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/cwbudde/go-hdf5/internal/utils"
)

// ObjectHeaderWriter provides functionality for writing HDF5 object headers.
// Supports both v1 (legacy, for superblock v0) and v2 (modern) formats.
type ObjectHeaderWriter struct {
	Version  uint8
	Flags    uint8
	Messages []MessageWriter

	// V1-specific fields (used only when Version == 1)
	RefCount uint32 // Reference count (always 1 for new files)
}

// MessageWriter represents a message that can be written to an object header.
type MessageWriter struct {
	Type MessageType
	Data []byte
}

// NewMinimalRootGroupHeader creates a minimal object header v2 for an empty root group.
// This is suitable for MVP file creation - just enough to make a valid HDF5 file.
//
// The root group header contains:
//   - Object Header v2 with minimal flags (no times, no attribute phase change)
//   - Link Info message (empty, compact storage)
//
// Returns an ObjectHeaderWriter ready to be written to file.
func NewMinimalRootGroupHeader() *ObjectHeaderWriter {
	// Create minimal Link Info message for an empty group
	// Link Info message format (compact storage, no dense links):
	//   Version: 0 (1 byte)
	//   Flags: 0x00 (1 byte) - compact storage, no index
	//   Max Compact (optional): 2 bytes if flags & 0x01
	//   Min Dense (optional): 2 bytes if flags & 0x01
	//   Heap Address (8 bytes): 0xFFFFFFFFFFFFFFFF (UNDEF for compact)
	//   B-tree Address (8 bytes): 0xFFFFFFFFFFFFFFFF (UNDEF for compact)
	//
	// For MVP empty group: Version=0, Flags=0, no optional fields, two UNDEF addresses
	linkInfoData := make([]byte, 18) // 1+1+8+8 = 18 bytes
	linkInfoData[0] = 0              // Version 0
	linkInfoData[1] = 0              // Flags: compact storage, no tracking

	// Heap address (UNDEF for compact storage)
	binary.LittleEndian.PutUint64(linkInfoData[2:10], 0xFFFFFFFFFFFFFFFF)

	// B-tree name index address (UNDEF for compact storage)
	binary.LittleEndian.PutUint64(linkInfoData[10:18], 0xFFFFFFFFFFFFFFFF)

	return &ObjectHeaderWriter{
		Version: 2,
		Flags:   0, // Minimal flags: no times, no attribute phase change
		Messages: []MessageWriter{
			{
				Type: MsgLinkInfo,
				Data: linkInfoData,
			},
		},
	}
}

// ObjectHeaderV2ChecksumSize is the size of the checksum that terminates every
// version 2 object header chunk (H5O_SIZEOF_CHKSUM).
const ObjectHeaderV2ChecksumSize = 4

// objectHeaderV2Signature is the signature of version 2 object headers.
const objectHeaderV2Signature = "OHDR"

// Size calculates the total size of the object header in bytes.
// This is used for pre-allocation before writing.
//
// Returns:
//   - Total size in bytes
//
// For object header v1:
//   - Header: 16 bytes (version, reserved, num_messages, ref_count, header_size, padding)
//   - Messages: sum of (2 + 2 + 1 + 3 + len(data)) for each message (8-byte aligned)
//
// For object header v2:
//   - Header: 4 (signature) + 1 (version) + 1 (flags) + 1/2/4 (chunk size)
//   - Messages: sum of (1 + 2 + 1 + len(data)) for each message
//   - Checksum: 4 bytes
func (ohw *ObjectHeaderWriter) Size() uint64 {
	switch ohw.Version {
	case 1:
		return ohw.sizeV1()
	case 2:
		return ohw.sizeV2()
	default:
		// Should never happen - validated at creation time
		panic(fmt.Sprintf("unsupported object header version: %d", ohw.Version))
	}
}

// sizeV1 calculates size for object header v1.
// V1 format:
//   - 16-byte header
//   - Message headers (8 bytes each)
//   - Message data (variable, 8-byte aligned)
//
// IMPORTANT: The "Object Header Size" field in v1 includes ONLY:
//   - The 16-byte header
//   - All message headers (8 bytes each)
//   - It does NOT include message data!
//
// This function returns the TOTAL size (header + message headers + message data)
// for allocation purposes, but writeToV1() calculates the "Object Header Size"
// field separately.
func (ohw *ObjectHeaderWriter) sizeV1() uint64 {
	headerSize := uint64(16) // V1 header is always 16 bytes

	// Calculate total message size with 8-byte alignment
	var totalMessageSize uint64
	for _, msg := range ohw.Messages {
		// Each v1 message:
		// - Header: Type (2) + Size (2) + Flags (1) + Reserved (3) = 8 bytes
		// - Data: variable
		// - Total aligned to 8-byte boundary
		msgSize := 8 + uint64(len(msg.Data))
		// Align to 8-byte boundary
		if msgSize%8 != 0 {
			msgSize += 8 - (msgSize % 8)
		}
		totalMessageSize += msgSize
	}

	// Return total size (header + all messages including data)
	return headerSize + totalMessageSize
}

// sizeV2 calculates size for object header v2 (current implementation).
func (ohw *ObjectHeaderWriter) sizeV2() uint64 {
	// Calculate message data size
	var messageDataSize uint64
	for _, msg := range ohw.Messages {
		// Each message: Type (1) + Size (2) + Flags (1) + Data (variable)
		messageDataSize += 1 + 2 + 1 + uint64(len(msg.Data))
	}

	// Determine chunk size field width based on message size
	// Flags bits 0-1 encode chunk size width: 00=1byte, 01=2byte, 10=4byte
	var chunkSizeBytes uint64
	if messageDataSize <= 255 {
		chunkSizeBytes = 1
	} else if messageDataSize <= 65535 {
		chunkSizeBytes = 2
	} else {
		chunkSizeBytes = 4
	}

	// Signature (4) + Version (1) + Flags (1) + Chunk Size (1/2/4) + Messages + Checksum (4).
	// The "Size of Chunk #0" field covers only the messages; the trailing
	// Jenkins lookup3 checksum is NOT part of it but is part of the on-disk header.
	return 4 + 1 + 1 + chunkSizeBytes + messageDataSize + ObjectHeaderV2ChecksumSize
}

// WriteTo writes the object header to the writer at the specified address.
// Returns the total size written (useful for allocation tracking).
//
// Object Header v1 format:
//   - Version (1 byte)
//   - Reserved (1 byte)
//   - Number of Messages (2 bytes)
//   - Object Reference Count (4 bytes)
//   - Object Header Size (4 bytes)
//   - Padding to 8-byte alignment (4 bytes)
//   - Messages (each 8-byte aligned)
//
// Object Header v2 format:
//   - Signature: "OHDR" (4 bytes)
//   - Version: 2 (1 byte)
//   - Flags: (1 byte)
//   - [Optional fields based on flags]
//   - Size of Chunk 0: (1, 2, 4, or 8 bytes based on flags bits 0-1)
//   - Messages: variable size
//
// For MVP v2:
//   - No timestamp fields (flags bit 5 = 0)
//   - No attribute phase change (flags bit 4 = 0)
//   - Chunk size in 1 byte (flags bits 0-1 = 0)
func (ohw *ObjectHeaderWriter) WriteTo(w io.WriterAt, address uint64) (uint64, error) {
	switch ohw.Version {
	case 1:
		return ohw.writeToV1(w, address)
	case 2:
		return ohw.writeToV2(w, address)
	default:
		return 0, fmt.Errorf("unsupported object header version: %d", ohw.Version)
	}
}

// writeToV1 writes an object header v1 to the writer.
// V1 format (HDF5 spec III.A.1):
//   - Version (1 byte) = 1
//   - Reserved (1 byte) = 0
//   - Number of Messages (2 bytes, little-endian)
//   - Object Reference Count (4 bytes, little-endian)
//   - Object Header Size (4 bytes, little-endian) - total size including header
//   - Padding to 8-byte alignment (4 bytes of zeros)
//   - Messages (each 8-byte aligned):
//   - Type (2 bytes, little-endian)
//   - Size (2 bytes, little-endian)
//   - Flags (1 byte)
//   - Reserved (3 bytes)
//   - Data (variable, padded to 8-byte boundary)
func (ohw *ObjectHeaderWriter) writeToV1(w io.WriterAt, address uint64) (uint64, error) {
	// Calculate total size for buffer allocation
	totalSize := ohw.sizeV1()
	buf := make([]byte, totalSize)

	// "Object Header Size" field: number of bytes of message data (message
	// headers + 8-byte aligned message bodies) that follow the 16-byte prefix
	// (H5O__cache_deserialize: chunk0_size).
	objectHeaderSize := uint32(totalSize - 16) //nolint:gosec // G115: Safe - header size limited by HDF5 spec

	offset := 0

	// Header (16 bytes)
	buf[offset] = 1 // Version
	offset++

	buf[offset] = 0 // Reserved
	offset++

	// Number of messages (2 bytes)
	binary.LittleEndian.PutUint16(buf[offset:offset+2], uint16(len(ohw.Messages))) //nolint:gosec // G115: Safe - message count limited by HDF5 spec
	offset += 2

	// Object reference count (4 bytes) - always 1 for new files
	binary.LittleEndian.PutUint32(buf[offset:offset+4], ohw.RefCount)
	offset += 4

	// Object header size (4 bytes) - header + message headers ONLY (no message data!)
	// For 1 message: 16 (header) + 8 (message header) = 24 bytes
	binary.LittleEndian.PutUint32(buf[offset:offset+4], objectHeaderSize)
	offset += 4

	// Padding to 8-byte alignment (4 bytes of zeros)
	// Already zero from make(), just advance offset
	offset += 4

	// Write messages
	for _, msg := range ohw.Messages {
		// Message type (2 bytes, little-endian)
		binary.LittleEndian.PutUint16(buf[offset:offset+2], uint16(msg.Type))
		offset += 2

		// Message data size (2 bytes, little-endian). In version 1 headers the
		// size includes the padding to a multiple of 8 bytes; the C library
		// rejects unaligned sizes ("message not aligned").
		alignedSize := (len(msg.Data) + 7) &^ 7
		binary.LittleEndian.PutUint16(buf[offset:offset+2], uint16(alignedSize)) //nolint:gosec // G115: Safe - message size validated
		offset += 2

		// Message flags (1 byte)
		buf[offset] = 0 // For MVP: no flags
		offset++

		// Reserved (3 bytes) - already zero from make()
		offset += 3

		// Message data
		copy(buf[offset:offset+len(msg.Data)], msg.Data)
		offset += len(msg.Data)

		// Pad to 8-byte boundary
		msgSize := 8 + len(msg.Data) // Header (8 bytes) + Data
		if msgSize%8 != 0 {
			padding := 8 - (msgSize % 8)
			// Padding bytes already zero from make()
			offset += padding
		}
	}

	// Write to file
	n, err := w.WriteAt(buf, int64(address)) //nolint:gosec // Safe: address within file bounds
	if err != nil {
		return 0, fmt.Errorf("failed to write object header v1 at address %d: %w", address, err)
	}

	if n != len(buf) {
		return 0, fmt.Errorf("incomplete object header v1 write: wrote %d bytes, expected %d", n, len(buf))
	}

	return totalSize, nil
}

// writeToV2 writes an object header v2 to the writer.
// V2 format (current MVP implementation).
func (ohw *ObjectHeaderWriter) writeToV2(w io.WriterAt, address uint64) (uint64, error) {
	// Calculate message data size
	var messageDataSize uint64
	for _, msg := range ohw.Messages {
		// Each message has:
		// - Type (1 byte for v2)
		// - Size (2 bytes for v2)
		// - Flags (1 byte for v2)
		// - Data (variable)
		messageDataSize += 1 + 2 + 1 + uint64(len(msg.Data))
	}

	// Calculate total chunk size
	// Chunk contains all messages
	chunkSize := messageDataSize

	// Determine chunk size encoding based on size
	// Flags bits 0-1 encode chunk size width: 00=1byte, 01=2byte, 10=4byte, 11=8byte
	var flags uint8
	var chunkSizeBytes int
	if chunkSize <= 255 {
		flags = 0 // 1-byte encoding
		chunkSizeBytes = 1
	} else if chunkSize <= 65535 {
		flags = 1 // 2-byte encoding
		chunkSizeBytes = 2
	} else if chunkSize <= 4294967295 {
		flags = 2 // 4-byte encoding
		chunkSizeBytes = 4
	} else {
		return 0, fmt.Errorf("chunk size %d too large (max 4GB)", chunkSize)
	}

	// Build header
	// Signature (4) + Version (1) + Flags (1) + Chunk Size (1/2/4) + Messages (variable) + Checksum (4).
	// "Size of Chunk #0" excludes the checksum (H5O_SIZEOF_CHKSUM), but the checksum
	// is part of the header on disk and MUST be present: the HDF5 C library reads
	// prefix + chunk0 size + 4 bytes and verifies the checksum.
	headerSize := 4 + 1 + 1 + uint64(chunkSizeBytes) + chunkSize + ObjectHeaderV2ChecksumSize
	buf := make([]byte, headerSize)

	offset := 0

	// Write signature "OHDR" (4 bytes, little-endian format).
	copy(buf[offset:offset+4], objectHeaderV2Signature)
	offset += 4

	// Version
	buf[offset] = ohw.Version
	offset++

	// Flags (merge with caller's flags, preserving other bits)
	buf[offset] = ohw.Flags | flags
	offset++

	// Chunk 0 size (1, 2, or 4 bytes depending on flags)
	switch chunkSizeBytes {
	case 1:
		buf[offset] = uint8(chunkSize)
		offset++
	case 2:
		binary.LittleEndian.PutUint16(buf[offset:offset+2], uint16(chunkSize))
		offset += 2
	case 4:
		binary.LittleEndian.PutUint32(buf[offset:offset+4], uint32(chunkSize))
		offset += 4
	}

	// Write messages
	for _, msg := range ohw.Messages {
		// Message type (1 byte for v2)
		buf[offset] = uint8(msg.Type) //nolint:gosec // Safe: message type is limited enum
		offset++

		// Message data size (2 bytes, little-endian)
		binary.LittleEndian.PutUint16(buf[offset:offset+2], uint16(len(msg.Data))) //nolint:gosec // Safe: message size validated
		offset += 2

		// Message flags (1 byte)
		// For MVP: flags = 0 (not shared, not constant, not shareable)
		buf[offset] = 0
		offset++

		// Message data
		copy(buf[offset:offset+len(msg.Data)], msg.Data)
		offset += len(msg.Data)
	}

	// Checksum (Jenkins lookup3) over everything from the signature up to here.
	binary.LittleEndian.PutUint32(buf[offset:offset+ObjectHeaderV2ChecksumSize], utils.JenkinsChecksum(buf[:offset]))

	// Write to file
	n, err := w.WriteAt(buf, int64(address)) //nolint:gosec // Safe: address within file bounds
	if err != nil {
		return 0, fmt.Errorf("failed to write object header v2 at address %d: %w", address, err)
	}

	if n != len(buf) {
		return 0, fmt.Errorf("incomplete object header v2 write: wrote %d bytes, expected %d", n, len(buf))
	}

	return headerSize, nil
}

// maxHeaderMessageSize is the largest message body an object header message
// can hold (2-byte size field).
const maxHeaderMessageSize = 0xFFFF

// AddMessageToObjectHeader adds a message to an object header.
// Only object header v2 is supported. The header may grow beyond its
// allocated chunk #0; WriteObjectHeader spills into a continuation chunk.
//
// Parameters:
//   - oh: Object header to modify
//   - msgType: Message type (e.g., MsgAttribute = 0x000C)
//   - msgData: Encoded message bytes
//
// Returns:
//   - error: Non-nil if header full or add fails
//
// Limitations:
//   - Only object header v2 supported
//   - No message flags (always 0)
//
// Reference: H5O.c - H5O_msg_append().
func AddMessageToObjectHeader(oh *ObjectHeader, msgType MessageType, msgData []byte) error {
	if oh == nil {
		return fmt.Errorf("object header is nil")
	}

	if oh.Version != 2 {
		return fmt.Errorf("only object header version 2 is supported for modification, got version %d", oh.Version)
	}

	// Header messages carry a 2-byte size field. Like libhdf5
	// (H5O_MESG_MAX_SIZE), larger messages cannot live in the header; the
	// caller moves attributes to dense storage instead. There is no limit on
	// the total size: WriteObjectHeader moves messages that do not fit into
	// chunk #0 into a continuation chunk.
	if len(msgData) > maxHeaderMessageSize {
		return fmt.Errorf("object header full: message of %d bytes exceeds the %d byte header message limit",
			len(msgData), maxHeaderMessageSize)
	}

	// Create new message
	newMessage := &HeaderMessage{
		Type:   msgType,
		Offset: 0, // Will be calculated during write
		Data:   make([]byte, len(msgData)),
	}
	copy(newMessage.Data, msgData)

	// Add to messages list
	oh.Messages = append(oh.Messages, newMessage)

	return nil
}

// WriteObjectHeader writes an object header back to disk at a given address.
// This is used when modifying object headers (e.g., adding attributes).
//
// For MVP (v0.11.1-beta):
//   - Only object header v2 supported
//   - No continuation blocks
//   - Overwrites existing header at the same address
//
// Parameters:
//   - w: Writer with WriteAt capability
//   - addr: File address where header is located
//   - oh: Object header to write
//   - sb: Superblock for encoding parameters
//
// Returns:
//   - error: Non-nil if write fails
//
// Reference: H5O.c - H5O_flush().
func WriteObjectHeader(w io.WriterAt, addr uint64, oh *ObjectHeader, sb *Superblock) error {
	_ = sb // Reserved for future use (v1 headers or encoding parameters)

	if oh == nil {
		return fmt.Errorf("object header is nil")
	}

	if oh.Version != 2 {
		return fmt.Errorf("only object header version 2 is supported for writing, got version %d", oh.Version)
	}

	// When the destination can be read back, rewrite the header inside the
	// space it already occupies (spilling into a continuation chunk if
	// needed). Writing a larger header at the same address would overwrite
	// whatever object was allocated right after it.
	if rw, ok := w.(ReaderWriterAt); ok {
		if capacity, _, err := readV2Chunk0Layout(rw, addr); err == nil && capacity > 0 {
			alloc, _ := w.(SpaceAllocator)
			return rewriteObjectHeaderV2InPlace(rw, alloc, addr, oh, sb)
		}
	}

	// Build object header writer from the object header
	ohw := &ObjectHeaderWriter{
		Version:  oh.Version,
		Flags:    oh.Flags &^ 0x03, // chunk size width is recomputed by WriteTo
		Messages: make([]MessageWriter, len(oh.Messages)),
	}

	// Convert messages
	for i, msg := range oh.Messages {
		ohw.Messages[i] = MessageWriter{
			Type: msg.Type,
			Data: msg.Data,
		}
	}

	// Write the header
	_, err := ohw.WriteTo(w, addr)
	if err != nil {
		return fmt.Errorf("failed to write object header at address %d: %w", addr, err)
	}

	return nil
}

// RewriteObjectHeaderV2 rewrites an object header v2 with updated messages.
// This handles the case where we need to modify an existing object header
// by reading it, modifying it, and writing it back.
//
// For MVP (v0.11.1-beta):
//   - Only supports v2 headers without continuation blocks
//   - Overwrites header at original location if size permits
//   - Returns error if new header doesn't fit in original space
//
// Parameters:
//   - w: Writer with WriteAt capability
//   - r: Reader for reading current header
//   - addr: File address of object header
//   - sb: Superblock
//   - newMessages: Additional messages to add
//
// Returns:
//   - error: Non-nil if operation fails
//
// Note: This is a simplified version for MVP. Full implementation would:
//   - Support continuation blocks
//   - Handle header relocation if needed
//   - Support v1 headers
func RewriteObjectHeaderV2(w io.WriterAt, r io.ReaderAt, addr uint64, sb *Superblock, newMessages []*HeaderMessage) error {
	// Read existing object header
	oh, err := ReadObjectHeader(r, addr, sb)
	if err != nil {
		return fmt.Errorf("failed to read object header: %w", err)
	}

	if oh.Version != 2 {
		return fmt.Errorf("only v2 headers supported for rewrite, got version %d", oh.Version)
	}

	// Add new messages
	for _, msg := range newMessages {
		err = AddMessageToObjectHeader(oh, msg.Type, msg.Data)
		if err != nil {
			return fmt.Errorf("failed to add message: %w", err)
		}
	}

	// Write back to same location
	err = WriteObjectHeader(w, addr, oh, sb)
	if err != nil {
		return fmt.Errorf("failed to write object header: %w", err)
	}

	return nil
}

// ReaderWriterAt combines io.ReaderAt and io.WriterAt.
type ReaderWriterAt interface {
	io.ReaderAt
	io.WriterAt
}

// RefreshObjectHeaderV2Checksum recomputes and rewrites the checksum of the
// first chunk of a version 2 object header located at addr.
//
// It must be called after any in-place patch of bytes inside an already
// written v2 object header (for example updating an address stored in a
// layout message), otherwise the HDF5 C library rejects the header with a
// checksum mismatch.
func RefreshObjectHeaderV2Checksum(rw ReaderWriterAt, addr uint64) error {
	prefix := make([]byte, 6)
	if _, err := rw.ReadAt(prefix, int64(addr)); err != nil { //nolint:gosec // Safe: address within file bounds
		return fmt.Errorf("failed to read object header prefix at %d: %w", addr, err)
	}
	if string(prefix[0:4]) != objectHeaderV2Signature || prefix[4] != 2 {
		return fmt.Errorf("no version 2 object header at address %d", addr)
	}
	flags := prefix[5]

	pos := uint64(6)
	if flags&0x20 != 0 { // times stored
		pos += 16
	}
	if flags&0x10 != 0 { // non-default attribute phase change values
		pos += 4
	}
	sizeBytes := uint64(1) << (flags & 0x03)
	sizeBuf := make([]byte, 8)
	if _, err := rw.ReadAt(sizeBuf[:sizeBytes], int64(addr+pos)); err != nil { //nolint:gosec // Safe: address within file bounds
		return fmt.Errorf("failed to read chunk size at %d: %w", addr+pos, err)
	}
	chunkSize := binary.LittleEndian.Uint64(sizeBuf)
	pos += sizeBytes

	const maxChunk = 1 << 32
	if chunkSize > maxChunk {
		return fmt.Errorf("object header chunk size %d too large", chunkSize)
	}

	buf := make([]byte, pos+chunkSize)
	if _, err := rw.ReadAt(buf, int64(addr)); err != nil { //nolint:gosec // Safe: address within file bounds
		return fmt.Errorf("failed to read object header at %d: %w", addr, err)
	}

	sum := make([]byte, ObjectHeaderV2ChecksumSize)
	binary.LittleEndian.PutUint32(sum, utils.JenkinsChecksum(buf))
	if _, err := rw.WriteAt(sum, int64(addr+pos+chunkSize)); err != nil { //nolint:gosec // Safe: address within file bounds
		return fmt.Errorf("failed to write object header checksum: %w", err)
	}
	return nil
}

// SpaceAllocator allocates file space (implemented by writer.FileWriter).
type SpaceAllocator interface {
	Allocate(size uint64) (uint64, error)
}

// readV2Chunk0Layout returns the chunk #0 size (excluding the checksum) and
// the prefix length (signature through the chunk-size field) of the v2 object
// header at addr.
func readV2Chunk0Layout(r io.ReaderAt, addr uint64) (chunkSize, prefixLen uint64, err error) {
	prefix := make([]byte, 6)
	if _, err := r.ReadAt(prefix, int64(addr)); err != nil { //nolint:gosec // Safe: address within file bounds
		return 0, 0, err
	}
	if string(prefix[0:4]) != objectHeaderV2Signature || prefix[4] != 2 {
		return 0, 0, fmt.Errorf("no version 2 object header at %d", addr)
	}
	flags := prefix[5]
	pos := uint64(6)
	if flags&0x20 != 0 {
		pos += 16
	}
	if flags&0x10 != 0 {
		pos += 4
	}
	width := uint64(1) << (flags & 0x03)
	sizeBuf := make([]byte, 8)
	if _, err := r.ReadAt(sizeBuf[:width], int64(addr+pos)); err != nil { //nolint:gosec // Safe: address within file bounds
		return 0, 0, err
	}
	return binary.LittleEndian.Uint64(sizeBuf), pos + width, nil
}

// encodeV2Messages encodes messages as v2 header messages (type, size, flags, data).
func encodeV2Messages(msgs []*HeaderMessage) ([]byte, error) {
	size := 0
	for _, m := range msgs {
		size += 4 + len(m.Data)
	}
	out := make([]byte, 0, size)
	for _, m := range msgs {
		if len(m.Data) > 0xFFFF {
			return nil, fmt.Errorf("message type %d too large (%d bytes)", m.Type, len(m.Data))
		}
		out = append(out, byte(m.Type))
		out = binary.LittleEndian.AppendUint16(out, uint16(len(m.Data))) //nolint:gosec // G115: checked above
		out = append(out, 0)
		out = append(out, m.Data...)
	}
	return out, nil
}

// appendV2Gap pads a v2 chunk with gap bytes, using NIL messages where possible.
func appendV2Gap(buf []byte, gap uint64) []byte {
	for gap >= 4 {
		n := gap - 4
		if n > 0xFFFF {
			n = 0xFFFF
		}
		buf = append(buf, byte(MsgNil))
		buf = binary.LittleEndian.AppendUint16(buf, uint16(n)) //nolint:gosec // G115: n <= 0xFFFF
		buf = append(buf, 0)
		buf = append(buf, make([]byte, n)...)
		gap -= 4 + n
	}
	// A gap smaller than a message header is allowed at the end of a chunk.
	return append(buf, make([]byte, gap)...)
}

// rewriteObjectHeaderV2InPlace rewrites the messages of the existing v2
// object header at addr without changing the size of chunk #0. Messages that
// do not fit are moved into a newly allocated continuation chunk ("OCHK").
func rewriteObjectHeaderV2InPlace(rw ReaderWriterAt, alloc SpaceAllocator, addr uint64, oh *ObjectHeader, sb *Superblock) error {
	capacity, prefixLen, err := readV2Chunk0Layout(rw, addr)
	if err != nil {
		return err
	}
	prefix := make([]byte, prefixLen)
	if _, err := rw.ReadAt(prefix, int64(addr)); err != nil { //nolint:gosec // Safe: address within file bounds
		return fmt.Errorf("failed to read object header prefix: %w", err)
	}
	if prefix[5]&0x04 != 0 {
		return fmt.Errorf("object header at %d tracks attribute creation order; rewriting not supported", addr)
	}

	// Drop NIL and continuation messages: layout is recomputed from scratch.
	// A single existing continuation chunk is reused when the spilled
	// messages still fit, so repeated attribute writes do not leak space.
	msgs := make([]*HeaderMessage, 0, len(oh.Messages))
	var reuse *continuationChunk
	conts := 0
	for _, m := range oh.Messages {
		if m.Type == MsgContinuation {
			conts++
			if c, ok := parseContinuation(m.Data, sb); ok {
				reuse = &c
			}
		}
		if m.Type == MsgNil || m.Type == MsgContinuation {
			continue
		}
		msgs = append(msgs, m)
	}
	if conts != 1 {
		reuse = nil
	}

	all, err := encodeV2Messages(msgs)
	if err != nil {
		return err
	}

	chunk := all
	if uint64(len(all)) > capacity {
		chunk, err = spillToContinuationChunk(rw, alloc, addr, capacity, msgs, sb, reuse)
		if err != nil {
			return err
		}
	}
	chunk = appendV2Gap(chunk, capacity-uint64(len(chunk)))

	buf := make([]byte, 0, uint64(len(prefix))+capacity+ObjectHeaderV2ChecksumSize)
	buf = append(buf, prefix...)
	buf = append(buf, chunk...)
	buf = binary.LittleEndian.AppendUint32(buf, utils.JenkinsChecksum(buf))
	if _, err := rw.WriteAt(buf, int64(addr)); err != nil { //nolint:gosec // Safe: address within file bounds
		return fmt.Errorf("failed to write object header at %d: %w", addr, err)
	}
	return nil
}

// continuationChunk is the location of an object header continuation chunk.
type continuationChunk struct {
	addr, length uint64
}

// parseContinuation decodes a continuation message (address, length).
func parseContinuation(data []byte, sb *Superblock) (continuationChunk, bool) {
	offsetSize, lengthSize := 8, 8
	if sb != nil && sb.OffsetSize != 0 {
		offsetSize, lengthSize = int(sb.OffsetSize), int(sb.LengthSize)
	}
	if len(data) < offsetSize+lengthSize {
		return continuationChunk{}, false
	}
	c := continuationChunk{
		addr:   readUintN(data[:offsetSize]),
		length: readUintN(data[offsetSize : offsetSize+lengthSize]),
	}
	return c, c.addr != 0 && c.length > 0
}

// readUintN decodes a little-endian unsigned integer of 1-8 bytes.
func readUintN(b []byte) uint64 {
	var v uint64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

// spillToContinuationChunk keeps the leading messages that fit into chunk #0
// (together with a continuation message), writes the rest into a new "OCHK"
// continuation chunk and returns the encoded (unpadded) chunk #0 messages.
func spillToContinuationChunk(rw ReaderWriterAt, alloc SpaceAllocator, addr, capacity uint64,
	msgs []*HeaderMessage, sb *Superblock, reuse *continuationChunk,
) ([]byte, error) {
	offsetSize, lengthSize := uint64(8), uint64(8)
	if sb != nil && sb.OffsetSize != 0 {
		offsetSize, lengthSize = uint64(sb.OffsetSize), uint64(sb.LengthSize)
	}
	contMsgSize := 4 + offsetSize + lengthSize
	if capacity < contMsgSize {
		return nil, fmt.Errorf("object header at %d too small (%d bytes) for a continuation message", addr, capacity)
	}

	var used uint64
	split := 0
	for split < len(msgs) {
		sz := 4 + uint64(len(msgs[split].Data))
		if used+sz > capacity-contMsgSize {
			break
		}
		used += sz
		split++
	}
	head, err := encodeV2Messages(msgs[:split])
	if err != nil {
		return nil, err
	}
	tail, err := encodeV2Messages(msgs[split:])
	if err != nil {
		return nil, err
	}

	// Continuation chunk: "OCHK" + messages (+ gap) + checksum.
	need := uint64(4 + len(tail) + ObjectHeaderV2ChecksumSize)
	var blockAddr, blockLen uint64
	if reuse != nil && reuse.length >= need {
		blockAddr, blockLen = reuse.addr, reuse.length
	} else {
		if alloc == nil {
			return nil, fmt.Errorf("object header at %d is full and no allocator is available for a continuation chunk", addr)
		}
		blockAddr, err = alloc.Allocate(need)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate continuation chunk: %w", err)
		}
		blockLen = need
	}
	block := make([]byte, 0, blockLen)
	block = append(block, "OCHK"...)
	block = append(block, tail...)
	block = appendV2Gap(block, blockLen-need)
	block = binary.LittleEndian.AppendUint32(block, utils.JenkinsChecksum(block))
	if _, err := rw.WriteAt(block, int64(blockAddr)); err != nil { //nolint:gosec // Safe: allocated address
		return nil, fmt.Errorf("failed to write continuation chunk: %w", err)
	}

	cont := make([]byte, offsetSize+lengthSize)
	writeUint64(cont[:offsetSize], blockAddr, int(offsetSize), binary.LittleEndian)          //nolint:gosec // G115: 2..8
	writeUint64(cont[offsetSize:], uint64(len(block)), int(lengthSize), binary.LittleEndian) //nolint:gosec // G115: 2..8
	contEnc, err := encodeV2Messages([]*HeaderMessage{{Type: MsgContinuation, Data: cont}})
	if err != nil {
		return nil, err
	}

	chunk := make([]byte, 0, len(head)+len(contEnc))
	chunk = append(chunk, head...)
	return append(chunk, contEnc...), nil
}
