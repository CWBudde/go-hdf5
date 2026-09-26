package core

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// GroupInfoMessage is the Group Info message (type 0x000A) of new-style
// groups. It stores the group's creation properties: the link count at which
// link storage switches between compact and dense, and size estimates for
// new groups. Fields not flagged as stored take the libhdf5 defaults.
//
// Reference: HDF5 Format Spec Section IV.A.2.j, H5Oginfo.c.
type GroupInfoMessage struct {
	Flags uint8 // GroupInfoLinkPhaseChange | GroupInfoEntryEstimates

	MaxCompact uint16 // Maximum number of compact links (if phase change stored)
	MinDense   uint16 // Minimum number of dense links (if phase change stored)

	EstNumEntries uint16 // Estimated number of entries (if estimates stored)
	EstNameLen    uint16 // Estimated link name length (if estimates stored)
}

// Group Info message flags.
const (
	GroupInfoLinkPhaseChange uint8 = 0x01 // MaxCompact and MinDense are stored
	GroupInfoEntryEstimates  uint8 = 0x02 // EstNumEntries and EstNameLen are stored
)

// libhdf5 defaults for groups (H5G_CRT_GINFO_MAX_COMPACT, H5G_CRT_GINFO_MIN_DENSE).
const (
	DefaultMaxCompactLinks uint16 = 8
	DefaultMinDenseLinks   uint16 = 6
)

// MaxCompactLinks returns the largest number of links kept in compact storage.
func (m *GroupInfoMessage) MaxCompactLinks() uint16 {
	if m.Flags&GroupInfoLinkPhaseChange != 0 {
		return m.MaxCompact
	}
	return DefaultMaxCompactLinks
}

// MinDenseLinks returns the link count below which dense storage converts
// back to compact storage.
func (m *GroupInfoMessage) MinDenseLinks() uint16 {
	if m.Flags&GroupInfoLinkPhaseChange != 0 {
		return m.MinDense
	}
	return DefaultMinDenseLinks
}

// EncodeGroupInfoMessage encodes a version 0 Group Info message. The zero
// value encodes the libhdf5 defaults as the two bytes {0, 0}.
func EncodeGroupInfoMessage(m *GroupInfoMessage) []byte {
	buf := []byte{0, m.Flags}
	if m.Flags&GroupInfoLinkPhaseChange != 0 {
		buf = binary.LittleEndian.AppendUint16(buf, m.MaxCompact)
		buf = binary.LittleEndian.AppendUint16(buf, m.MinDense)
	}
	if m.Flags&GroupInfoEntryEstimates != 0 {
		buf = binary.LittleEndian.AppendUint16(buf, m.EstNumEntries)
		buf = binary.LittleEndian.AppendUint16(buf, m.EstNameLen)
	}
	return buf
}

// ParseGroupInfoMessage decodes a version 0 Group Info message.
func ParseGroupInfoMessage(data []byte) (*GroupInfoMessage, error) {
	if len(data) < 2 {
		return nil, errors.New("group info message too short")
	}
	if data[0] != 0 {
		return nil, fmt.Errorf("unsupported group info message version %d", data[0])
	}
	m := &GroupInfoMessage{Flags: data[1]}
	if m.Flags&^(GroupInfoLinkPhaseChange|GroupInfoEntryEstimates) != 0 {
		return nil, fmt.Errorf("unknown group info message flags 0x%02x", m.Flags)
	}
	pos := 2
	next := func() (uint16, error) {
		if len(data) < pos+2 {
			return 0, errors.New("group info message truncated")
		}
		v := binary.LittleEndian.Uint16(data[pos:])
		pos += 2
		return v, nil
	}
	var err error
	if m.Flags&GroupInfoLinkPhaseChange != 0 {
		if m.MaxCompact, err = next(); err != nil {
			return nil, err
		}
		if m.MinDense, err = next(); err != nil {
			return nil, err
		}
	}
	if m.Flags&GroupInfoEntryEstimates != 0 {
		if m.EstNumEntries, err = next(); err != nil {
			return nil, err
		}
		if m.EstNameLen, err = next(); err != nil {
			return nil, err
		}
	}
	return m, nil
}
