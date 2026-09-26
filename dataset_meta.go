package hdf5

import (
	"fmt"

	"github.com/cwbudde/go-hdf5/internal/core"
)

// datasetInfo reads the datatype, dataspace and layout messages of the
// dataset without reading its data.
func (d *Dataset) datasetInfo() (*core.DatasetInfo, error) {
	header, err := core.ReadObjectHeader(d.file.osFile, d.address, d.file.sb)
	if err != nil {
		return nil, err
	}
	info, err := core.ReadDatasetInfo(header, d.file.sb)
	if err != nil {
		return nil, fmt.Errorf("dataset %s: %w", d.name, err)
	}
	return info, nil
}

// Shape returns the current dimensions of the dataset's dataspace, like
// h5py's Dataset.shape. A scalar dataspace yields an empty (non-nil) slice;
// a null dataspace (no elements) yields nil. No data is read.
func (d *Dataset) Shape() ([]uint64, error) {
	info, err := d.datasetInfo()
	if err != nil {
		return nil, err
	}
	switch info.Dataspace.Type {
	case core.DataspaceNull:
		return nil, nil
	case core.DataspaceScalar:
		return []uint64{}, nil
	}
	return append([]uint64{}, info.Dataspace.Dimensions...), nil
}

// MaxShape returns the maximum dimensions of the dataset's dataspace, like
// h5py's Dataset.maxshape. Unlimited dimensions are reported as Unlimited.
// When the file stores no maximum dimensions they equal the current ones.
// Scalar and null dataspaces yield the same result as Shape.
func (d *Dataset) MaxShape() ([]uint64, error) {
	info, err := d.datasetInfo()
	if err != nil {
		return nil, err
	}
	switch info.Dataspace.Type {
	case core.DataspaceNull:
		return nil, nil
	case core.DataspaceScalar:
		return []uint64{}, nil
	}
	if len(info.Dataspace.MaxDims) == len(info.Dataspace.Dimensions) {
		return append([]uint64{}, info.Dataspace.MaxDims...), nil
	}
	return append([]uint64{}, info.Dataspace.Dimensions...), nil
}

// NumElements returns the number of elements in the dataset's dataspace
// (1 for a scalar, 0 for a null dataspace). No data is read.
func (d *Dataset) NumElements() (uint64, error) {
	info, err := d.datasetInfo()
	if err != nil {
		return 0, err
	}
	if info.Dataspace.Type == core.DataspaceNull {
		return 0, nil
	}
	return info.Dataspace.TotalElements(), nil
}

// Datatype returns the dataset's datatype: its Class (core.DatatypeFixed,
// core.DatatypeFloat, core.DatatypeString, core.DatatypeCompound,
// core.DatatypeReference, ...) and element Size in bytes, plus the raw
// class-specific properties.
func (d *Dataset) Datatype() (*core.DatatypeMessage, error) {
	info, err := d.datasetInfo()
	if err != nil {
		return nil, err
	}
	return info.Datatype, nil
}
