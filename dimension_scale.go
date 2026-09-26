package hdf5

import (
	"encoding/binary"
	"errors"
	"fmt"
	"strings"

	"github.com/cwbudde/go-hdf5/internal/core"
)

// objectEntry is one object reachable from the root group.
type objectEntry struct {
	path string
	obj  Object
}

// objectIndex returns the address → object map of the file, built on
// first use by walking the group hierarchy. When an object is reachable by
// several paths (hard links), the first path in depth-first order wins.
func (f *File) objectIndex() map[uint64]objectEntry {
	f.objectsOnce.Do(f.buildObjectIndex)
	return f.objects
}

// buildObjectIndex walks the file once to fill f.objects. It runs under
// f.objectsOnce so concurrent first callers of Path, ObjectPath and
// Dereference do not race on the map.
func (f *File) buildObjectIndex() {
	idx := make(map[uint64]objectEntry)
	if f.root != nil && f.sb != nil {
		idx[f.sb.RootGroup] = objectEntry{path: "/", obj: f.root}
	}
	f.Walk(func(path string, obj Object) {
		var addr uint64
		switch o := obj.(type) {
		case *Dataset:
			addr = o.address
		case *Group:
			addr = o.address
		case *NamedDatatype:
			addr = o.address
		}
		if addr == 0 {
			return
		}
		if _, dup := idx[addr]; dup {
			return
		}
		if len(path) > 1 {
			path = strings.TrimSuffix(path, "/")
		}
		idx[addr] = objectEntry{path: path, obj: obj}
	})
	f.objects = idx
}

// Dereference returns the object (a *Dataset, *Group or *NamedDatatype) an
// object reference points to, like H5Rdereference. Only objects reachable
// from the root group can be resolved.
func (f *File) Dereference(ref ObjectRef) (Object, error) {
	e, ok := f.objectIndex()[uint64(ref)]
	if !ok {
		return nil, fmt.Errorf("object reference 0x%x does not point to an object in the file", uint64(ref))
	}
	return e.obj, nil
}

// ObjectPath returns the absolute path of the object an object reference
// points to (e.g. "/group/dataset"), like H5Iget_name. ok is false when no
// object reachable from the root group has that address.
func (f *File) ObjectPath(ref ObjectRef) (path string, ok bool) {
	e, ok := f.objectIndex()[uint64(ref)]
	return e.path, ok
}

// Reference returns an object reference to this dataset.
func (d *Dataset) Reference() ObjectRef {
	return ObjectRef(d.address)
}

// Path returns the absolute path of the dataset in its file. For a dataset
// reachable by several paths (hard links), one of them is returned.
func (d *Dataset) Path() string {
	if p, ok := d.file.ObjectPath(d.Reference()); ok {
		return p
	}
	return d.name
}

// attribute returns the named attribute; ok is false if the dataset has
// none.
func (d *Dataset) attribute(name string) (attr *core.Attribute, ok bool, err error) {
	attrs, err := d.Attributes()
	if err != nil {
		return nil, false, err
	}
	for _, a := range attrs {
		if a.Name == name {
			return a, true, nil
		}
	}
	return nil, false, nil
}

// stringAttribute reads a string-valued attribute; ok is false when the
// attribute is absent.
func (d *Dataset) stringAttribute(name string) (s string, ok bool, err error) {
	a, ok, err := d.attribute(name)
	if err != nil || !ok {
		return "", false, err
	}
	v, err := a.ReadValue()
	if err != nil {
		return "", true, fmt.Errorf("attribute %s: %w", name, err)
	}
	switch v := v.(type) {
	case string:
		return v, true, nil
	case []string:
		if len(v) == 1 {
			return v[0], true, nil
		}
	case []interface{}: // zero-size value, e.g. h5py's NAME="" for unnamed scales
		if len(v) == 0 {
			return "", true, nil
		}
	}
	return "", true, fmt.Errorf("attribute %s is not a string (%T)", name, v)
}

// IsDimensionScale reports whether the dataset is a dimension scale, like
// H5DSis_scale: it has the attribute CLASS="DIMENSION_SCALE".
func (d *Dataset) IsDimensionScale() bool {
	s, ok, err := d.stringAttribute(dimScaleClassAttr)
	return ok && err == nil && s == dimScaleClassValue
}

// DimensionScaleName returns the NAME attribute of a dimension scale, like
// H5DSget_scale_name. It returns "" when the dataset has no NAME.
func (d *Dataset) DimensionScaleName() (string, error) {
	s, _, err := d.stringAttribute(dimScaleNameAttr)
	return s, err
}

// DimensionList returns the dimension scales attached to each dimension of
// the dataset, as stored in its DIMENSION_LIST attribute: one slice of
// scale references per dimension. It returns nil when the dataset has no
// attached scales. Resolve the references with File.Dereference or
// File.ObjectPath, or use AttachedScales.
func (d *Dataset) DimensionList() ([][]ObjectRef, error) {
	a, ok, err := d.attribute(dimensionListAttr)
	if err != nil || !ok {
		return nil, err
	}
	return decodeDimensionList(a)
}

// AttachedScales returns the dimension scale datasets attached to dimension
// dim of the dataset (H5DSiterate_scales), in attachment order. The result
// is empty when no scale is attached to dim.
func (d *Dataset) AttachedScales(dim int) ([]*Dataset, error) {
	shape, err := d.Shape()
	if err != nil {
		return nil, err
	}
	if dim < 0 || dim >= len(shape) {
		return nil, fmt.Errorf("dimension %d out of range for rank %d", dim, len(shape))
	}
	lists, err := d.DimensionList()
	if err != nil {
		return nil, err
	}
	if dim >= len(lists) {
		return []*Dataset{}, nil
	}
	scales := make([]*Dataset, 0, len(lists[dim]))
	for _, ref := range lists[dim] {
		obj, err := d.file.Dereference(ref)
		if err != nil {
			return nil, fmt.Errorf("%s dimension %d: %w", dimensionListAttr, dim, err)
		}
		ds, ok := obj.(*Dataset)
		if !ok {
			return nil, fmt.Errorf("%s dimension %d: reference to %T, not a dataset", dimensionListAttr, dim, obj)
		}
		scales = append(scales, ds)
	}
	return scales, nil
}

// ReferenceList returns the datasets and dimensions a dimension scale is
// attached to, as stored in its REFERENCE_LIST attribute. It returns nil
// when the dataset has no REFERENCE_LIST. Resolve the dataset references
// with File.Dereference or File.ObjectPath.
func (d *Dataset) ReferenceList() ([]DimensionReference, error) {
	a, ok, err := d.attribute(referenceListAttr)
	if err != nil || !ok {
		return nil, err
	}
	return decodeReferenceList(a)
}

// decodeDimensionList decodes a DIMENSION_LIST attribute: a 1D array of
// variable-length sequences of object references.
func decodeDimensionList(attr *core.Attribute) ([][]ObjectRef, error) {
	if attr.Datatype == nil || attr.Datatype.Class != core.DatatypeVarLen {
		return nil, fmt.Errorf("%s is not a variable-length attribute", dimensionListAttr)
	}
	v, err := attr.ReadValue()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", dimensionListAttr, err)
	}
	switch v := v.(type) {
	case [][]ObjectRef:
		return v, nil
	case []interface{}: // empty attribute
		return nil, nil
	}
	return nil, fmt.Errorf("%s: unexpected value type %T", dimensionListAttr, v)
}

// decodeReferenceList decodes a REFERENCE_LIST attribute: a 1D array of
// the compound {dataset reference, uint32 dimension}.
func decodeReferenceList(attr *core.Attribute) ([]DimensionReference, error) {
	if attr.Datatype == nil || attr.Dataspace == nil || attr.Datatype.Class != core.DatatypeCompound {
		return nil, fmt.Errorf("%s is not a compound attribute", referenceListAttr)
	}
	refOff, dimOff := uint32(0), uint32(8) // libhdf5 ds_list_t layout
	if ct, err := core.ParseCompoundType(attr.Datatype); err == nil {
		for _, m := range ct.Members {
			switch m.Name {
			case "dataset":
				refOff = m.Offset
			case "dimension":
				dimOff = m.Offset
			}
		}
	}
	elemSize := uint64(attr.Datatype.Size)
	n := attr.Dataspace.TotalElements()
	if elemSize < uint64(refOff)+objectReferenceSize || elemSize < uint64(dimOff)+4 ||
		(n > 0 && elemSize > uint64(len(attr.Data))/n) {
		return nil, errors.New("unsupported " + referenceListAttr + " layout")
	}
	out := make([]DimensionReference, n)
	for i := range n {
		elem := attr.Data[i*elemSize : (i+1)*elemSize]
		out[i] = DimensionReference{
			Dataset: ObjectRef(binary.LittleEndian.Uint64(elem[refOff:])),
			Index:   int32(binary.LittleEndian.Uint32(elem[dimOff:])), //nolint:gosec // G115: two's complement
		}
	}
	return out, nil
}
