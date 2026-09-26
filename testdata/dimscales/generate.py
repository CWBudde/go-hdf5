"""Generate the dimension-scale fixtures read by dimension_scale_test.go.

    python3 testdata/dimscales/generate.py

Requires h5py and netCDF4 (netCDF-C). The outputs are committed so the tests
run without Python.
"""

import os

import h5py
import netCDF4
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

with h5py.File(os.path.join(here, "h5py_dimscales.h5"), "w", track_order=False) as f:
    t = f.create_dataset("time", data=np.arange(3, dtype="f8"))
    t.make_scale("time")
    fr = f.create_dataset("freq", data=np.arange(4, dtype="f8") * 10)
    fr.make_scale("frequency")
    alt = f.create_dataset("freq_alt", data=np.arange(4, dtype="i4"))
    alt.make_scale()  # no NAME

    d = f.create_dataset("data", data=np.zeros((3, 4)), maxshape=(None, 4), chunks=(1, 4))
    d.dims[0].attach_scale(t)
    d.dims[1].attach_scale(fr)
    d.dims[1].attach_scale(alt)

    g = f.create_group("grp")
    v = g.create_dataset("v", data=np.ones(3, dtype="f4"))
    v.dims[0].attach_scale(t)

    f.create_dataset("scalar", data=np.float64(1.5))
    f.create_dataset("empty", data=h5py.Empty("f8"))
    f.create_dataset("cube", data=np.zeros((2, 3, 5), dtype="i2"))

    d.attrs["ref"] = t.ref
    d.attrs["refs"] = np.array([t.ref, v.ref], dtype=h5py.ref_dtype)
    cdt = np.dtype([("obj", h5py.ref_dtype), ("n", "<i4")])
    f.create_dataset("refcompound", data=np.array([(t.ref, 7), (g.ref, 8)], dtype=cdt))

with netCDF4.Dataset(os.path.join(here, "netcdf4_dimscales.nc"), "w", format="NETCDF4") as nc:
    nc.createDimension("M", 3)
    nc.createDimension("N", 4)
    nc.createDimension("R", 2)
    m = nc.createVariable("M", "f8", ("M",))
    m[:] = np.arange(3)
    ir = nc.createVariable("Data.IR", "f8", ("M", "R", "N"))
    ir[:] = np.zeros((3, 2, 4))
    sp = nc.createVariable("SourcePosition", "f8", ("M", "R"))
    sp[:] = np.zeros((3, 2))
