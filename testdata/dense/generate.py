"""Generate the dense-storage fixtures read by dense_storage_read_test.go.

    python3 testdata/dense/generate.py

Requires h5py and netCDF4 (netCDF-C). The outputs are committed so the tests
run without Python. They exercise dense link and attribute storage as the
HDF5 C library writes it: fractal heaps with checksummed direct blocks under
an indirect root block, and version 2 B-tree name indexes of depth >= 1.
"""

import os

import h5py
import netCDF4
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

LONG = "_a_rather_long_link_name_to_fill_the_fractal_heap_quickly"

# h5py (libver latest): 400 hard links with long names at the root, 300 root
# attributes and 300 attributes on one dataset.
with h5py.File(os.path.join(here, "h5py_many.h5"), "w", libver="latest") as f:
    d = f.create_dataset("data", data=np.arange(4, dtype="f8"))
    for i in range(400):
        f["l%03d%s" % (i, LONG)] = d
    for i in range(300):
        f.attrs["attr%03d" % i] = "value %d" % i
        d.attrs["dattr%03d" % i] = np.float64(i)

# netCDF-C: 150 variables and 300 global attributes (creation-order tracked,
# as netCDF-4 always does), one variable with 200 attributes.
with netCDF4.Dataset(os.path.join(here, "netcdf4_many.nc"), "w") as ds:
    ds.createDimension("n", 2)
    for i in range(150):
        v = ds.createVariable("var%03d%s" % (i, LONG), "f8", ("n",))
        v[:] = [i, i]
    for i in range(300):
        ds.setncattr("gattr%03d" % i, "global %d" % i)
    v = ds.variables["var000" + LONG]
    for i in range(200):
        v.setncattr("vattr%03d" % i, np.int32(i))

# netCDF-C with the link names (in creation order) of a SOFA SingleRoomSRIR
# file: the link "ReceiverUp" ends exactly at the end of the first
# (checksummed) direct block of the link heap.
with netCDF4.Dataset(os.path.join(here, "netcdf4_block_end.nc"), "w") as ds:
    for dim in ["E", "M", "R", "N", "C", "I", "S"]:
        ds.createDimension(dim, 1)
    names = [
        "RoomTemperature", "RoomVolume", "RoomCornerA", "RoomCornerB",
        "RoomCorners", "ListenerPosition", "ListenerView", "ListenerUp",
        "ReceiverDescriptions", "ReceiverPosition", "ReceiverView",
        "ReceiverUp", "SourcePosition", "SourceView", "SourceUp",
        "EmitterDescriptions", "EmitterPosition", "EmitterView", "EmitterUp",
        "MeasurementDate", "Data.IR", "Data.SamplingRate", "Data.Delay",
    ]
    for name in names:
        ds.createVariable(name, "f8", ("M",))[:] = [1.0]
