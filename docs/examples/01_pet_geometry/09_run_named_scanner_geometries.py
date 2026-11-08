"""
Named PET scanner geometries
============================

The presets below use published or PETRIC2/STIR detector layouts. Flat
blocks and cylindrical models can produce different LOR endpoints, and
virtual detector positions are retained only by the cylindrical Siemens
models. The example plots one transaxial detector ring per geometry.
"""

# %%
import matplotlib.pyplot as plt

import parallelproj
import parallelproj.pet_scanners as scanners
from parallelproj._examples_utils import suggest_array_backend_and_device

xp, dev = suggest_array_backend_and_device(None, None)

# %%
# The Siemens block and cylindrical models are different representations:
# the former contains physical positions only, the latter includes placeholders.
scanner_models = [
    ("mMR: flat blocks", scanners.SiemensMMRPETScannerGeometry(xp, dev)),
    (
        "mMR: STIR cylinder",
        scanners.SiemensMMRPETScannerGeometry(xp, dev, model="cylindrical"),
    ),
    ("Vision 600: flat blocks", scanners.SiemensVision600PETScannerGeometry(xp, dev)),
    ("GE Discovery MI", scanners.GEDiscoveryMIPETScannerGeometry(xp, dev)),
    ("GE Discovery 690", scanners.GEDiscovery690PETScannerGeometry(xp, dev)),
    ("Mediso AnyScan", scanners.MedisoAnyScanPETScannerGeometry(xp, dev)),
    ("NeuroLF", scanners.NeuroLFPETScannerGeometry(xp, dev)),
    ("SAFIR prototype", scanners.SAFIRDualRingPETScannerGeometry(xp, dev)),
]

fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
for ax, (title, scanner) in zip(axes.flat, scanner_models):
    n = scanner.num_lor_endpoints_per_ring
    xy = parallelproj.to_numpy_array(scanner.all_lor_endpoints[:n])[:, :2]
    is_virtual = parallelproj.to_numpy_array(scanner.virtual_lor_endpoint_mask)
    ax.scatter(xy[~is_virtual, 0], xy[~is_virtual, 1], s=3, label="physical")
    if is_virtual.any():
        ax.scatter(xy[is_virtual, 0], xy[is_virtual, 1], s=10, label="virtual")
        ax.legend(fontsize="small")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

fig.show()

# %%
# These models specify endpoint geometry, not a direct mapping to a PETRIC
# sinogram. See the scanner API documentation for virtual-mask phase,
# acquisition conventions and the assumptions behind each model.
