:tocdepth: 2

PET scanner geometries ``parallelproj.pet_scanners``
----------------------------------------------------

The scanner geometry is the first link in the projector construction chain, and
which scanner class you pick selects one of the two projector routes (see
:doc:`api_projectors`):

* **Cylindrical scanners with a single layer of LOR endpoints** (the most common
  case): use :class:`.RegularPolygonPETScannerGeometry`, or
  :class:`.DemoPETScannerGeometry` for a ready-made example.  These feed the
  regular-polygon route (via :class:`.Michelogram` and
  :class:`.RegularPolygonPETLORDescriptor`).
* **General modular scanners** (block / panel geometries, or anything without
  cylindrical symmetry): build a :class:`.ModularizedPETScannerGeometry` from
  :class:`.BlockPETScannerModule` objects, which feeds the equal-block route
  (via :class:`.EqualBlockPETLORDescriptor`).

Ready-made scanner models
~~~~~~~~~~~~~~~~~~~~~~~~~

The following presets use detector dimensions from published sources and
PETRIC2/STIR headers (see each class for its source and assumptions):

* :class:`.SAFIRDualRingPETScannerGeometry`: flat-block SAFIR prototype.
* :class:`.NeuroLFPETScannerGeometry`: flat octagonal NeuroLF 15 mm geometry,
  including gaps between the crystal blocks.
* :class:`.SiemensMMRPETScannerGeometry`,
  :class:`.SiemensVision600PETScannerGeometry`,
  :class:`.SiemensVisionQuadraPETScannerGeometry` and
  :class:`.SiemensMCTPETScannerGeometry`: approximate flat-block models by
  default; ``model="cylindrical"`` selects a STIR-inspired geometry with
  virtual positions. The Vision 600 also offers ``model="castor_blocks"``.
* :class:`.GEDiscoveryMIPETScannerGeometry` (3-, 4-, 5- or 6-block-ring
  variants), :class:`.GEDiscovery690PETScannerGeometry`, and
  :class:`.MedisoAnyScanPETScannerGeometry`: the cylindrical models given in
  the PETRIC2 headers. Discovery 690 also offers a CASToR GATE flat-block
  model via ``model="castor_blocks"``.
* :class:`.GESignaPETMRScannerGeometry` and
  :class:`.SiemensHRRTPETScannerGeometry`: STIR-inspired cylindrical models.
  The HRRT approximation has one interaction depth and does not describe its
  two physical detector layers.
* :class:`.SiemensInveonPETScannerGeometry`,
  :class:`.PhilipsVereosPETScannerGeometry` and
  :class:`.UnitedImagingUExplorerPETScannerGeometry`: flat-block models based
  on published scanner descriptions and OpenGATE, with approximations described
  in their class documentation.

For example::

    import array_api_compat.numpy as xp
    from parallelproj.pet_scanners import SiemensMMRPETScannerGeometry

    physical = SiemensMMRPETScannerGeometry(xp, "cpu")
    logical = SiemensMMRPETScannerGeometry(xp, "cpu", model="cylindrical")
    is_virtual = logical.virtual_lor_endpoint_mask  # one Boolean per detector in a ring
    assert int(xp.sum(is_virtual)) == 56

The flat-block Siemens models omit virtual positions; their cylindrical models
include them. ``virtual_ring_mask`` identifies virtual axial rings, and
``all_virtual_lor_endpoint_mask`` combines both masks in flattened endpoint
order. Generic regular-polygon geometries have no declared virtual
positions. A mask is metadata: it is not automatically applied to projections.
In particular, view mashing and axial compression can combine several detector
pairs in one sinogram bin, so masking must account for each contributing pair.
The phases of the mCT and Vision virtual masks relative to measured data are
unverified.

These presets describe endpoint coordinates only. Their detector ordering and
orientation do not map directly to STIR projection data. Matching measured data
also requires the correct acquisition conventions, LOR pairs, TOF and
correction factors. Flat-block Siemens layouts are approximations at a single
interaction depth, not calibrated crystal maps. The PETRIC2 NeuroLF Hoffman
header uses a 134 mm inner radius, whereas the Esser header uses 135.5 mm;
pass ``inner_radius=134.0`` for the former.

The array backend ``xp`` and device ``dev`` passed here determine where all
downstream computation runs (CPU or GPU).  See :doc:`quickstart` for a runnable
example of the regular-polygon route.

.. automodule:: parallelproj.pet_scanners
