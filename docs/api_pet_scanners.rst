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

:class:`.SAFIRDualRingPETScannerGeometry` models the flat blocks and axial
block gap of STIR's SAFIR dual-ring prototype. :class:`.SiemensMMRPETScannerGeometry`
and :class:`.SiemensVision600PETScannerGeometry` use STIR's idealized
cylindrical models, including virtual positions. All three work with the
regular-polygon projector route::

    import array_api_compat.numpy as xp
    from parallelproj.pet_scanners import SAFIRDualRingPETScannerGeometry

    scanner = SAFIRDualRingPETScannerGeometry(xp, "cpu")

These describe endpoint coordinates only. In particular, virtual positions
are not masked and the ordering and orientation do not map directly to STIR
projection data; a suitable mapping and masking are needed when using measured
data. See the class documentation for the source parameters and assumptions.

The array backend ``xp`` and device ``dev`` passed here determine where all
downstream computation runs (CPU or GPU).  See :doc:`quickstart` for a runnable
example of the regular-polygon route.

.. automodule:: parallelproj.pet_scanners
