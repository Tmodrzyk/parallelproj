:tocdepth: 2

PET scanner geometries ``parallelproj.pet_scanners``
----------------------------------------------------

Scanner geometries describe the positions of line-of-response (LOR) endpoints.
The geometry determines which projector route to use (see :doc:`api_projectors`):

* :class:`.RegularPolygonPETScannerGeometry` describes rings of endpoints on
  the sides of a regular polygon. It supports both evenly spaced cylindrical
  endpoints and flat detector blocks. Use it with :class:`.Michelogram` and
  :class:`.RegularPolygonPETLORDescriptor`. All named scanner models below use
  this route.
* :class:`.ModularizedPETScannerGeometry` describes a general arrangement of
  scanner modules. Geometries made from :class:`.BlockPETScannerModule` objects
  with the same number of endpoints per module can be used with
  :class:`.EqualBlockPETLORDescriptor`.

:class:`.DemoPETScannerGeometry` provides a configurable example of a
regular-polygon scanner.

Ready-made scanner models
~~~~~~~~~~~~~~~~~~~~~~~~~

The following classes provide ready-made geometries for real scanners. See each
class for its references and constructor options.

* :class:`.SiemensMMRPETScannerGeometry`,
  :class:`.SiemensVision600PETScannerGeometry`,
  :class:`.SiemensVisionQuadraPETScannerGeometry` and
  :class:`.SiemensMCTPETScannerGeometry`: flat-block models by default,
  ``model="cylindrical"`` selects a STIR-based cylindrical model.
* :class:`.GEDiscoveryMIPETScannerGeometry`: a cylindrical model with 3, 4, 5
  or 6 axial block rings, selected with ``num_block_rings``.
* :class:`.GEDiscovery690PETScannerGeometry`: a cylindrical model by default,
  ``model="castor_blocks"`` selects a flat-block model.
* :class:`.MedisoAnyScanPETScannerGeometry`: a cylindrical model.
* :class:`.GESignaPETMRScannerGeometry` and
  :class:`.SiemensHRRTPETScannerGeometry`: STIR-inspired cylindrical models.
* :class:`.PhilipsVereosPETScannerGeometry` and
  :class:`.UnitedImagingUExplorerPETScannerGeometry`: flat-block models.

.. automodule:: parallelproj.pet_scanners
