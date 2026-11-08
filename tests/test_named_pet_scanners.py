"""Checks for the published scanner models and their endpoint coordinates."""

import numpy as np
import pytest

import parallelproj
import parallelproj.pet_lors as ppl
import parallelproj.pet_scanners as pps

from .config import pytestmark


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_safir_geometry(xp, dev, axis):
    scanner = pps.SAFIRDualRingPETScannerGeometry(xp, dev, symmetry_axis=axis)
    assert scanner.num_rings == 16
    assert scanner.num_lor_endpoints_per_ring == 180
    assert scanner.num_lor_endpoints == 16 * 180
    rings = parallelproj.to_numpy_array(scanner.ring_positions)
    expected = np.r_[np.arange(8) * 2.2, 18.1 + np.arange(8) * 2.2] - 16.75
    np.testing.assert_allclose(rings, expected, atol=2e-6)
    np.testing.assert_allclose(np.diff(rings)[7], 2.7, atol=2e-6)

    endpoints = parallelproj.to_numpy_array(scanner.all_lor_endpoints)
    np.testing.assert_allclose(endpoints[:180, axis], -16.75, atol=2e-6)
    # The first face is flat and its crystals have a 2.2 mm pitch.
    first_face = endpoints[:15]
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(first_face, axis=0), axis=1), 2.2, atol=1e-5
    )
    transverse = [i for i in range(3) if i != axis]
    distance = np.linalg.norm(first_face[:, transverse], axis=1)
    np.testing.assert_allclose(distance[7], 69.05, atol=1e-5)
    np.testing.assert_allclose(distance[0], np.hypot(69.05, 7 * 2.2), atol=1e-5)


@pytest.mark.parametrize(
    "scanner_class,num_rings,num_detectors,radius,spacing",
    [
        (pps.SiemensMMRPETScannerGeometry, 64, 504, 335.0, 4.0625),
        (pps.SiemensVision600PETScannerGeometry, 80, 798, 417.0, 3.29114),
    ],
)
def test_stir_cylindrical_geometry(
    xp, dev, scanner_class, num_rings, num_detectors, radius, spacing
):
    scanner = scanner_class(xp, dev, model="cylindrical")
    assert scanner.num_rings == num_rings
    assert scanner.num_lor_endpoints_per_ring == num_detectors
    assert scanner.num_lor_endpoints == num_rings * num_detectors
    rings = parallelproj.to_numpy_array(scanner.ring_positions)
    np.testing.assert_allclose(
        rings, spacing * (np.arange(num_rings) - (num_rings - 1) / 2), atol=2e-5
    )
    endpoints = parallelproj.to_numpy_array(scanner.all_lor_endpoints[:num_detectors])
    np.testing.assert_allclose(
        np.linalg.norm(endpoints[:, :2], axis=1), radius, atol=1e-4
    )
    np.testing.assert_allclose(endpoints[:, 2], rings[0], atol=1e-5)
    # Antipodal crystals are separated by exactly a half ring.
    np.testing.assert_allclose(
        endpoints[: num_detectors // 2, :2],
        -endpoints[num_detectors // 2 :, :2],
        atol=4e-4,
    )


@pytest.mark.parametrize(
    "scanner_class",
    [
        pps.SAFIRDualRingPETScannerGeometry,
        pps.SiemensMMRPETScannerGeometry,
        pps.SiemensVision600PETScannerGeometry,
    ],
)
def test_scanner_works_with_lor_descriptor(xp, dev, scanner_class):
    scanner = scanner_class(xp, dev)
    michelogram = ppl.Michelogram(scanner.num_rings, 0, span=1)
    descriptor = ppl.RegularPolygonPETLORDescriptor(scanner, michelogram)
    assert descriptor.scanner is scanner
    assert descriptor.num_views == scanner.num_lor_endpoints_per_ring // 2


@pytest.mark.parametrize(
    "scanner_class,sides,physical,pitch,virtual_period",
    [
        (pps.SiemensMMRPETScannerGeometry, 56, 8, 4.0, 9),
        (pps.SiemensVision600PETScannerGeometry, 38, 20, 3.2, 21),
    ],
)
def test_siemens_block_models_and_virtual_masks(
    xp, dev, scanner_class, sides, physical, pitch, virtual_period
):
    scanner = scanner_class(xp, dev)
    assert scanner.model == "blocks"
    assert scanner.num_sides == sides
    assert scanner.num_lor_endpoints_per_ring == sides * physical
    positions = parallelproj.to_numpy_array(scanner.all_lor_endpoints[:physical])
    np.testing.assert_allclose(positions[:, 1], -scanner.radius, atol=1e-5)
    np.testing.assert_allclose(np.diff(positions[:, 0]), pitch, atol=2e-5)
    assert not parallelproj.to_numpy_array(scanner.virtual_lor_endpoint_mask).any()

    logical = scanner_class(xp, dev, model="cylindrical")
    mask = parallelproj.to_numpy_array(logical.virtual_lor_endpoint_mask)
    expected = np.arange(sides * virtual_period) % virtual_period == 0
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(
        parallelproj.to_numpy_array(logical.all_virtual_lor_endpoint_mask),
        np.tile(expected, logical.num_rings),
    )
    with pytest.raises(ValueError, match="model must"):
        scanner_class(xp, dev, model="unknown")


@pytest.mark.parametrize(
    "scanner_class,kwargs,num_rings,num_detectors,radius,spacing",
    [
        (
            pps.GEDiscoveryMIPETScannerGeometry,
            {"num_block_rings": 3},
            27,
            544,
            380.5,
            5.52296,
        ),
        (
            pps.GEDiscoveryMIPETScannerGeometry,
            {"num_block_rings": 4},
            36,
            544,
            380.5,
            5.52296,
        ),
        (pps.GEDiscovery690PETScannerGeometry, {}, 24, 576, 414.5, 6.54),
        (pps.MedisoAnyScanPETScannerGeometry, {}, 38, 648, 430.0, 3.9),
    ],
)
def test_petric_cylindrical_scanners(
    xp, dev, scanner_class, kwargs, num_rings, num_detectors, radius, spacing
):
    scanner = scanner_class(xp, dev, **kwargs)
    assert scanner.num_rings == num_rings
    assert scanner.num_lor_endpoints_per_ring == num_detectors
    assert scanner.radius == radius
    np.testing.assert_allclose(
        parallelproj.to_numpy_array(scanner.ring_positions),
        spacing * (np.arange(num_rings) - (num_rings - 1) / 2),
        atol=2e-5,
    )
    assert not parallelproj.to_numpy_array(scanner.virtual_lor_endpoint_mask).any()
    descriptor = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(num_rings, 0, span=1)
    )
    assert descriptor.num_views == num_detectors // 2


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_neurolf_block_geometry(xp, dev, axis):
    scanner = pps.NeuroLFPETScannerGeometry(
        xp, dev, symmetry_axis=axis, inner_radius=134.0
    )
    assert scanner.num_rings == 48
    assert scanner.num_lor_endpoints_per_ring == 256
    assert scanner.num_sides == 8
    assert scanner.radius == pytest.approx(142.26007)
    rings = parallelproj.to_numpy_array(scanner.ring_positions)
    expected = 27.36 * (np.arange(48) // 8) + 3.313 * (np.arange(48) % 8)
    expected -= (5 * 27.36 + 7 * 3.313) / 2
    np.testing.assert_allclose(rings, expected, atol=2e-5)
    np.testing.assert_allclose(np.diff(rings)[7::8], 27.36 - 7 * 3.313, atol=2e-5)
    positions = parallelproj.to_numpy_array(scanner.modules[0].lor_endpoint_positions)
    np.testing.assert_allclose(np.diff(positions)[7::8], 27.36 - 7 * 3.313, atol=2e-5)
    np.testing.assert_allclose(positions, -positions[::-1], atol=2e-5)
    assert not parallelproj.to_numpy_array(scanner.virtual_lor_endpoint_mask).any()
    descriptor = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(48, 0, span=1)
    )
    assert descriptor.num_views == 128


def test_discovery_mi_rejects_unsupported_ring_counts(xp, dev):
    with pytest.raises(ValueError, match="num_block_rings"):
        pps.GEDiscoveryMIPETScannerGeometry(xp, dev, num_block_rings=2)


@pytest.mark.parametrize(
    "scanner_class,kwargs",
    [
        (pps.NeuroLFPETScannerGeometry, {}),
        (pps.SiemensMMRPETScannerGeometry, {}),
        (pps.SiemensMMRPETScannerGeometry, {"model": "cylindrical"}),
        (pps.MedisoAnyScanPETScannerGeometry, {}),
    ],
)
def test_named_scanners_project_through_centre(xp, dev, scanner_class, kwargs):
    import parallelproj.projectors as ppp

    scanner = scanner_class(xp, dev, **kwargs)
    descriptor = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 0, span=1)
    )
    projector = ppp.RegularPolygonPETProjector(
        descriptor,
        img_shape=(5, 5, 1),
        voxel_size=(4.0, 4.0, 10.0),
        views=xp.asarray([0], device=dev),
    )
    image = xp.ones((5, 5, 1), dtype=xp.float32, device=dev)
    projection = projector(image)
    assert bool(xp.any(projection > 0))
    assert bool(xp.any(projector.adjoint(projection) > 0))
