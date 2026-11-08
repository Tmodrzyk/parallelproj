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
    scanner = scanner_class(xp, dev)
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
