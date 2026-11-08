"""PET scanner geometry classes describing detector module layouts and endpoint coordinates.

Covers both modular scanners (:class:`ModularizedPETScannerGeometry`, where
modules can be placed arbitrarily) and the regular-polygon scanner
(:class:`RegularPolygonPETScannerGeometry`), which is the primary geometry
used throughout parallelproj.  Provides methods for computing LOR endpoint
world coordinates and visualising the detector arrangement.
"""

from __future__ import annotations

import abc
import enum
import warnings
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from ._backend import Array, to_numpy_array

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


class RingEndpointOrdering(enum.Enum):
    """Direction in which endpoint indices increase around a detector ring.

    The ordering is defined for the standard view of the scanner: looking
    along the symmetry axis from its **negative** side toward the positive
    side (for ``symmetry_axis=2`` this is looking from -z toward +z, the
    default 3D view, with +x to the right and +y pointing down).  Index 0
    sits at the 12 o'clock position (the top, i.e. the -y direction for
    ``symmetry_axis=2``); the two conventions differ in which way the
    indices then advance:

    ``CLOCKWISE``
        Indices advance clockwise on screen (toward +x first).  This is the
        default.

    ``COUNTERCLOCKWISE``
        Indices advance counterclockwise on screen (toward -x first).
    """

    CLOCKWISE = enum.auto()
    """Indices increase clockwise (default)."""
    COUNTERCLOCKWISE = enum.auto()
    """Indices increase counterclockwise."""


class PETScannerModule(abc.ABC):
    """Abstract base class for a single detector module in a PET scanner.

    A module groups a fixed number of LOR endpoints (detector elements) and
    exposes their world coordinates via :meth:`get_raw_lor_endpoints` (before
    any affine transform) and :meth:`get_lor_endpoints` (after).  An optional
    4x4 affine transformation matrix can be supplied to reposition the module
    in world space.

    Concrete subclasses:

    - :class:`BlockPETScannerModule` -- a 3-D rectangular grid of crystals
    - :class:`RegularPolygonPETScannerModule` -- one flat side of a regular-polygon scanner
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        num_lor_endpoints: int,
        affine_transformation_matrix: Array | None = None,
    ) -> None:
        """

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        num_lor_endpoints : int
            number of LOR endpoints in the module
        affine_transformation_matrix : Array | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._xp = xp
        self._dev = dev
        self._num_lor_endpoints = num_lor_endpoints
        self._lor_endpoint_numbers = self.xp.arange(num_lor_endpoints, device=self.dev)

        if affine_transformation_matrix is None:
            self._affine_transformation_matrix = self.xp.eye(4, device=self.dev)
            self._has_affine_transformation = False
        else:
            self._affine_transformation_matrix = affine_transformation_matrix
            self._has_affine_transformation = True

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self._xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self._dev

    @property
    def num_lor_endpoints(self) -> int:
        """total number of LOR endpoints in the module

        Returns
        -------
        int
        """
        return self._num_lor_endpoints

    @property
    def lor_endpoint_numbers(self) -> Array:
        """array enumerating all the LOR endpoints in the module

        Returns
        -------
        Array
        """
        return self._lor_endpoint_numbers

    @property
    def affine_transformation_matrix(self) -> Array:
        """4x4 affine transformation matrix

        Returns
        -------
        Array
            4x4 identity matrix if no transformation was provided
        """
        return self._affine_transformation_matrix

    @abc.abstractmethod
    def get_raw_lor_endpoints(self, inds: Array | None = None) -> Array:
        """mapping from LOR endpoint indices within module to an array of "raw" world coordinates

        Parameters
        ----------
        inds : Array | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        Array
            float array of shape (len(inds), 3) with the world coordinates of the LOR endpoints
        """
        raise NotImplementedError

    def get_lor_endpoints(self, inds: Array | None = None) -> Array:
        """mapping from LOR endpoint indices within module to an array of "transformed" world coordinates

        Parameters
        ----------
        inds : Array | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        Array
            float array of shape (len(inds), 3) with the world coordinates of the LOR endpoints
            after applying the affine transformation (if any)
        """

        lor_endpoints = self.get_raw_lor_endpoints(inds)

        if self._has_affine_transformation:
            tmp = self.xp.ones(
                (lor_endpoints.shape[0], 4), device=self.dev, dtype=self.xp.float32
            )
            tmp[:, :-1] = lor_endpoints
            lor_endpoints = (tmp @ self.affine_transformation_matrix.T)[:, :3]

        return lor_endpoints

    def show_lor_endpoints(
        self,
        ax: Axes3D,
        annotation_fontsize: float = 0,
        annotation_prefix: str = "",
        annotation_offset: int = 0,
        transformed: bool = True,
        **kwargs,
    ) -> None:
        """show the LOR coordinates in a 3D scatter plot

        Parameters
        ----------
        ax : Axes3D
            3D matplotlib axes
        annotation_fontsize : float, optional
            fontsize of LOR endpoint number annotation, by default 0
        annotation_prefix : str, optional
            prefix for annotation, by default ''
        annotation_offset : int, optional
            number to add to crystal number, by default 0
        transformed : bool, optional
            use transformed instead of raw coordinates, by default True
        """

        if transformed:
            all_lor_endpoints = self.get_lor_endpoints()
        else:
            all_lor_endpoints = self.get_raw_lor_endpoints()

        # convert to numpy array
        all_lor_endpoints = to_numpy_array(all_lor_endpoints)

        ax.scatter(
            all_lor_endpoints[:, 0],
            all_lor_endpoints[:, 1],
            all_lor_endpoints[:, 2],  # type: ignore[arg-type]
            **kwargs,
        )

        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )

        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_zlabel("x2")

        if annotation_fontsize > 0:
            for i in self.lor_endpoint_numbers:
                ax.text(
                    all_lor_endpoints[int(i), 0],
                    all_lor_endpoints[int(i), 1],
                    all_lor_endpoints[int(i), 2],
                    f"{annotation_prefix}{i+annotation_offset}",
                    fontsize=annotation_fontsize,
                )


class BlockPETScannerModule(PETScannerModule):
    """Block (rectangular cuboid) PET scanner module"""

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        shape: tuple[int, int, int],
        spacing: tuple[float, float, float],
        affine_transformation_matrix: Array | None = None,
    ) -> None:
        """

        Parameters
        ----------
        xp : ModuleType
            array module to use for storing the LOR endpoints
        dev : str
            device to use for storing the LOR endpoints
        shape : tuple[int, int, int]
            shape of the regular grid of LOR endpoints forming the block module
        spacing : tuple[float, float, float]
            spacing between the LOR endpoints in each direction
        affine_transformation_matrix : Array | None, optional
            4x4 affine transformation matrix applied eagerly to the LOR endpoint
            coordinates at construction time.  The transformed positions are
            stored directly; the matrix itself is not retained, so subsequent
            calls to :meth:`get_lor_endpoints` apply no further transform.
            ``None`` (default) leaves the endpoints in their local frame.
        """

        self._shape = shape
        self._spacing = spacing

        # calculate the LOR endpoints
        x0 = spacing[0] * (np.arange(shape[0], dtype=np.float32) - (shape[0] - 1) / 2)
        x1 = spacing[1] * (np.arange(shape[1], dtype=np.float32) - (shape[1] - 1) / 2)
        x2 = spacing[2] * (np.arange(shape[2], dtype=np.float32) - (shape[2] - 1) / 2)

        # in the current version (1.12.0) of array_api_compat.torch the indexing kwargs is ignored
        # which is why we stick to numpy
        X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing="ij")

        lor_endpoints_np = np.stack(
            (X0.ravel(), X1.ravel(), X2.ravel()),
            axis=-1,
        )

        self._lor_endpoints: Array = xp.asarray(lor_endpoints_np, device=dev)

        if affine_transformation_matrix is not None:
            tmp = xp.ones(
                (self._lor_endpoints.shape[0], 4), device=dev, dtype=xp.float32
            )
            tmp[:, :-1] = self._lor_endpoints

            self._lor_endpoints = (tmp @ affine_transformation_matrix.T)[:, :3]

        super().__init__(xp, dev, shape[0] * shape[1] * shape[2], None)

    @property
    def shape(self) -> tuple[int, int, int]:
        """shape of the block module

        Returns
        -------
        tuple[int, int, int]
        """
        return self._shape

    @property
    def spacing(self) -> tuple[float, float, float]:
        """spacing of the block module

        Returns
        -------
        tuple[float, float, float]
        """
        return self._spacing

    @property
    def lor_endpoints(self) -> Array:
        """LOR endpoints of the block module

        Returns
        -------
        Array
        """
        return self._lor_endpoints

    def get_raw_lor_endpoints(self, inds: Array | None = None) -> Array:
        """Return world coordinates of the requested crystal endpoints.

        Looks up pre-computed endpoint positions from the stored grid array.

        Parameters
        ----------
        inds : Array | None, optional
            Integer indices into the module's endpoint list.
            ``None`` returns all endpoints (default).

        Returns
        -------
        Array
            Float array of shape ``(len(inds), 3)`` with world coordinates.
        """
        if inds is None:
            inds = self.lor_endpoint_numbers

        return self.xp.take(self.lor_endpoints, inds, axis=0)


class RegularPolygonPETScannerModule(PETScannerModule):
    """Regular polygon PET scanner module (detectors on a regular polygon)"""

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        radius: float,
        num_sides: int,
        num_lor_endpoints_per_side: int | None = None,
        lor_spacing: float | None = None,
        ax0: int = 2,
        ax1: int = 1,
        affine_transformation_matrix: Array | None = None,
        phis: None | Array = None,
        ring_endpoint_ordering: RingEndpointOrdering = RingEndpointOrdering.CLOCKWISE,
        phi0: float = 0.0,
        lor_endpoint_positions: Array | None = None,
    ) -> None:
        """

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        radius : float
            inner radius of the regular polygon
        num_sides: int
            number of sides of the regular polygon
        num_lor_endpoints_per_side: int or None, optional
            number of LOR endpoints per side.  Required when
            ``lor_endpoint_positions`` is not given; ignored otherwise.
        lor_spacing : float or None, optional
            uniform spacing between LOR endpoints in mm.  Required when
            ``lor_endpoint_positions`` is not given; ignored otherwise.
        ax0 : int, optional
            axis number for the first direction, by default 2
        ax1 : int, optional
            axis number for the second direction, by default 1
        affine_transformation_matrix : Array | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        phis : None | Array, optional
            angle of each side, by default None
            means that the sides are equally spaced around a circle
        ring_endpoint_ordering : RingEndpointOrdering, optional
            direction in which endpoint indices increase around the ring, by
            default ``RingEndpointOrdering.CLOCKWISE``.
        phi0 : float, optional
            azimuthal offset of side 0 in radians, by default 0.  With the
            default (and ``symmetry_axis=2``) side 0 is centred on the -y axis,
            which is the "top" of the default 3D view (viewed from -z with +y
            pointing down).  A positive ``phi0`` is a right-hand rotation about
            the symmetry axis, i.e. for ``symmetry_axis=2`` it moves side 0
            toward +x.  Only applied when ``phis`` is ``None``; ignored when
            ``phis`` is provided explicitly.
        lor_endpoint_positions : Array or None, optional
            1-D array of crystal positions (in mm) along each polygon side,
            with 0 at the centre of the side.  When given, overrides
            ``num_lor_endpoints_per_side`` and ``lor_spacing``.

            For radial sinogram symmetry to hold (see
            :func:`~parallelproj.sinogram_symmetries.build_radial_class_indices`),
            the array must be **anti-symmetric about 0**:
            ``pos[i] == -pos[N-1-i]`` for all ``i``.  A ``UserWarning`` is
            issued if this condition is not met.

            Examples for a side with **even** N=6 (3 crystals each half,
            uniform 2 mm pitch, 1 mm gap)::

                lor_endpoint_positions = xp.asarray(
                    [-3.5, -1.5, -0.5, 0.5, 1.5, 3.5], device=dev
                )

            Examples for **odd** N=5 (2 crystals each half + one centre
            crystal, uniform 2 mm pitch)::

                lor_endpoint_positions = xp.asarray(
                    [-4.0, -2.0, 0.0, 2.0, 4.0], device=dev
                )
        """

        self._radius = radius
        self._num_sides = num_sides
        self._ax0 = ax0
        self._ax1 = ax1
        self._ring_endpoint_ordering = ring_endpoint_ordering
        self._phi0 = phi0

        if lor_endpoint_positions is not None:
            # lor_endpoint_positions is already an Array on the correct device
            max_abs = max(float(xp.max(xp.abs(lor_endpoint_positions))), 1.0)
            if not bool(
                xp.all(
                    xp.abs(lor_endpoint_positions + xp.flip(lor_endpoint_positions))
                    < 1e-4 * max_abs
                )
            ):
                warnings.warn(
                    "lor_endpoint_positions is not anti-symmetric about 0 "
                    "(pos[i] != -pos[N-1-i]). Radial sinogram symmetry "
                    "(build_radial_class_indices) may not be physically valid.",
                    UserWarning,
                    stacklevel=2,
                )
            self._lor_endpoint_positions = lor_endpoint_positions
            self._num_lor_endpoints_per_side = int(lor_endpoint_positions.shape[0])
            self._lor_spacing = None
        elif num_lor_endpoints_per_side is not None and lor_spacing is not None:
            N = num_lor_endpoints_per_side
            self._lor_endpoint_positions = lor_spacing * (
                xp.arange(N, dtype=xp.float32, device=dev) - (N - 1) / 2.0
            )
            self._num_lor_endpoints_per_side = N
            self._lor_spacing = lor_spacing
        else:
            raise ValueError(
                "Provide either lor_endpoint_positions or both "
                "num_lor_endpoints_per_side and lor_spacing."
            )

        super().__init__(
            xp,
            dev,
            num_sides * self._num_lor_endpoints_per_side,
            affine_transformation_matrix,
        )

        # angle of each "side"
        if phis is None:
            self._phis = (
                phi0
                + 2
                * self.xp.pi
                * self.xp.arange(self._num_sides, dtype=xp.float32, device=dev)
                / self.num_sides
            )
        else:
            self._phis = phis

    @property
    def radius(self) -> float:
        """inner radius of the regular polygon

        Returns
        -------
        float
        """
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides of the regular polygon

        Returns
        -------
        int
        """
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side

        Returns
        -------
        int
        """
        return self._num_lor_endpoints_per_side

    @property
    def ax0(self) -> int:
        """axis number for the first module direction

        Returns
        -------
        int
        """
        return self._ax0

    @property
    def ax1(self) -> int:
        """axis number for the second module direction

        Returns
        -------
        int
        """
        return self._ax1

    @property
    def lor_spacing(self) -> float | None:
        """Uniform spacing between LOR endpoints in mm, or ``None`` when
        custom ``lor_endpoint_positions`` were supplied."""
        return self._lor_spacing

    @property
    def lor_endpoint_positions(self) -> Array:
        """1-D float32 Array of crystal positions along each polygon side (mm),
        on the same device as the scanner.

        Anti-symmetric about 0 for standard scanners (uniform or gap layout).
        """
        return self._lor_endpoint_positions

    @property
    def phis(self) -> Array:
        """azimuthal angle of each side

        Returns
        -------
        Array
        """
        return self._phis

    @property
    def ring_endpoint_ordering(self) -> RingEndpointOrdering:
        """direction in which endpoint indices increase around the ring"""
        return self._ring_endpoint_ordering

    @property
    def phi0(self) -> float:
        """azimuthal offset of side 0 in radians (only applied when phis=None)"""
        return self._phi0

    # abstract method from base class to be implemented
    def get_raw_lor_endpoints(self, inds: Array | None = None) -> Array:
        """Compute world coordinates for the requested crystal endpoints.

        Calculates endpoint positions analytically from the scanner geometry
        (radius, number of sides, crystals per side, azimuthal offset),
        respecting the configured :class:`RingEndpointOrdering`.

        Parameters
        ----------
        inds : Array | None, optional
            Integer indices into the module's endpoint list.
            ``None`` returns all endpoints (default).

        Returns
        -------
        Array
            Float array of shape ``(len(inds), 3)`` with world coordinates.
        """
        if inds is None:
            inds = self.lor_endpoint_numbers

        if self._ring_endpoint_ordering is RingEndpointOrdering.COUNTERCLOCKWISE:
            side = inds // self._num_lor_endpoints_per_side
            within = inds - side * self._num_lor_endpoints_per_side
            new_side = self.xp.astype(
                (self._num_sides - side) % self._num_sides, self.xp.int32
            )
            new_within = self.xp.astype(
                self._num_lor_endpoints_per_side - 1 - within, self.xp.int32
            )
            inds = new_side * self._num_lor_endpoints_per_side + new_within

        side = inds // self.num_lor_endpoints_per_side
        within_side = inds - side * self.num_lor_endpoints_per_side

        # Look up the physical position along the side for each endpoint.
        # For uniform spacing this is lor_spacing*(i - (N-1)/2); for custom
        # positions it is whatever the user supplied.
        pos = self.xp.take(self._lor_endpoint_positions, within_side)

        phi = self.xp.take(self._phis, side)

        lor_endpoints = self.xp.zeros(
            (inds.shape[0], 3), device=self.dev, dtype=self.xp.float32
        )

        cosphi = self.xp.cos(phi)
        sinphi = self.xp.sin(phi)

        # The transaxial ``ax0`` coordinate is reflected (leading minus sign) so
        # that -- for ``symmetry_axis=2`` viewed from -z with +y pointing down
        # (the default 3D view) -- side 0 (``phi0=0``) sits at the *top* (-y) and
        # increasing ``phi``/``phi0`` is a right-hand rotation about +z (module 0
        # moves toward +x).  Endpoint numbering then runs clockwise on screen for
        # ``RingEndpointOrdering.CLOCKWISE`` (the identity ordering).
        lor_endpoints[:, self.ax0] = -cosphi * self.radius + sinphi * pos
        lor_endpoints[:, self.ax1] = sinphi * self.radius + cosphi * pos

        return lor_endpoints


class ModularizedPETScannerGeometry:
    """A PET scanner geometry built from an ordered list of :class:`PETScannerModule` objects.

    Each module contributes a contiguous block of LOR endpoints to the global
    flat index space.  The global index of endpoint ``k`` in module ``i`` is
    ``all_lor_endpoints_index_offset[i] + k``.  All modules must share the
    same array namespace (``xp``) and device.

    Use :class:`RegularPolygonPETScannerGeometry` for the common case of a
    cylindrical scanner with stacked regular-polygon rings.  Use this class
    directly when the scanner has an irregular or custom module layout.
    """

    def __init__(self, modules: Sequence[PETScannerModule]):
        """
        Parameters
        ----------
        modules : Sequence[PETScannerModule]
            a sequence of scanner modules
        """

        # member variable that determines whether we want to use
        # a numpy or cupy array to store the array of all lor endpoints
        self._modules = modules
        self._num_modules = len(self._modules)
        self._num_lor_endpoints_per_module = self.xp.asarray(
            [x.num_lor_endpoints for x in self._modules], device=self.dev
        )
        self._num_lor_endpoints = int(self.xp.sum(self._num_lor_endpoints_per_module))

        # declare attributes set by setup_all_lor_endpoints so they are visible in __init__
        self._all_lor_endpoints_index_offset: Array = None  # type: ignore[assignment]
        self._all_lor_endpoints: Array = None  # type: ignore[assignment]
        self._all_lor_endpoints_module_number: Array = None  # type: ignore[assignment]

        self.setup_all_lor_endpoints()

    def setup_all_lor_endpoints(self) -> None:
        """calculate the position of all lor endpoints by iterating over
        the modules and calculating the transformed coordinates of all
        module endpoints
        """

        offsets = [0]
        for module in self._modules[:-1]:
            offsets.append(offsets[-1] + module.num_lor_endpoints)
        self._all_lor_endpoints_index_offset = self.xp.asarray(offsets, device=self.dev)

        self._all_lor_endpoints = self.xp.zeros(
            (self._num_lor_endpoints, 3), device=self.dev, dtype=self.xp.float32
        )

        for i, module in enumerate(self._modules):
            self._all_lor_endpoints[
                int(self._all_lor_endpoints_index_offset[i]) : int(
                    self._all_lor_endpoints_index_offset[i] + module.num_lor_endpoints
                ),
                :,
            ] = self.xp.astype(module.get_lor_endpoints(), self.xp.float32)

        self._all_lor_endpoints_module_number = self.xp.asarray(
            [
                i
                for i, module in enumerate(self._modules)
                for _ in range(module.num_lor_endpoints)
            ],
            device=self.dev,
        )

    @property
    def modules(self) -> Sequence[PETScannerModule]:
        """sequence of modules defining the scanner"""
        return self._modules

    @property
    def num_modules(self) -> int:
        """the number of modules defining the scanner"""
        return self._num_modules

    @property
    def num_lor_endpoints_per_module(self) -> Array:
        """array showing how many LOR endpoints are in every module"""
        return self._num_lor_endpoints_per_module

    @property
    def num_lor_endpoints(self) -> int:
        """the total number of LOR endpoints in the scanner"""
        return self._num_lor_endpoints

    @property
    def all_lor_endpoints_index_offset(self) -> Array:
        """the offset in the linear (flattend) index for all LOR endpoints"""
        return self._all_lor_endpoints_index_offset

    @property
    def all_lor_endpoints_module_number(self) -> Array:
        """the module number of all LOR endpoints"""
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints(self) -> Array:
        """the world coordinates of all LOR endpoints"""
        return self._all_lor_endpoints

    @property
    def xp(self) -> ModuleType:
        """Array module of the first module.

        All modules in the scanner must share the same array namespace;
        this property returns the namespace of the first one as representative.
        """
        return self._modules[0].xp

    @property
    def dev(self) -> str:
        """Device of the first module.

        All modules in the scanner must reside on the same device;
        this property returns the device of the first one as representative.
        """
        return self._modules[0].dev

    def linear_lor_endpoint_index(
        self,
        module: Array,
        index_in_module: Array,
    ) -> Array:
        """transform the module + index_in_modules indices into a flattened / linear LOR endpoint index

        Parameters
        ----------
        module : Array
            containing module numbers
        index_in_module : Array
            containing index in modules

        Returns
        -------
        Array
            the flattened LOR endpoint index
        """

        return (
            self.xp.take(self.all_lor_endpoints_index_offset, module, axis=0)
            + index_in_module
        )

    def get_lor_endpoints(self, module: Array, index_in_module: Array) -> Array:
        """get the coordinates for LOR endpoints defined by module and index in module

        Parameters
        ----------
        module : Array
            the module number of the LOR endpoints
        index_in_module : Array
            the index in module number of the LOR endpoints

        Returns
        -------
        Array
            the 3 world coordinates of the LOR endpoints
        """
        return self.xp.take(
            self.all_lor_endpoints,
            self.linear_lor_endpoint_index(module, index_in_module),
            axis=0,
        )

    def show_lor_endpoints(
        self, ax: Axes3D, show_linear_index: bool = True, **kwargs
    ) -> None:
        """show all LOR endpoints in a 3D plot

        Parameters
        ----------
        ax : Axes3D
            a 3D matplotlib axes
        show_linear_index : bool, optional
            annotate the LOR endpoints with the linear LOR endpoint index
        **kwargs : keyword arguments
            passed to show_lor_endpoints() of the scanner module
        """
        for i, module in enumerate(self.modules):
            if show_linear_index:
                offset = int(to_numpy_array(self.all_lor_endpoints_index_offset[i]))
                prefix = ""
            else:
                offset = 0
                prefix = f"{i},"

            module.show_lor_endpoints(
                ax, annotation_offset=offset, annotation_prefix=prefix, **kwargs
            )


class RegularPolygonPETScannerGeometry(ModularizedPETScannerGeometry):
    """A cylindrical PET scanner built from stacked regular-polygon rings.

    Each axial ring is a :class:`RegularPolygonPETScannerModule` with
    ``num_sides`` sides and ``num_lor_endpoints_per_side`` crystals per side,
    giving ``num_lor_endpoints_per_ring = num_sides * num_lor_endpoints_per_side``
    endpoints per ring.  Rings are stacked axially; the global flat endpoint
    index increases first within a ring and then across rings, so endpoint
    ``r * num_lor_endpoints_per_ring + k`` belongs to ring ``r`` and in-ring
    position ``k``.
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        radius: float,
        num_sides: int,
        ring_positions: Array,
        symmetry_axis: int,
        *,
        num_lor_endpoints_per_side: int | None = None,
        lor_spacing: float | None = None,
        phis: None | Array = None,
        ring_endpoint_ordering: RingEndpointOrdering = RingEndpointOrdering.CLOCKWISE,
        phi0: float = 0.0,
        lor_endpoint_positions: Array | None = None,
    ) -> None:
        """
        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        radius : float
            inner radius of the regular polygon (distance from centre to detector face) in mm
        num_sides : int
            number of sides (faces) of each regular polygon
        ring_positions : Array
            1D array with the coordinate of the rings along the ring axis
        symmetry_axis : int
            the ring axis (0,1,2)
        num_lor_endpoints_per_side : int or None, optional
            number of LOR endpoints in each side.  Required when
            ``lor_endpoint_positions`` is not given; ignored otherwise.
        lor_spacing : float or None, optional
            uniform spacing between LOR endpoints in mm.  Required when
            ``lor_endpoint_positions`` is not given; ignored otherwise.
        phis : None | Array, optional
            angle of each side, by default None
            means that the sides are equally spaced around a circle
        ring_endpoint_ordering : RingEndpointOrdering, optional
            direction in which endpoint indices increase around the ring, by
            default ``RingEndpointOrdering.CLOCKWISE``.
        phi0 : float, optional
            azimuthal offset of side 0 in radians, by default 0.  With the
            default (and ``symmetry_axis=2``) side 0 is centred on the -y axis,
            which is the "top" of the default 3D view (viewed from -z with +y
            pointing down).  A positive ``phi0`` is a right-hand rotation about
            the symmetry axis, i.e. for ``symmetry_axis=2`` it moves side 0
            toward +x.  Only applied when ``phis`` is ``None``; ignored when
            ``phis`` is provided explicitly.
        lor_endpoint_positions : Array or None, optional
            Custom 1-D array of crystal positions along each polygon side in mm.
            When given, overrides ``num_lor_endpoints_per_side`` and
            ``lor_spacing``.  See :class:`RegularPolygonPETScannerModule` for
            details and anti-symmetry requirements.
        """

        self._radius = radius
        self._num_sides = num_sides
        self._symmetry_axis = symmetry_axis
        self._ring_positions = ring_positions
        self._ring_endpoint_ordering = ring_endpoint_ordering
        self._phi0 = phi0

        # Resolve positions: custom or uniform
        if lor_endpoint_positions is not None:
            # lor_endpoint_positions is already an Array on the correct device
            _positions = lor_endpoint_positions
            self._num_lor_endpoints_per_side = int(_positions.shape[0])
            self._lor_spacing = None
        elif num_lor_endpoints_per_side is not None and lor_spacing is not None:
            N = num_lor_endpoints_per_side
            _positions = lor_spacing * (
                xp.arange(N, dtype=xp.float32, device=dev) - (N - 1) / 2.0
            )
            self._num_lor_endpoints_per_side = N
            self._lor_spacing = lor_spacing
        else:
            raise ValueError(
                "Provide either lor_endpoint_positions or both "
                "num_lor_endpoints_per_side and lor_spacing."
            )

        if symmetry_axis == 0:
            self._ax0 = 2
            self._ax1 = 1
        elif symmetry_axis == 1:
            self._ax0 = 0
            self._ax1 = 2
        elif symmetry_axis == 2:
            self._ax0 = 1
            self._ax1 = 0
        else:
            raise ValueError("symmetry_axis must be 0, 1, or 2")

        modules = []

        for ring in range(self.num_rings):
            aff_mat = xp.eye(4, device=dev, dtype=xp.float32)
            aff_mat[symmetry_axis, -1] = xp.astype(ring_positions[ring], xp.float32)

            modules.append(
                RegularPolygonPETScannerModule(
                    xp,
                    dev,
                    radius,
                    num_sides,
                    affine_transformation_matrix=aff_mat,
                    ax0=self._ax0,
                    ax1=self._ax1,
                    phis=phis,
                    ring_endpoint_ordering=ring_endpoint_ordering,
                    phi0=phi0,
                    lor_endpoint_positions=_positions,
                )
            )

        super().__init__(tuple(modules))

        self._all_lor_endpoints_index_in_ring = (
            self.xp.arange(self.num_lor_endpoints, device=dev)
            - self.all_lor_endpoints_ring_number * self.num_lor_endpoints_per_module[0]
        )

    @property
    def radius(self) -> float:
        """Inner radius of the regular polygon (distance from centre to detector face) in mm."""
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides (faces) of each polygon"""
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side (face) in each polygon"""
        return self._num_lor_endpoints_per_side

    @property
    def num_rings(self) -> int:
        """number of rings (regular polygons)"""
        return self._ring_positions.shape[0]

    @property
    def lor_spacing(self) -> float | None:
        """Uniform spacing between LOR endpoints in mm, or ``None`` when
        custom ``lor_endpoint_positions`` were supplied."""
        return self._lor_spacing

    @property
    def lor_endpoint_positions(self) -> Array:
        """1-D float32 Array of crystal positions along each polygon side (mm),
        on the same device as the scanner."""
        return self.modules[0].lor_endpoint_positions

    @property
    def symmetry_axis(self) -> int:
        """The symmetry axis. Also called axial (or ring) direction."""
        return self._symmetry_axis

    @property
    def ring_endpoint_ordering(self) -> RingEndpointOrdering:
        """direction in which endpoint indices increase around the ring"""
        return self._ring_endpoint_ordering

    @property
    def phi0(self) -> float:
        """azimuthal offset of side 0 in radians (only applied when phis=None)"""
        return self._phi0

    @property
    def all_lor_endpoints_ring_number(self) -> Array:
        """Ring (axial module) index for every LOR endpoint.

        For a regular-polygon scanner each axial ring is one module, so this
        is an alias for ``all_lor_endpoints_module_number``.  Values range
        from ``0`` to ``num_rings - 1``.
        """
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints_index_in_ring(self) -> Array:
        """the index within the ring (regular polygon) of all LOR endpoints"""
        return self._all_lor_endpoints_index_in_ring

    @property
    def num_lor_endpoints_per_ring(self) -> int:
        """the number of LOR endpoints per ring (regular polygon)"""
        return int(self._num_lor_endpoints_per_module[0])

    @property
    def ring_positions(self) -> Array:
        """the ring (regular polygon) positions"""
        return self._ring_positions

    @property
    def virtual_lor_endpoint_mask(self) -> Array:
        """Boolean mask of virtual positions within one ring.

        This follows this geometry's endpoint order, which is not necessarily
        the index convention of an external sinogram. Generic geometries
        have no virtual positions unless a subclass specifies them. Applying
        the mask to compressed or view-mashed data requires the contributing
        detector pairs, not just a mask per output bin.
        """
        indices = self.xp.arange(self.num_lor_endpoints_per_ring, device=self.dev)
        period = getattr(self, "_virtual_endpoint_period", None)
        if period is None:
            return self.xp.zeros(indices.shape, dtype=self.xp.bool, device=self.dev)
        return indices % period == 0

    @property
    def all_virtual_lor_endpoint_mask(self) -> Array:
        """Boolean virtual-position mask in flattened endpoint order.

        An endpoint is virtual if either its ring or its in-ring position is
        virtual. No mask is applied automatically by the projectors.
        """
        indices = self.xp.arange(self.num_lor_endpoints, device=self.dev)
        transaxial = self.xp.take(
            self.virtual_lor_endpoint_mask, indices % self.num_lor_endpoints_per_ring
        )
        axial = self.xp.take(
            self.virtual_ring_mask, indices // self.num_lor_endpoints_per_ring
        )
        return transaxial | axial

    @property
    def virtual_ring_mask(self) -> Array:
        """Boolean mask of axial virtual rings in this geometry's ring order."""
        indices = self.xp.arange(self.num_rings, device=self.dev)
        period = getattr(self, "_virtual_ring_period", None)
        if period is None:
            return self.xp.zeros(indices.shape, dtype=self.xp.bool, device=self.dev)
        return indices % period == period - 1


class DemoPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Demo PET scanner geometry consisting of a 34-ogon with 16 LOR endpoints per side and 36 rings"""

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        radius: float = 0.5 * (744.1 + 2 * 8.51),
        num_sides: int = 34,
        num_lor_endpoints_per_side: int = 16,
        lor_spacing: float = 4.03125,
        num_rings: int = 36,
        symmetry_axis: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        xp : ModuleType
            array module
        dev : str
            the device to use
        radius : float, optional
            radius of the regular polygon, by default 0.5*(744.1 + 2 * 8.51)
        num_sides : int, optional
            number of sides of the polygon, by default 34
        num_lor_endpoints_per_side : int, optional
            number of LOR endpoints per side, by default 16
        lor_spacing : float, optional
            spacing between the LOR endpoints, by default 4.03125
        num_rings : int, optional
            number of rings, by default 36
        symmetry_axis : int, optional
            symmetry (axial) axis of the scanner, by default 2
        """

        ring_positions = (
            5.32 * xp.arange(num_rings, device=dev, dtype=xp.float32)
            + (xp.astype(xp.arange(num_rings, device=dev) // 9, xp.float32)) * 2.8
        )
        ring_positions = ring_positions - 0.5 * xp.max(ring_positions)

        super().__init__(
            xp,
            dev,
            radius=radius,
            num_sides=num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            ring_positions=ring_positions,
            symmetry_axis=symmetry_axis,
        )


class SAFIRDualRingPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """STIR's block model of the SAFIR dual-ring prototype.

    The 12 flat detector faces each contain 15 crystals at 2.2 mm pitch.
    The 16 axial rings comprise two blocks of eight crystals, with a
    18.1 mm centre-to-centre block spacing. The detector plane is at
    64.05 + 5 mm, including STIR's average depth of interaction.

    Parameters are from STIR's ``SAFIRDualRingPrototype`` definition:
    https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx
    Coordinates are centred axially, in mm. The default orientation and
    endpoint numbering follow parallelproj, not STIR's detector map.
    """

    def __init__(self, xp: ModuleType, dev: str, symmetry_axis: int = 2) -> None:
        ring_indices = xp.arange(16, device=dev)
        rings = (
            2.2 * xp.astype(ring_indices % 8, xp.float32)
            + 18.1 * xp.astype(ring_indices // 8, xp.float32)
            - 16.75
        )
        super().__init__(
            xp,
            dev,
            radius=69.05,
            num_sides=12,
            num_lor_endpoints_per_side=15,
            lor_spacing=2.2,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )


class SiemensMMRPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Siemens Biograph mMR endpoint geometry.

    ``model="blocks"`` (default) uses 56 flat faces with eight physical
    crystals at 4 mm pitch, and 64 rings. ``model="cylindrical"`` uses
    STIR's 504 equally spaced positions per ring, including 56 virtual
    positions. In both cases the effective radius is 328 + 7 mm and the
    axial pitch is 4.0625 mm. The flat model is an idealized approximation:
    the interaction depth and detailed gaps are not individually calibrated.

    Sources: https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx
    and https://doi.org/10.1088/1361-6560/ad1cf9. Coordinates are in mm.
    The virtual mask uses position 0 of each group of nine, matching STIR's
    mMR gap convention. Neither model provides a mapping to STIR sinogram
    bins; orientation and endpoint numbering follow parallelproj.
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        symmetry_axis: int = 2,
        *,
        model: str = "blocks",
    ) -> None:
        if model not in ("blocks", "cylindrical"):
            raise ValueError("model must be 'blocks' or 'cylindrical'")
        self._virtual_endpoint_period = 9 if model == "cylindrical" else None
        self._model = model
        rings = 4.0625 * (xp.arange(64, device=dev, dtype=xp.float32) - 31.5)
        super().__init__(
            xp,
            dev,
            radius=335.0,
            num_sides=56 if model == "blocks" else 504,
            num_lor_endpoints_per_side=8 if model == "blocks" else 1,
            lor_spacing=4.0 if model == "blocks" else 0.0,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )

    @property
    def model(self) -> str:
        """Detector model: ``blocks`` or ``cylindrical``."""
        return self._model


class SiemensVision600PETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Siemens Biograph Vision 600 endpoint geometry.

    ``model="blocks"`` (default) uses 38 flat faces with 20 physical
    crystals at an approximate 3.2 mm pitch, and 80 rings.
    ``model="cylindrical"`` uses STIR's 798 equally spaced positions,
    including 38 virtual positions per ring. In both cases the effective
    radius is 410 + 7 mm and the axial pitch is 3.29114 mm. The flat model
    approximates the physical block layout at a single interaction depth.
    ``model="castor_blocks"`` reproduces the dimensions of the public GATE
    Vision 600 model distributed in CASToR format: 3.2 mm axial pitch and
    an effective radius of 410 + 10 mm (crystal-centre interaction).

    Sources: https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12664451/, and
    https://zenodo.org/records/17649221/files/PET_SIEMENS_VISION600.geom.
    Coordinates are in mm. For the cylindrical model, the virtual mask places a placeholder
    at position 0 of each group of 21 in parallelproj's own convention. Its
    phase relative to a particular STIR/vendor dataset has not been verified.
    Neither model maps directly to STIR sinogram bins; STIR's view offset is
    not applied here.
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        symmetry_axis: int = 2,
        *,
        model: str = "blocks",
    ) -> None:
        if model not in ("blocks", "cylindrical", "castor_blocks"):
            raise ValueError("model must be 'blocks', 'cylindrical' or 'castor_blocks'")
        self._virtual_endpoint_period = 21 if model == "cylindrical" else None
        self._model = model
        rings = (3.2 if model == "castor_blocks" else 3.29114) * (
            xp.arange(80, device=dev, dtype=xp.float32) - 39.5
        )
        super().__init__(
            xp,
            dev,
            radius=420.0 if model == "castor_blocks" else 417.0,
            num_sides=798 if model == "cylindrical" else 38,
            num_lor_endpoints_per_side=1 if model == "cylindrical" else 20,
            lor_spacing=0.0 if model == "cylindrical" else 3.2,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )

    @property
    def model(self) -> str:
        """Detector model: ``blocks``, ``cylindrical`` or ``castor_blocks``."""
        return self._model


class GEDiscoveryMIPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """STIR cylindrical geometry for the GE Discovery MI scanner.

    The 3- and 4-block-ring PETRIC2 variants have 27 or 36 crystal rings,
    respectively, with 544 detector positions per ring. STIR's effective
    radius is 380.5 mm and its axial pitch is 5.52296 mm. The flat-block
    detector geometry cannot be inferred from these cylindrical headers.
    See https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx.
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        num_block_rings: int = 4,
        symmetry_axis: int = 2,
    ) -> None:
        if num_block_rings not in (3, 4, 5, 6):
            raise ValueError("num_block_rings must be 3, 4, 5, or 6")
        num_rings = 9 * num_block_rings
        rings = 5.52296 * (
            xp.arange(num_rings, device=dev, dtype=xp.float32) - (num_rings - 1) / 2
        )
        super().__init__(
            xp,
            dev,
            radius=380.5,
            num_sides=544,
            num_lor_endpoints_per_side=1,
            lor_spacing=0.0,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )


class GEDiscovery690PETScannerGeometry(RegularPolygonPETScannerGeometry):
    """STIR cylindrical geometry for the GE Discovery 690/710.

    STIR and PETRIC2 describe 24 rings, 576 positions per ring, an effective
    radius of 405.1 + 9.4 mm and an axial pitch of 6.54 mm. Flat-block
    crystal spacing is not specified by the PETRIC2 cylindrical header.
    ``model="castor_blocks"`` uses a published GATE model: 32 flat faces
    with 18 crystals at 4.2 mm pitch and 24 rings at 6.3 mm pitch. Its
    effective radius is 405.1 + 12.5 mm (crystal-centre interaction).
    See https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx.
    GATE model: https://github.com/zhaozhijun714/CASToR-Reconstruction-Toolkit-Source-Code/blob/ae9147a31869cb1441e35d719d7728c7e07ab3f8/config/scanner/GATE_GE_D690.geom
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        symmetry_axis: int = 2,
        *,
        model: str = "cylindrical",
    ) -> None:
        if model not in ("cylindrical", "castor_blocks"):
            raise ValueError("model must be 'cylindrical' or 'castor_blocks'")
        self._model = model
        rings = (6.3 if model == "castor_blocks" else 6.54) * (
            xp.arange(24, device=dev, dtype=xp.float32) - 11.5
        )
        super().__init__(
            xp,
            dev,
            radius=417.6 if model == "castor_blocks" else 414.5,
            num_sides=32 if model == "castor_blocks" else 576,
            num_lor_endpoints_per_side=18 if model == "castor_blocks" else 1,
            lor_spacing=4.2 if model == "castor_blocks" else 0.0,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )

    @property
    def model(self) -> str:
        """Detector model: ``cylindrical`` or ``castor_blocks``."""
        return self._model


class MedisoAnyScanPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Cylindrical geometry from PETRIC2's Mediso AnyScan SCP header.

    This model has 38 rings, 648 positions per ring, a 424 + 6 mm effective
    radius, and 3.9 mm ring spacing. The header describes a cylindrical
    reconstruction geometry and does not provide physical block spacing.
    See https://petric.tomography.stfc.ac.uk/2/data/files/Mediso_NEMA_IQ/prompts.hs.
    """

    def __init__(self, xp: ModuleType, dev: str, symmetry_axis: int = 2) -> None:
        rings = 3.9 * (xp.arange(38, device=dev, dtype=xp.float32) - 18.5)
        super().__init__(
            xp,
            dev,
            radius=430.0,
            num_sides=648,
            num_lor_endpoints_per_side=1,
            lor_spacing=0.0,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )


class NeuroLFPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """PETRIC2 NeuroLF 15 mm block geometry.

    Eight flat detector faces each have four 8-crystal blocks; six axial
    blocks of eight crystals form 48 rings. Both crystal pitches are 3.313 mm
    and both block pitches are 27.36 mm. The default inner radius of 135.5 mm
    and mean interaction depth of 8.26007 mm follow the Esser header and
    STIR example. The PETRIC2 Hoffman header instead gives 134 mm for the
    inner radius; use ``inner_radius=134.0`` for that geometry. All lengths
    are in mm. Endpoint numbering/orientation are parallelproj conventions.

    Source: https://github.com/UCL/STIR/blob/614b80c2762f2b270a8e80a793119113c18e6f31/examples/python/NeuroLF_reconstruction_example/stir_data_generation.py
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        symmetry_axis: int = 2,
        *,
        inner_radius: float = 135.5,
        average_depth_of_interaction: float = 8.26007,
    ) -> None:
        indices = xp.arange(48, device=dev)
        rings = 27.36 * xp.astype(indices // 8, xp.float32) + 3.313 * xp.astype(
            indices % 8, xp.float32
        )
        rings = rings - (5 * 27.36 + 7 * 3.313) / 2
        indices = xp.arange(32, device=dev)
        positions = 27.36 * xp.astype(indices // 8, xp.float32) + 3.313 * xp.astype(
            indices % 8, xp.float32
        )
        positions = positions - (3 * 27.36 + 7 * 3.313) / 2
        super().__init__(
            xp,
            dev,
            radius=inner_radius + average_depth_of_interaction,
            num_sides=8,
            lor_endpoint_positions=positions,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )


class SiemensMCTPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Siemens Biograph mCT TrueV endpoint geometry.

    ``model="blocks"`` (default) uses the CASToR-format GATE example's
    48 flat faces, 13 crystals per face, and four 13-ring axial blocks.
    Crystal pitches are 4.01 mm transaxially and 4.017462963 mm axially,
    with a 4.034925926 mm gap between axial blocks. The 437.2 mm endpoint
    radius is the example's 427.6 mm front radius plus 9.6 mm mean depth.
    ``model="cylindrical"`` uses STIR's 55 x 672 logical positions at a
    428 mm radius and 4.054 mm ring spacing, including virtual detector
    positions and three virtual rings. Masks use parallelproj conventions;
    their phase relative to a particular dataset must be checked.

    Sources: https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx
    and https://github.com/zhaozhijun714/CASToR-Reconstruction-Toolkit-Source-Code/blob/ae9147a31869cb1441e35d719d7728c7e07ab3f8/config/scanner/PET_SIEMENS_BIOGRAPH_MCT.geom
    """

    def __init__(
        self, xp: ModuleType, dev: str, symmetry_axis: int = 2, *, model: str = "blocks"
    ) -> None:
        if model not in ("blocks", "cylindrical"):
            raise ValueError("model must be 'blocks' or 'cylindrical'")
        self._model = model
        if model == "blocks":
            indices = xp.arange(52, device=dev)
            pitch = 4.017462963
            block_pitch = 13 * pitch + 4.034925926
            rings = block_pitch * xp.astype(
                indices // 13, xp.float32
            ) + pitch * xp.astype(indices % 13, xp.float32)
            rings = rings - (3 * block_pitch + 12 * pitch) / 2
        else:
            self._virtual_endpoint_period = 14
            self._virtual_ring_period = 14
            rings = 4.054 * (xp.arange(55, device=dev, dtype=xp.float32) - 27)
        super().__init__(
            xp,
            dev,
            radius=437.2 if model == "blocks" else 428.0,
            num_sides=48 if model == "blocks" else 672,
            num_lor_endpoints_per_side=13 if model == "blocks" else 1,
            lor_spacing=4.01 if model == "blocks" else 0.0,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )

    @property
    def model(self) -> str:
        """Detector model: ``blocks`` or ``cylindrical``."""
        return self._model


class SiemensVisionQuadraPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Siemens Biograph Vision Quadra endpoint geometry.

    ``model="blocks"`` (default) approximates 38 flat detector faces with
    20 crystals per side and four groups of 80 physical rings. The physical
    groups are separated by one crystal-pitch gap. ``model="cylindrical"``
    uses STIR's 323 x 798 logical positions, including three virtual axial
    rings and one virtual position per 21 transaxial positions. Both use
    STIR's effective radius 410 + 7 mm and axial pitch 3.29114 mm. The
    logical masks use parallelproj conventions and need dataset-specific
    verification; neither model encodes acquisition ring-difference limits.

    Sources: https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx
    and https://www.sciencedirect.com/science/article/pii/S0939388923001137.
    """

    def __init__(
        self, xp: ModuleType, dev: str, symmetry_axis: int = 2, *, model: str = "blocks"
    ) -> None:
        if model not in ("blocks", "cylindrical"):
            raise ValueError("model must be 'blocks' or 'cylindrical'")
        self._model = model
        if model == "blocks":
            indices = xp.arange(320, device=dev)
            logical_indices = indices + indices // 80
            rings = 3.29114 * (xp.astype(logical_indices, xp.float32) - 161)
        else:
            self._virtual_endpoint_period = 21
            self._virtual_ring_period = 81
            rings = 3.29114 * (xp.arange(323, device=dev, dtype=xp.float32) - 161)
        super().__init__(
            xp,
            dev,
            radius=417.0,
            num_sides=38 if model == "blocks" else 798,
            num_lor_endpoints_per_side=20 if model == "blocks" else 1,
            lor_spacing=3.2 if model == "blocks" else 0.0,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )

    @property
    def model(self) -> str:
        """Detector model: ``blocks`` or ``cylindrical``."""
        return self._model


class GESignaPETMRScannerGeometry(RegularPolygonPETScannerGeometry):
    """STIR cylindrical geometry of the GE SIGNA PET/MR.

    The single-interaction-depth model uses 45 rings, 448 positions per
    ring, a 311.8 + 8.5 mm radius and 5.56 mm ring spacing. It is not a
    physical block model and does not encode STIR's view offset.
    Source: https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx.
    """

    def __init__(self, xp: ModuleType, dev: str, symmetry_axis: int = 2) -> None:
        rings = 5.56 * (xp.arange(45, device=dev, dtype=xp.float32) - 22)
        super().__init__(
            xp,
            dev,
            radius=320.3,
            num_sides=448,
            num_lor_endpoints_per_side=1,
            lor_spacing=0.0,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )


class SiemensInveonPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Approximate flat-block Siemens Inveon preclinical scanner geometry.

    Models 16 faces of 20 crystals, 80 rings and one effective interaction
    depth. Transaxial and axial pitches are 1.63 and 1.59 mm; the effective
    radius is 80.5 + 4.58 mm. The four axial blocks are treated as contiguous.
    Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC2888286/ and
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2838767/.
    """

    def __init__(self, xp: ModuleType, dev: str, symmetry_axis: int = 2) -> None:
        rings = 1.59 * (xp.arange(80, device=dev, dtype=xp.float32) - 39.5)
        super().__init__(
            xp,
            dev,
            radius=85.08,
            num_sides=16,
            num_lor_endpoints_per_side=20,
            lor_spacing=1.63,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )


class PhilipsVereosPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Flat-block Philips Vereos model based on OpenGATE's simulation.

    Models 18 flat faces, 32 crystals per face and 40 rings. Each face has
    four groups of eight and the axial direction five groups of eight;
    groups are spaced by 32.85 mm and crystals by 4 mm. The endpoint plane
    is at the 391.5 mm crystal centre radius of the OpenGATE model. The
    orientation and endpoint order are parallelproj conventions.
    Source: https://github.com/OpenGATE/opengate/blob/master/opengate/contrib/pet/philipsvereos.py.
    """

    def __init__(self, xp: ModuleType, dev: str, symmetry_axis: int = 2) -> None:
        def positions(n):
            indices = xp.arange(n, device=dev)
            values = 32.85 * xp.astype(indices // 8, xp.float32) + 4.0 * xp.astype(
                indices % 8, xp.float32
            )
            return values - ((n // 8 - 1) * 32.85 + 28.0) / 2

        super().__init__(
            xp,
            dev,
            radius=391.5,
            num_sides=18,
            lor_endpoint_positions=positions(32),
            ring_positions=positions(40),
            symmetry_axis=symmetry_axis,
        )


class SiemensHRRTPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Single-depth cylindrical approximation to the Siemens ECAT HRRT.

    Uses STIR's 104 rings, 576 detector positions, 234.765 + 7 mm
    effective radius and 2.4375 mm axial spacing. The physical HRRT has
    two detector layers and an octagonal arrangement: neither is modelled
    here, and the STIR depth is an estimate. This preset cannot represent
    depth-of-interaction-specific detector pairs.
    Source: https://github.com/UCL/STIR/blob/rel_6.4.0/src/buildblock/Scanner.cxx.
    """

    def __init__(self, xp: ModuleType, dev: str, symmetry_axis: int = 2) -> None:
        rings = 2.4375 * (xp.arange(104, device=dev, dtype=xp.float32) - 51.5)
        super().__init__(
            xp,
            dev,
            radius=241.765,
            num_sides=576,
            num_lor_endpoints_per_side=1,
            lor_spacing=0.0,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )


class UnitedImagingUExplorerPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Approximate flat-block uEXPLORER total-body geometry.

    Models 24 flat faces with 35 crystals each and 672 rings (564480
    positions). There are eight axial units, each containing 14 blocks
    of six crystals. Crystals use a 2.85 mm pitch; within each unit the
    block-centre pitch is approximated as 240.2 / 14 mm and neighbouring
    units are separated by a 2.6 mm edge gap. The effective radius is
    393 + 9.05 mm (crystal centre). Transaxial sub-block gaps are not
    specified, so the 35 positions on each face use uniform pitch. This is
    a model of detector points, not a calibrated scanner map.

    Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC8729871/ and
    https://jnm.snmjournals.org/content/60/3/299.
    """

    def __init__(self, xp: ModuleType, dev: str, symmetry_axis: int = 2) -> None:
        indices = xp.arange(672, device=dev)
        unit = xp.astype(indices // 84, xp.float32)
        block = xp.astype((indices % 84) // 6, xp.float32)
        crystal = xp.astype(indices % 6, xp.float32)
        rings = unit * 242.8 + block * (240.2 / 14) + crystal * 2.85
        rings = rings - (7 * 242.8 + 13 * (240.2 / 14) + 5 * 2.85) / 2
        super().__init__(
            xp,
            dev,
            radius=402.05,
            num_sides=24,
            num_lor_endpoints_per_side=35,
            lor_spacing=2.85,
            ring_positions=rings,
            symmetry_axis=symmetry_axis,
        )
