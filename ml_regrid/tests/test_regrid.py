from __future__ import division, print_function

import unittest

from itertools import product
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from ml_regrid.iris_tools import (
    create_derive_cube, get_cube_grid_points, get_land_sea_index,
    transform_cube_by_masked_index, update_cube_with_new_data)

import numpy as npfrom numpy.testing import (
    assert_almost_equal, assert_array_equal, assert_raises)

from scipy.interpolate import RegularGridInterpolator

lsm_aps2_data = np.array(
        [[1., 0., 0.],
         [1., 1., 0.],
         [1., 1., 0.]])

surface_alt_aps2_data = np.array(
        [[40., 0., 0.],
         [272.5, 49.5, 0.],
         [569., 139.375, 0.]])

# For APS3, the index points lsm_aps3.data[821:827, 461:467], shape=(6, 6)
lsm_aps3_data = np.array(
      [[1, 1, 0, 0, 0, 0],
       [1, 1, 1, 0, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0, 0]])

surface_alt_aps3_data = np.array(
        [[30.79901695, 24.64077759, 0., 0., 0., 0.],
         [40.51713181, 57.62937927, 44.1620636, 0., 0., 0.],
         [61.04179001, 77.5438385, 68.0868454, 27.50967216, 0., 0.],
         [197.31484985, 109.88461304, 81.08721161, 47.29803085, 0., 0.],
         [500.52633667, 204.61886597, 96.57264709, 57.00541687, 0., 0.],
         [801.8571167, 337.81097412, 131.99372864, 66.28527832, 0., 0.]])

t_scn_aps3_data = np.array(
     [[300.0625, 299.1875, 300.859375, 300.4375, 300.640625, 300.8125],
      [300.734375, 303.546875, 300.828125, 300.53125, 300.578125, 300.796875],
      [302.0625, 301.078125, 300.703125, 299.90625, 300.640625, 300.71875],
      [300.796875, 299.890625, 299.703125, 300.046875, 300.71875, 300.65625],
      [297.078125, 298.875, 299.09375, 300.078125, 300.71875, 300.65625],
      [296.125, 297.875, 298.5625, 299.328125, 300.59375, 300.734375]])

sfc_prs_aps3_data = np.array(
      [[1212528., 1212361., 1212363., 1212647., 1212919., 1213073.],
       [1212195., 1211922., 1212181., 1212605., 1212950., 1213131.],
       [1212043., 1212038., 1212174., 1212588., 1212954., 1213181.],
       [1212325., 1212124., 1212306., 1212599., 1212949., 1213226.],
       [1212454., 1212399., 1212434., 1212652., 1213012., 1213305.],
       [1212653., 1212496., 1212555., 1212774., 1213106., 1213395.]])

dpt_scn_aps3_data = np.array(
    [[293.515625, 294.03125, 294.5625, 294.53125, 294.171875, 294.],
     [293.1875, 290.59375, 292.15625, 294.078125, 294.09375, 293.6875],
     [292.59375, 291.71875, 291.390625, 291.890625, 293.828125, 293.640625],
     [291.25, 291.8125, 291.296875, 291.328125, 293.65625, 293.515625],
     [292.03125, 291.734375, 291.515625, 291.015625, 293.296875, 293.359375],
     [292.328125, 291.4375, 291.875, 291.3125, 292.90625, 293.171875]])


def empty_3d_cube_aps3(data, name=None, unit=None, stash=None, **kwargs):
    """
    Prepare some iris cubes at APS3 grids for testing
    """
    if data is None:
        data = np.empty([6, 6])

    cube = Cube(data)

    time = AuxCoord([0], 'time', units='hours since epoch')

    latitude = DimCoord([6.26953125, 6.38671875, 6.50390625,
                         6.62109375, 6.73828125, 6.85546875],
                       	standard_name='latitude', units='degrees')

    longitude = DimCoord([81.12304688, 81.29882812, 81.47460938,
                          81.65039062, 81.82617188, 82.00195312],
                         standard_name='longitude', units='degrees')
    cube.add_dim_coord(latitude, 0)
    cube.add_dim_coord(longitude, 1)
    cube.add_aux_coord(time)

    if name:
        cube.long_name = name
    if unit:
        cube.units = unit
    if stash:
        cube.attributes['STASH'] = stash

    return cube

def empty_3d_cube_aps3(data, name=None, unit=None, stash=None, **kwargs):
    """
    Prepare some iris cubes at APS3 grids for testing
    """
    if data is None:
        data = np.empty([6, 6])

    cube = Cube(data)

    time = AuxCoord([0], 'time', units='hours since epoch')

    latitude = DimCoord([6.26953125, 6.38671875, 6.50390625,
                         6.62109375, 6.73828125, 6.85546875],
                       	standard_name='latitude', units='degrees')

    longitude = DimCoord([81.12304688, 81.29882812, 81.47460938,
                          81.65039062, 81.82617188, 82.00195312],
                         standard_name='longitude', units='degrees')
    cube.add_dim_coord(latitude, 0)
    cube.add_dim_coord(longitude, 1)
    cube.add_aux_coord(time)

    if name:
        cube.long_name = name
    if unit:
        cube.units = unit
    if stash:
        cube.attributes['STASH'] = stash

    return cube

def regrid_cube_by_scheme(param_cube, target_cube, scheme=None):
    """
    Use Area-weighted Scheme to regrid a specific cube from a cube list.
    ------
    Input:
	param_cube: a cube for the specified paramerter;
        target_cube: a cube with targetted grid
        scheme: an iris scheme, i.e. 'linear', 'nearest', 'area_weighted'
    Output:
	drv_cube: the regridded cube (derived).
    """
    # claim the scheme
    if scheme:
	if scheme == 'linear':
            regrid_scheme = iris.analysis.Linear()
        elif scheme == 'nearest':
            regrid_scheme = iris.analysis.Nearest()
        elif scheme == 'area_weighted':
            regrid_scheme = iris.analysis.AreaWeighted()
        else:
            raise ValueError(
                "The input scheme is not an option! \
                 Only 'linear', 'nearest', 'area_weighted' are available.")
    else:
	raise ValueError("Need input a scheme!")

    # Besure that the horizontal grid coordinates of both the source and
    # grid cubes must have contiguous bounds.
    besure_cube_has_continuous_bounds(param_cube)
    besure_cube_has_continuous_bounds(target_cube)

    # Use the given scheme to regrid
    drv_cube = param_cube.regrid(target_cube, regrid_scheme)
    return drv_cube


class TestMdsRegrid(unittest.TestCase):
    def setUp(self):
        # Create sample cubes for testing
        self.topo_aps2 = empty_3d_cube_aps2(
            surface_alt_aps2_data, 'surface_altitude', 'm', 'm01s00i033')
        self.topo_aps3 = empty_3d_cube_aps3(
            surface_alt_aps3_data, 'surface_altitude', 'm', 'm01s00i033')
        self.lsm_aps2 = empty_3d_cube_aps2(
            lsm_aps2_data, 'land_area_fraction', '1', )
        self.lsm_aps3 = empty_3d_cube_aps3(
            lsm_aps3_data, 'land_binary_mask', '1')
        self.t_scn_aps3 = empty_3d_cube_aps3(
            t_scn_aps3_data, 'air_temperature', 'K')
        self.sfc_prs_aps3 = empty_3d_cube_aps3(
            sfc_prs_aps3_data, 'air_pressure_at_sea_level', 'Pa')
        self.dpt_scn_aps3 = empty_3d_cube_aps3(
            dpt_scn_aps3_data, 'dew_point_temperature', 'K', 'm01s03i250')

        # For (x, y), it should be (lon, lat) accordingly
        self.x_aps2 = self.topo_aps2.coord('longitude').points
        self.y_aps2 = self.topo_aps2.coord('latitude').points
        self.x_aps3 = self.topo_aps3.coord('longitude').points
        self.y_aps3 = self.topo_aps3.coord('latitude').points

        # This Iris derived will be used in later
        self.drv_t_scn_iris = regrid_cube_by_scheme(
                                self.t_scn_aps3, self.topo_aps2,
                                scheme='linear')
        self.input_cubes = CubeList([])
        self.input_cubes.extend(
            [self.dpt_scn_aps3, self.sfc_prs_aps3, self.t_scn_aps3])

        self.in_grids = CubeList([])
        self.in_grids.extend([self.topo_aps3, self.lsm_aps3])
        self.out_grids = CubeList([])
        self.out_grids.extend([self.topo_aps2, self.lsm_aps2])

    def test_regrid_linear_iris(self):
        """
        Test the difference between IRIS and scipy linear regridding
        """
        t_scn_iris_data = self.drv_t_scn_iris.data
        # This interpolation is in the original form.
        # The "interp_by_scipy" is an updated function based on it.
        scipy_interp = RegularGridInterpolator(
            (self.x_aps3, self.y_aps3), self.t_scn_aps3.data, method='linear')

        t_scn_scipy_data = scipy_interp(
            list(product(self.x_aps2, self.y_aps2))).reshape(3, 3)

        self.assertAlmostEqual(t_scn_iris_data.any(),
                               t_scn_scipy_data.any())
        
    def test_regrid_linear_coastline_iris(self):
        """
        Test the hypothesis that IRIS regridding already applies the coastline
        inside its algorithms. Use scipy linear two stage regridding to prove.
        """
        # Calculate Scipy two stage data
        drv_t_scn_scipy = two_stage_interp(
            self.t_scn_aps3, self.topo_aps2, self.lsm_aps3, self.lsm_aps2)
        scipy_results = drv_t_scn_scipy.data
        iris_results = self.drv_t_scn_iris.data
        assert_almost_equal(iris_results, scipy_results, decimal=2)
