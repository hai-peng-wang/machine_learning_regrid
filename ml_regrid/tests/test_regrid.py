from __future__ import division, print_function

import unittest

from itertools import product
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

from ml_regrid.iris_tools import (
    create_derive_cube, get_cube_grid_points, get_land_sea_index,
    transform_cube_by_masked_index, update_cube_with_new_data)

from ml_regird.tests.prepare_sample_for_test import (
    dpt_scn_aps3, lsm_aps2, lsm_aps3, precip_aps3,
    q_scn_aps3, sfc_prs_aps3, t_scn_aps3,
    topo_aps2, topo_aps3, u10_aps3, v10_aps3)

import numpy as npfrom numpy.testing import (
    assert_almost_equal, assert_array_equal, assert_raises)

from scipy.interpolate import RegularGridInterpolator


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
        self.topo_aps2 = topo_aps2
        self.topo_aps3 = topo_aps3
        self.lsm_aps2 = lsm_aps2
        self.lsm_aps3 = lsm_aps3
        self.t_scn_aps3 = t_scn_aps3
        self.sfc_prs_aps3 = sfc_prs_aps3
        self.precip_aps3 = precip_aps3
        self.dpt_scn_aps3 = dpt_scn_aps3
        self.q_scn_aps3 = q_scn_aps3

        # Note that u10 is sitting on different longitude (x)
        self.u10_aps3 = u10_aps3

        self.v10_aps3 = v10_aps3

        # For (x, y), it should be (lon, lat) accordingly
        self.x_tgt = self.topo_tgt.coord('longitude').points
        self.y_tgt = self.topo_tgt.coord('latitude').points
        self.x_src = self.topo_src.coord('longitude').points
        self.y_src = self.topo_src.coord('latitude').points

        # This Iris derived will be used in later
        self.drv_t_scn_iris = regrid_cube_by_scheme(
                                self.t_scn_src, self.topo_tgt,
                                scheme='linear')
        self.input_cubes = CubeList([])
        self.input_cubes.extend(
            [self.dpt_scn_src, self.sfc_prs_src, self.t_scn_src])

        self.in_grids = CubeList([])
        self.in_grids.extend([self.topo_src, self.lsm_src])
        self.out_grids = CubeList([])
        self.out_grids.extend([self.topo_tgt, self.lsm_tgt])

    def test_regrid_linear_iris(self):
        """
        Test the difference between IRIS and scipy linear regridding
        """
        t_scn_iris_data = self.drv_t_scn_iris.data
        # This interpolation is in the original form.
        # The "interp_by_scipy" is an updated function based on it.
        scipy_interp = RegularGridInterpolator(
            (self.x_src, self.y_src), self.t_scn_src.data, method='linear')

        t_scn_scipy_data = scipy_interp(
            list(product(self.x_tgt, self.y_tgt))).reshape(3, 3)

        self.assertAlmostEqual(t_scn_iris_data.any(),
                               t_scn_scipy_data.any())
        
    def test_regrid_linear_coastline_iris(self):
        """
        Test the hypothesis that IRIS regridding already applies the coastline
        inside its algorithms. Use scipy linear two stage regridding to prove.
        """
        # Calculate Scipy two stage data
        drv_t_scn_scipy = two_stage_interp(
            self.t_scn_src, self.topo_tgt, self.lsm_src, self.lsm_tgt)
        scipy_results = drv_t_scn_scipy.data
        iris_results = self.drv_t_scn_iris.data
        assert_almost_equal(iris_results, scipy_results, decimal=2)

    def test_t_scn_by_regrid_linear_iris_coastline_correction(self):
        """
        Testing the regrid_iris_coastline_correction
        """
        # Need to find the coastline points first
        # xv, yv = np.meshgrid(self.x_aps2, self.y_aps2)
        # tgrid = np.dstack((xv, yv))

        # weight = self.interpolator.compute_interp_weights(tgrid)
        # print(weight)
        # output_data = interpolator.interp_using_pre_computed_weights(weights)
        # Then add land/sea mask for coastline correction

        regridder = MdsRegridder(self.topo_aps3, self.topo_aps2,
                                 self.lsm_aps3, self.lsm_aps2)
        t_scn_by_iris_cc_list = \
            regridder._regrid_iris_coastline_correction(self.t_scn_aps3)
        [t_scn_by_iris_cc] = t_scn_by_iris_cc_list.extract('air_temperature')
        # Should be different with/without coastline correction
        # For the sample, only values at two mixed pnt are changed
        # Here 'Expected value' are values after the coastline correstion
        # Index  -/x   Delta     Target   Expected value   Actual Value
        # 1     x   0.031209  0.002000    300.695034      300.663825
        # 11     x   1.302968  0.002000    300.765625      299.462657
        self.show_difference(self.drv_t_scn_iris, t_scn_by_iris_cc,
                             entropy=1.334, accuracy=1.31)

    def test_t_scn_by_regrid_nearest_iris_coastline(self):
        """
        Test nearest regrid with coastline correction
        :return: Just show the difference between linear and nearest
        """
        # Get GFE results
        # drv_gfe_coastline_nearest = regrid_nearest_gfe_coastline_pigps(
        #    self.input_cubes, self.in_grids, self.out_grids)
        # drv_t_scn_gfe_coastline = extract_cube_by_name(
        #   drv_gfe_coastline_nearest, 'air_temperature')
        # Note: GFE results have 'nan' for the third column

        # Get Iris results
        drv_iris_coastline_nearest = regrid_nearest_iris_coastline(
            self.input_cubes, self.in_grids, self.out_grids)
        drv_t_scn_iris_coastline = extract_cube_by_name(
            drv_iris_coastline_nearest, 'air_temperature')

        self.show_difference(self.drv_t_scn_iris, drv_t_scn_iris_coastline,
                             entropy=7.050, accuracy=1.7)
	
    def show_difference(self, cube_a, cube_b, entropy=None, accuracy=None):
        """
        Show the difference between two data sets.
        :param cube_a: a cube with the examing data set
        :param cube_b: a cube with expected results
        :param accuracy: measurement for each points
        :param entropy: sum of total delta
        :return: show difference if fails; otherwise, test passes
        """
        result = cube_a.data.flatten()
        expected = cube_b.data.flatten()

        accuracy = np.repeat(accuracy, len(result))

        if accuracy is None:
            accuracy = np.repeat(0.001, len(result))
        if entropy is None:
            entropy = 0.02
        # Now assert the data is as expected.
        delta = np.abs(result - expected)
        msg = ('\nEntropy: {}\nExpected entropy: {}\n'
               'Index  -/x   Delta     Target   Expected value   Actual Value'
               ''.format(delta.sum(), entropy))
        template = '\n{0:3}     {1:2}  {2:6f}  {3:6f}    {4:6f}      {5:6f}'

        for i, (r_del, t_del, r, t) in enumerate(zip(delta, accuracy,
                                                     result, expected)):
            msg += template.format(i, '-' if r_del < t_del else 'x',
                                   r_del, t_del,
                                   t, r)
        # Ensure each accuracy component falls below the target.
        assert_array_less(delta, accuracy, msg)
        # Ensure that our entropy is close to the expected one.
        # If this fails because the result is now smaller than expected, good!
        # It means we can tighten the expected entropy *AND* the target delta.
        if np.abs(entropy - delta.sum()) > 0.001:
            self.fail(msg)
	
	
if __name__ == '__main__':
    unittest.main()

