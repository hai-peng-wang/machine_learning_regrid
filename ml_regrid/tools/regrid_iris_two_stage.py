from __future__ import division, print_function

import iris

from iris.analysis._scipy_interpolate import _RegularGridInterpolator

import numpy as np

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
    
def regrid_iris_two_stage(
        cube_src, cube_tgt, lsm_src, lsm_tgt, algorithm='linear'):
    """
    Do two stages interpolation (land & sea) separately and then
    integrate the results together. This is working for area-weighted.
    ------
    Input:
        cube_src: the cube with source grid;
        cube_tgt: the cube with target grid;
        lsm_src: land/sea mask cube from source grids
        lsm_src: land/sea mask cube from target grids
        algorithm: iris regrid algorithms, which could be
                'linear', 'nearest' and 'area-weighted'.
    Output:
        drv_cube_land_aps2: an updated regridded cube with integrated data
                            of both land and sea although named by land.
    """
    iris_regrid_algorithm = {'linear': iris.analysis.Linear(),
                             'nearest': iris.analysis.Nearest(),
                             'area-weighted': iris.analysis.AreaWeighted()}

    # For area-weighted, both the source and grid cubes must
    # have contiguous bounds
    if algorithm == 'area-weighted':
        besure_cube_has_continuous_bounds(cube_src)
        besure_cube_has_continuous_bounds(cube_tgt)

    # Get the land and sea index for both grids
    land_src_indx, sea_src_indx = get_land_sea_index(lsm_src)
    land_tgt_indx, sea_tgt_indx = get_land_sea_index(lsm_tgt)

    # Mask the APS3 sea points
    cube_land_src = transform_cube_by_masked_index(cube_src, land_src_indx)
    # Mask the APS3 land points
    cube_sea_src = transform_cube_by_masked_index(cube_src, sea_src_indx)

    # Derive the cube_tgt from src, and tests on smale samples found that
    # using the 'cube_tgt' will be the same as using 'cube_tgt_land'
    drv_cube_land_tgt = cube_land_src.regrid(
        cube_tgt, iris_regrid_algorithm[algorithm])
    drv_cube_sea_tgt = cube_sea_src.regrid(
        cube_tgt, iris_regrid_algorithm[algorithm])

    # Combine the land & sea data together
    combined_data = drv_cube_land_tgt.data
    combined_data = np.where(sea_tgt_indx,
                             drv_cube_sea_tgt.data, combined_data)
    # update the cube with integrated data
    drv_cube_land_tgt.data = combined_data
    # return the updated cube (land) although the name is confusing
    return drv_cube_land_tgt


class IrisRegridder(object):
    """
    Regrid the cubes with a range of optional algorithms.
    """
    def __init__(self, src_topo, tgt_topo,
                 src_lsm=None, tgt_lsm=None):
        """
        src_topo: a cube of source topograph (required)
        tgt_topo: a cube of target topograph (required)
        src_lsm: a cube of source land binary mask (optional)
        tgt_lsm: a cube of target land binary mask (optional)
        """
        self.src_topo = src_topo
        self.tgt_topo = tgt_topo

        # calculate dz cube for height adjustment
        self.drv_tgt_topo = regrid_cube_by_scheme(
                                self.src_topo, self.tgt_topo, 'linear')
        self.dz = self.tgt_topo - self.drv_tgt_topo

        if src_lsm and tgt_lsm:
            # Providing both lsm are available
            self.src_lsm = src_lsm
            self.tgt_lsm = tgt_lsm
            self.src_lsm_land_indx, self.src_lsm_sea_indx = \
                get_land_sea_index(self.src_lsm)
            self.tgt_lsm_land_indx, self.tgt_lsm_sea_indx = \
                get_land_sea_index(self.tgt_lsm)

    def _regrid(self, input_cubes, algorithm='linear'):
        """
        Method can be overridden to provide regridding of input cubes.
        ------
        Args:
            input_cubes: the source input cubes
            algorithm: a string descrbing the regridding scheme used,
                       i.e. "linear", "nearest", or "area_weighted"
        Returns:
            The interpolated cube with the target grid.
        """
        # If the input is only a param_cube, turn it into CubeList
        if not isinstance(input_cubes, iris.cube.CubeList):
            input_cube_list = iris.cube.CubeList([])
            input_cube_list.append(input_cubes)
            input_cubes = input_cube_list

        regridded_cubes = iris.cube.CubeList([])

        for cube in input_cubes:
            full_log.info("Processing " + cube.name())
            # To conserve the quantities when regridding
            drv_cube = regrid_cube_by_scheme(cube, self.tgt_topo,
                                             scheme=algorithm)
            regridded_cubes.append(drv_cube)

        return regridded_cubes

    def _regrid_iris_coastline_correction(
            self, input_cubes, algorithm='linear'):
        """
        Use iris.analysis._interpolate for coastline correction and
        then combine with Iris.regrid;
        For 'area-weighted', use '_regrid_iris_two_stage' for 'cc'.
         ------
        Args:
            input_cubes: the source input cubes
            algorithm: a string descrbing the regridding scheme used,
                       i.e. only "linear" and "nearest"
        Returns:
            The interpolated cube with coastline correction.
        """
        # If the input is only a param_cube, turn it into CubeList
        if not isinstance(input_cubes, iris.cube.CubeList):
            input_cube_list = iris.cube.CubeList([])
            input_cube_list.append(input_cubes)
            input_cubes = input_cube_list

        regridded_cubes = iris.cube.CubeList([])

        if self.src_lsm is None or self.tgt_lsm is None:
            raise ValueError("Need land/sea mask to initialize IrisRegridder!")

        for cube in input_cubes:
            # Masked the source sea points
            cube_src_land = transform_cube_by_masked_index(
                cube, self.src_lsm_land_indx)
            # Masked the source land points
            cube_src_sea = transform_cube_by_masked_index(
                cube, self.src_lsm_sea_indx)

            lat_src, lon_src = get_cube_grid_points(src_topo)
            lat_tgt, lon_tgt = get_cube_grid_points(tgt_topo)
            # Define the interpolator for land and sea
            # Note: the grid_points need to be (x_points, y_points)
            # And x_points = longitude_points, y_points = latitude_pnt
            interpolator_land = _RegularGridInterpolator(
                (lon_src, lat_src), cube_src_land.data, method=algorithm,
                bounds_error=False, fill_value=None)
            interpolator_sea = _RegularGridInterpolator(
                (lon_src, lat_src), cube_src_sea.data, method=algorithm,
                bounds_error=False, fill_value=None)
            # Define the target grid
            xv, yv = np.meshgrid(lat_tgt, lon_tgt)
            tgrid_tgt = np.dstack((yv, xv))

            # Calc the tgrid with land/sea index
            # By purposely using lsm for tgrid didn't make difference
            # tgrid_tgt_land = _create_masked_tgrid(
            #    self.tgt_lsm_land_indx, tgrid_tgt)
            # tgrid_tgt_sea = _create_masked_tgrid(
            #    self.tgt_lsm_sea_indx, tgrid_tgt)

            # interp by (masked) tgrid with land/sea
            output_data_land = _interp_masked_grid(
                interpolator_land, tgrid_tgt)
            output_data_sea = self._interp_masked_grid(
                interpolator_sea, tgrid_tgt)

            combined_data = np.where(self.tgt_lsm_land_indx,
                                     output_data_sea, output_data_land)

            drv_cube = create_derive_cube(cube, self.tgt_topo)
            drv_cube.data = combined_data

            regridded_cubes.append(drv_cube)

        return regridded_cubes

    def _create_masked_tgrid(self, masked_index, tgrid):
        """
        Create the target grid with masked index.
        :param masked_index: a land or sea mask
        :param tgrid: a grid from target [x_tgt, y_tgt]
        :return: a grid with mask
        """
        row_mask = masked_index
        # column stack
        new_mask = np.stack((row_mask, row_mask), -1)
        masked_tgrid = np.ma.masked_array(tgrid, mask=new_mask)
        return masked_tgrid

    def _interp_masked_grid(self, interpolator, tgrid):
        """
        Interpolate to masked_tgrid with weight_index.
        :param interpolator: a masked scipy interpolator
        :param tgrid: a target grid (can be masked or not)
        :return: interpolated data array
        """
        weight_index = interpolator.compute_interp_weights(tgrid)
        output_data = \
            interpolator.interp_using_pre_computed_weights(weight_index)
        return output_data
