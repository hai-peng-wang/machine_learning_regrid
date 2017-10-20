from __future__ import division, print_function

import numpy as np

import iris

from iris.analysis._scipy_interpolate import _RegularGridInterpolator

from ml_regrid.tools.iris_tools import (
    besure_cube_has_continuous_bounds, get_land_sea_index,
    produce_land_sea_bool, transform_cube_by_masked_index)

import scipy.spatial.qhull as qhull
from scipy.interpolate import griddata


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
        drv_cube_land_tgt: an updated regridded cube with integrated data
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

    # Mask the src sea points
    cube_land_src = transform_cube_by_masked_index(cube_src, land_src_indx)
    # Mask the src land points
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


def interp_weights(xy, uv, d=2):
    """
    Interpolate source grid (xy) to target grid (uv).
    This method will speedup scipy griddata (working for 2d).
    -----
    Input:
        xy: the source grid with shape (x*y, 2);
        uv: the target grid with shape (u*v, 2.
    Output:
        vertices: the vertices of the enclosing simplex;
        weights: the weights for the interpolation
    """
    # 1.Triangulate the irregular grid coordinates
    tri = qhull.Delaunay(xy)
    # 2.For each point in the new grid, the triangulation is searched to
    # find in which triangle does it lay
    simplex = tri.find_simplex(uv)
    # 3.The barycentric coordinates of each new grid point with respect
    # to the vertices of the enclosing simplex are computed
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    
    # The first three steps are identical for all interpolations, so if stored
    # for each new grid point, the indices of the vertices of the enclosing
    # simplex and the weights for the interpolation, one would minimize the
    # amount of computations by a lot.
    return vertices, np.hstack((bary, 1-bary.sum(axis=1, keepdims=True)))


def pad(data):
    """
    Interpolate NaN values in a numpy array.
    """
    # building on Winston's answer
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(
        bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data


def interpolate(values, vtx, wts, fill_value=np.nan):
    """
    The last step to interpolate (replace the griddata), which is per case.
    ------
    Input:
        values: the values on source grids;
        vtx:  the vertices of the enclosing simplex;
        wts: the weights for the interpolation
    Output:
        the flattened numpy array, which needs reshape(len(lon_tgt), len(lat))
    """
    # 4. An interpolated values is computed for that grid point, using the
    # barycentric coordinates, and the values of the function at the vertices
    # of the enclosing simplex.
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    # There're cases of wts < 0 unfortunately, fill with nan
    ret[np.any(wts < 0, axis=1)] = fill_value
    # Interpolate a few 'nan'
    ret = pad(ret)
    return ret


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
        # Prepare the lat, lon
        self.lat_src, self.lon_src = get_cube_grid_points(self.topo_src)
        self.lat_tgt, self.lon_tgt = get_cube_grid_points(self.topo_tgt)
        # Define the source/target grid
        xv, yv = np.meshgrid(self.lon_src, self.lat_src)
        self.grid_src = np.dstack((yv, xv))
        xv, yv = np.meshgrid(self.lon_tgt, self.lat_tgt)
        self.grid_tgt = np.dstack((yv, xv))

        # calculate dz cube for height adjustment
        self.drv_tgt_topo = regrid_cube_by_scheme(
                                self.src_topo, self.tgt_topo, 'linear')
        self.dz = self.tgt_topo - self.drv_tgt_topo

        if src_lsm and tgt_lsm:
            # Providing both lsm are available
            self.src_lsm = src_lsm
            self.tgt_lsm = tgt_lsm
            # Need True/False index for each Land/Sea point
            self.lsm_land_bool_src, self.lsm_sea_bool_src = \
                produce_land_sea_bool(self.lsm_src)
            self.lsm_land_bool_tgt, self.lsm_sea_bool_tgt = \
                produce_land_sea_bool(self.lsm_tgt)
                
            # Prepare the coastline index for correction to speedup
            [X, Y] = np.meshgrid(self.lon_src, self.lat_src)
            [Xi, Yi] = np.meshgrid(self.lon_tgt, self.lat_tgt)

            # The source grid need for CC interpolation
            xy = np.zeros([X.shape[0] * X.shape[1], 2])
            xy[:, 0] = Y.flatten()
            xy[:, 1] = X.flatten()

            # The tgt grid need for CC interpolation
            uv = np.zeros([Xi.shape[0] * Xi.shape[1], 2])
            uv[:, 0] = Yi.flatten()
            uv[:, 1] = Xi.flatten()

            # If only use uv coastline pnt
            # Prepare some index for later CC
            # Here we'd just use default 'linear' to interp for CC
            combined_data = self._regrid_cube_by_lsm(self.lsm_src)
            self.coast_pnt_bool = np.isnan(combined_data)# The reshape can be a better way?
            uv_coast_pnt = uv[new_index].reshape(int(len(uv[new_index])/2), 2)

            # Calculate the vertices and weights
            self.vtx, self.wts = interp_weights(xy, uv_coast_pnt)

            row_index = self.coast_pnt_bool
            new_index = np.stack((row_index, row_index), -1)
            new_index = new_index.flatten().reshape(
                Xi.shape[0] * Xi.shape[1], 2)
            # The reshape can be a better way?
            uv_coast_pnt = uv[new_index].reshape(int(len(uv[new_index])/2), 2)

            # Calculate the vertices and weights
            self.vtx, self.wts = interp_weights(xy, uv_coast_pnt)
      
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
    
    def _regrid_cube_by_lsm(self, cube, algorithm='linear'):
        """
        Regrid the input cube by the land/sea index;
        Inside, the coastline points will be specified as nan.
        --------
        Input:
            cube: a cube for regridding;
            algorithm: regridding algorithms used.
        Output:
            combined_data: a regridded data array with nan for next
                           steop coastline correction.
        """
        # We do some calculation about the land/sea index and
        # coastline points which can be used for all parameters
        # being regridded
        cube_src_sea_data = np.where(
            (self.lsm_src.data == 0), cube.data, np.nan)
        cube_src_land_data = np.where(
            (self.lsm_src.data != 0), cube.data, np.nan)

        cube_src_land = cube.copy()
        cube_src_land.data = cube_src_land_data

        cube_src_sea = cube.copy()
        cube_src_sea.data = cube_src_sea_data

        cube_tgt_land = regrid_cube_by_scheme(
            cube_src_land, self.topo_tgt, scheme=algorithm)
        cube_tgt_sea = regrid_cube_by_scheme(
            cube_src_sea, self.topo_tgt, scheme=algorithm)

        combined_data = np.where(
            (self.lsm_tgt.data == 0), cube_tgt_sea.data, cube_tgt_land.data)
        return combined_data


    def _regrid_iris_coastline_correction(
            self, input_cubes, algorithm='linear'):
        """
        Use Scipy for coastline correction; the vertices and weights for
        coastline points are calculated by 'interp_weights' only once;
        then interpolate parameter data wise to replace the purposely
        placed 'nan' values in the initial regridded results by Iris.
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

        if self.lsm_src is None or self.lsm_tgt is None:
            raise ValueError("Need land/sea mask to initialize MdsRegridder!")

        for cube in input_cubes:
            # Need check if the input cube has the same shape of grid
            # For example, surface_downward_northward_stress has (1537, 2048)
            # compared to lsm_src (1536, 2048); then Do Regrid first
            if cube.shape != self.lsm_src.shape:
                cube = cube.regrid(self.lsm_src, iris.analysis.Linear())

            # If there are 'nan' in cube.data, don't do CC since there are
            # parameters with lots of 'nan', which shouldn't be interpolated
            # i.e. "Air_pressure_at_convective_cloud_top"
            if (np.isnan(cube.data.sum()) or
                    cube.name() in coastline_correction_exclude_list):
                # It seems that 'nearest' is interpolating better for
                # those params falling in this category
                drv_cube = regrid_cube_by_scheme(cube, self.topo_tgt,
                                                 scheme='nearest')
                regridded_cubes.append(drv_cube)
            # Do "cc" to cube
            else:
                # Get the combined the data with nan
                combined_data = self._regrid_cube_by_lsm(cube, algorithm)

                # All the 'nan' values are coastline points to be interpolated
                # The first interpolation is using 'np.einnum' to speedup
                output_coast_values = interpolate(
                        cube.data, self.vtx, self.wts)

                combined_data[self.coast_pnt_bool] = output_coast_values

                # Occasionally, there could be 'nan'; pad it.
                if np.isnan(combined_data.sum()):
                    combined_data = pad(combined_data)

                # This will just create a dummy regridded cube with all
                # attributes plus cell_methods if exits
                drv_cube = create_derive_cube(cube, self.topo_tgt)
                drv_cube.data = combined_data

                regridded_cubes.append(drv_cube)

        return regridded_cubes
    
     def _regrid_iris_coastline_correction_deprecated(
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
        # kafka_log, full_log = getLoggers()
        # If the input is only a param_cube, turn it into CubeList
        if not isinstance(input_cubes, iris.cube.CubeList):
            input_cube_list = iris.cube.CubeList([])
            input_cube_list.append(input_cubes)
            input_cubes = input_cube_list

        regridded_cubes = iris.cube.CubeList([])

        if self.lsm_src is None or self.lsm_tgt is None:
            raise ValueError("Need land/sea mask to initialize MdsRegridder!")

        for cube in input_cubes:
            full_log.info("Processing " + cube.name())
            # Need check if the input cube has the same shape of grid
            if cube.shape != self.lsm_src.shape:
                cube = cube.regrid(self.lsm_src, iris.analysis.Linear())
            # Separate land/sea data
            cube_src_land_data = np.where(
                (self.lsm_src.data != 0), cube.data, np.nan)
            cube_src_sea_data = np.where(
                (self.lsm_src.data == 0), cube.data, np.nan)

            # Prepare the interpolator
            # We need to extropolate the points outside bounds
            # So set bounds_error=False, fill_value=None
            interpolator_land = _RegularGridInterpolator(
                (self.lat_src, self.lon_src), cube_src_land_data,
                method=algorithm, bounds_error=False, fill_value=None)
            interpolator_sea = _RegularGridInterpolator(
                (self.lat_src, self.lon_src), cube_src_sea_data,
                method=algorithm, bounds_error=False, fill_value=None)

            # Interpolate separately by land/sea
            output_data_land = self._interp_masked_grid(
                interpolator_land, self.grid_tgt)
            output_data_sea = self._interp_masked_grid(
                interpolator_sea, self.grid_tgt)
            # Combine the data
            combined_data = np.where(
                (self.lsm_tgt.data == 0), output_data_sea, output_data_land)

            # Since the combined_data has 'nan', where it is a point
            # along the coastline. Find it and do correction
            coast_pnt_bool = np.isnan(combined_data)
            
            # Now retrieving the points being both coastline and land
            coast_pnt_bool_land = np.logical_and(
                coast_pnt_bool, self.lsm_land_bool_tgt)
            coast_points_land = self.grid_tgt[coast_pnt_bool_land]
            # The way to pass an array of coast points to
            # 'latlons' is much faster.
            out_value_land = self._regrid_coastline_pnt(
                cube, self.grid_src, coast_points_land,
                self.lsm_land_bool_src, algorithm)

            # Do the same with sea
            coast_pnt_bool_sea = np.logical_and(
                coast_pnt_bool, self.lsm_sea_bool_tgt)
            coast_points_sea = self.grid_tgt[coast_pnt_bool_sea]
            # More works (using Cython) need to optimize the function
            out_value_sea = self._regrid_coastline_pnt(
                cube, self.grid_src, coast_points_sea,
                self.lsm_sea_bool_src, algorithm)

            # Now need to replace those coastline points with value nan
            combined_data[coast_pnt_bool_land] = out_value_land
            combined_data[coast_pnt_bool_sea] = out_value_sea
            
            drv_cube = create_derive_cube(cube, self.topo_tgt)
            drv_cube.data = combined_data

            regridded_cubes.append(drv_cube)

        return regridded_cubes

    def _regrid_coastline_pnt(
            self, cube, grid, latlons, lsm_indx, algorithm='linear'):
        """
        Regrid the input point (latlons) on a grid
        :param cube: input source cube
        :param grid: the target grid
        :param latlons: the array of input point, each pnt [lat, lon]
        :param lsm_indx: the bool index from land/sea mask
        :param algorithm: the used algorithm, i.e. 'linear' or 'nearest'
        Note: For griddata, method can be above two or 'cubic'
        :return: the regridded value on the input point
        """
        # We do not want it be done twice for 'nearest'
        if algorithm == 'nearest':
            output_pnt = griddata(
                grid[lsm_indx], cube.data[lsm_indx], latlons, method=algorithm)
        elif algorithm == 'linear':
            # Generally, derived data will be valid if there are three points
            output_pnt = griddata(
                grid[lsm_indx], cube.data[lsm_indx], latlons, method=algorithm)

            # There are scenario like only two or one sea points nearby, then
            # the output is 'nan'; use 'nearest' specifically like GFE does
            # if output_pnt == np.nan: (this doesn't work)
            # The array.sum() will do the faster check on loop
            if np.isnan(output_pnt.sum()):
                # First create a 'nan' index
                output_pnt_is_nan = np.isnan(output_pnt)
                # Use the index for interp
                output_pnt_nan = griddata(
                    grid[lsm_indx], cube.data[lsm_indx],
                    latlons[output_pnt_is_nan], method='nearest')
                # Replace the nan
                output_pnt[output_pnt_is_nan] = output_pnt_nan
        else:
            print('{} is not a supported method for griddata'.format(
                  algorithm))
        return output_pnt

    def _closest_points_index(self, point, grid):
        """
        Produce the bool index for node (lat, lon) on grid nodes
        :param point: a grid point [lat, lon] not on the grid
        :param grid: a source grid
        :return: the bool value (index) on those nearest points to the point
        """
        deltas = np.abs(grid - point)
        # the grid has shape (i, j, k) i.e. (8, 6, 2) for sample testing area
        dist_2 = np.einsum('ijk,ijk->ij', deltas, deltas)
        # Index be true if the distance is minimum
        index_bool = (dist_2 == np.min(dist_2))
        return index_bool
  
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
