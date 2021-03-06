from __future__ import (absolute_


import, division, print_function)

import numpy as np


def besure_cube_has_continuous_bounds(cube):
    """
    Make sure that grid coordinates of both the source and grid cubes
    must have contiguous bounds for the area-weighted regridding.
    """
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()
    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    return cube


def get_land_sea_index(lsm_cube):
    """
    Return the land_indx and sea_indx from a given lsm_cube.
    """
    # We like the masked points to be false
    sea_index = (lsm_cube.data != 0)
    land_index = (lsm_cube.data == 0)
    return land_index, sea_index


def produce_land_sea_index(lsm_cube):
    """
    Return the land_indx and sea_indx from a given lsm_cube.
    Note: this method is not for mask; but for true value to be reserved
    """
    # Need to be true for the sea points
    sea_index = (lsm_cube.data == 0)
    # The land points
    land_index = (lsm_cube.data != 0)
    return land_index, sea_index


def produce_land_sea_bool(lsm_cube):
    """
    Return the land_bool and sea_bool from a given lsm_cube.
    Note: this method is not for mask; but for true value to be reserved
    """
    # Need to be true for the sea points
    sea_bool = (lsm_cube.data == 0)
    # The land points
    land_bool = (lsm_cube.data != 0)
    return land_bool, sea_bool


def transform_cube_by_masked_index(cube, masked_indx):
    """
    Transform the input cube based the land or sea mark given
    """
    new_cube_data = np.ma.masked_array(cube.data, mask=masked_indx)
    new_cube = cube.copy()
    new_cube.data = new_cube_data
    return new_cube


def get_cube_grid_points(cube):
    """
    Return the (latitude, longitude) as (x, y) grids from an input cube.
    """
    x = cube.coord('latitude').points
    y = cube.coord('longitude').points
    return (x, y)


def interp_by_scipy(src_data, src_grids, trg_grids, method='linear'):
    """
    Use Scipy to interpolate input source data to target grid.
    ------
    Input:
        src_data: the metadata on src_grids
        src_grids: a source grid tuple (1d_x_array, 1d_y_array)
        tgt_grids: a target grid tuple (1d_x_array, 1d_y_array)
        method: the interpolate method used by scipy RegularGridInterpolator
                only available "linear" & "nearest"
    """
    from itertools import product
    from scipy.interpolate import RegularGridInterpolator
    # Define the interpolator
    scipy_interp = RegularGridInterpolator(
            src_grids, src_data, method=method)
    # Derive
    drv_scipy_data = scipy_interp(
            list(product(trg_grids[0], trg_grids[1])))
    drv_scipy_data = drv_scipy_data.reshape(
            len(trg_grids[0]), len(trg_grids[1]))
    return drv_scipy_data


def two_stage_interp(cube_src, topo_tgt, lsm_src, lsm_tgt, method='linear'):
    """
    Do two stages interpolation (land & sea) seperately and then
    integrate the results together by using Scipy Interpolate.
    ------
    Input:
        cube_src: the phenomenon cube with the source grid;
        topo_tgt: the topography cube with the target grid;
        lsm_src: land sea mask for the source grid;
        lsm_tgt: land sea mask for the target grid;
        method: the interpolate method used by scipy RegularGridInterpolator
                only available "linear" & "nearest"
    Output:
        drv_cube: the regridded cube on the target grid
                  with integrated data of both land and sea.
    """
    # Get the land and sea index for both grids
    land_indx_src, sea_indx_src = get_land_sea_index(lsm_src)
    land_indx_tgt, sea_indx_tgt = get_land_sea_index(lsm_tgt)

    # Mask the src sea points for land cube
    cube_land_src = transform_cube_by_masked_index(cube_src, sea_indx_src)
    # Mask the src land points
    cube_sea_src = transform_cube_by_masked_index(cube_src, land_indx_src)
    # Mask the tgt sea points
    topo_land_tgt = transform_cube_by_masked_index(topo_tgt, sea_indx_tgt)
    # Mask the tgt land points
    topo_sea_tgt = transform_cube_by_masked_index(topo_tgt, land_indx_tgt)

    # Get the grids for both source and target
    land_grids_src = get_cube_grid_points(cube_land_src)
    sea_grids_src = get_cube_grid_points(cube_sea_src)

    land_grids_tgt = get_cube_grid_points(topo_land_tgt)
    sea_grids_tgt = get_cube_grid_points(topo_sea_tgt)

    # Derive the cube_tgt from src
    drv_cube_land_data = interp_by_scipy(
        cube_land_src.data, land_grids_src, land_grids_tgt, method=method)

    drv_cube_sea_data = interp_by_scipy(
        cube_sea_src.data, sea_grids_src, sea_grids_tgt, method=method)

    # Combine the land & sea data together
    drv_cube_tgt = create_derive_cube(cube_src, topo_tgt)
    # Update with land data
    combined_data = np.where(land_indx_tgt,
                             drv_cube_land_data, drv_cube_tgt.data)
    # Update with sea data
    combined_data = np.where(sea_indx_tgt,
                             drv_cube_sea_data, combined_data)
    # update the cube with integrated data
    drv_cube = update_cube_with_new_data(drv_cube_tgt, combined_data)
    return drv_cube


def get_screen_param_from_cube_list(input_cubes, scn_param_name, mean=False):
    """
    Get the required screen parameter from the cube lists.
    -----
    Input:
        input_cubes: a cube list
        scn_param_name: a string of the required cube name
        mean: define if cell_methods has mean
    Output:
        param_scn: a screen parameter cube
    """
    param_cube = None
    for cube in input_cubes:
        if not mean:
            if scn_param_name in cube.name() and not cube.cell_methods:
                if (not cube.coords('model_level_number') and
                        not cube.coords('air_pressure') and
                        not cube.coords('pressure')):
                    param_cube = cube
                    break
        else:
            # TODO: If the param not only has mean but other methods
            if scn_param_name in cube.name() and cube.cell_methods:
                if (not cube.coords('model_level_number') and
                        not cube.coords('air_pressure') and
                        not cube.coords('pressure')):
                    param_cube = cube
                    break
    if param_cube is None:
        raise ValueError('No such parameter {} available!'.
                         format(scn_param_name))
    else:
        return param_cube
    
