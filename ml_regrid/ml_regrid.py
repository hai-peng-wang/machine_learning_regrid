from sklearn.preprocessing import FunctionTransformer

from test_grid import *


def get_grid_vector(lat, lon):
    """
    Convert input lat, lon (nparray) into grid_mesh for scipy interpolate;
    First step to prepare for scipy interpolation.
    """
    import numpy as np
    lat_mesh, lon_mesh = np.meshgrid(lat, lon)
    lat_vec = np.ravel(lat_mesh)
    lon_vec = np.ravel(lon_mesh)
    return lat_vec, lon_vec

def get_grid_points(lat, lon):
    """
    Convert input lat, lon (nparray) into grid_points for scipy interpolate.
    """
    lat_vec, lon_vec = get_grid_vector(lat, lon)
    grid_points = np.vstack((lat_vec, lon_vec)).T
    return grid_points

def interpolate_by_scipy_linear(src_data):
    """
    Use Scipy to interpolate input source data to target grid.
    ------
    Input:
        src_data: the metadata on src_grids
    Output:
        tgt_data: the interpolated data on target grids.
    Note: the following parameters are required
        lat_src or lat_tgt: the latitude 1d np.array from source or target grids
        lon_src or lon_tgt: the longitude 1d np.array from source or target grids
        method: the interpolate method used by scipy RegularGridInterpolator
                only available "linear" & "nearest"
    """
    import scipy.interpolate as interpolator

    # Need prepare grid_points and vestor for scipy interpolator
    src_data_vec = np.ravel(src_data)
    grid_points_src = get_grid_points(lat_src, lon_src)
    lat_tgt_vec, lon_tgt_vec = get_grid_vector(lat_tgt, lon_tgt)
    
    # Define the interpolator
    scipy_interp = interpolator.LinearNDInterpolator(grid_points_src, src_data_vec)
    # Derive
    tgt_data = scipy_interp(lat_tgt_vec, lon_tgt_vec)
    # Need reshape back to 2d
    tgt_data = tgt_data.reshape(len(lat_tgt), len(lon_tgt))

    return tgt_data
    
def interpolate_by_scipy_nearest(src_data):
    """
    Use Scipy to interpolate input source data to target grid.
    ------
    Input:
        src_data: the metadata on src_grids
    Output:
        tgt_data: the interpolated data on target grids.
    Note: the following parameters are required
        lat_src or lat_tgt: the latitude 1d np.array from source or target grids
        lon_src or lon_tgt: the longitude 1d np.array from source or target grids
        method: the interpolate method used by scipy RegularGridInterpolator
                only available "linear" & "nearest"
    """
    import scipy.interpolate as interpolator
    global lat_src, lon_src, lat_tgt, lon_tgt
    # Need prepare grid_points and vestor for scipy interpolator
    src_data_vec = np.ravel(src_data)
    grid_points_src = get_grid_points(lat_src, lon_src)
    lat_tgt_vec, lon_tgt_vec = get_grid_vector(lat_tgt, lon_tgt)
    
    # Define the interpolator
    scipy_interp = interpolator.NearestNDInterpolator(grid_points_src, src_data_vec)
    # Derive
    tgt_data = scipy_interp(lat_tgt_vec, lon_tgt_vec)
    # Need reshape back to 2d
    tgt_data = tgt_data.reshape(len(lat_tgt), len(lon_tgt))
    return tgt_data


def regrid_cube_by_iris_scheme(param_cube, target_cube, scheme=None):
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


topo_tgt = empty_3d_cube_tgt(
    surface_alt_tgt_data, 'surface_altitude', 'm')
topo_src = empty_3d_cube_src(
    surface_alt_src_data, 'surface_altitude', 'm')
lsm_tgt = empty_3d_cube_tgt(
    lsm_tgt_data, 'land_area_fraction', '1', )
lsm_src = empty_3d_cube_src(
    lsm_src_data, 'land_area_fraction', '1')
t_scn_src = empty_3d_cube_src(
    t_scn_src_data, 'air_temperature', 'K')
dpt_scn_src = empty_3d_cube_src(
    dpt_scn_src_data, 'dew_point_temperature', 'K')
sfc_prs_src = empty_3d_cube_src(
    sfc_prs_src_data, 'air_pressure_at_sea_level', 'Pa')
    
 X = t_scn.data
 transformer = FunctionTransformer(interpolate_by_scipy_linear)
 y = transformer.transform(X)
 
