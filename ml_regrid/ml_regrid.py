from sklearn.preprocessing import FunctionTransformer

from test_grid import *

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
 
topo_aps2 = empty_3d_cube_aps2(
    surface_alt_aps2_data, 'surface_altitude', 'm', 'm01s00i033')
topo_aps3 = empty_3d_cube_aps3(
    surface_alt_aps3_data, 'surface_altitude', 'm', 'm01s00i033')
lsm_aps2 = empty_3d_cube_aps2(
    lsm_aps2_data, 'land_area_fraction', '1', )
lsm_aps3 = empty_3d_cube_aps3(
    lsm_aps3_data, 'land_area_fraction', '1')
t_scn_aps3 = empty_3d_cube_aps3(
    t_scn_aps3_data, 'air_temperature', 'K')
dpt_scn_aps3 = empty_3d_cube_aps3(
    dpt_scn_aps3_data, 'dew_point_temperature', 'K', 'm01s03i250')
sfc_prs_aps3 = empty_3d_cube_aps3(
    sfc_prs_aps3_data, 'air_pressure_at_sea_level', 'Pa')
    
 X = t_scn.data
 transformer = FunctionTransformer(interpolate_by_scipy_linear)
 y = transformer.transform(X)
 
