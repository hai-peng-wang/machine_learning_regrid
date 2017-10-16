import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np

def interp_weights(xy, uv, d=2):
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    
def produce_land_sea_bool(lsm_cube):
    """    Return the land_bool and sea_bool from a given lsm_cube.
    Note: this method is not for mask; but for true value to be reserved
    """
    # Need to be true for the sea points
    sea_bool = (lsm_cube.data == 0)
    # The land points
    land_bool = (lsm_cube.data != 0)
    return land_bool, sea_bool
    
cube = t_scn_aps3
lsm_src = lsm_aps3
lsm_tgt = lsm_aps2

cube_src_sea_data = np.where((lsm_src.data==0), cube.data, np.nan)
cube_src_land_data = np.where((lsm_src.data!=0), cube.data, np.nan)

topo_src = topo_aps3
topo_tgt = topo_aps2

# Define the target grid
xv, yv = np.meshgrid(lon_tgt, lat_tgt)
tgrid_tgt = np.dstack((yv, xv))

def _interp_masked_grid(interpolator, tgrid):
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
    
output_data_land = _interp_masked_grid(interpolator_land, tgrid_tgt)
output_data_sea = _interp_masked_grid(interpolator_sea, tgrid_tgt)

combined_data = np.where((lsm_tgt.data==0), output_data_sea, output_data_land)

lsm_land_bool_src, lsm_sea_bool_src = produce_land_sea_bool(lsm_src)
lsm_land_bool_tgt, lsm_sea_bool_tgt = produce_land_sea_bool(lsm_tgt)

coast_pnt_bool = np.isnan(combined_data)

coast_pnt_bool_land = np.logical_and(coast_pnt_bool, lsm_land_bool_tgt)
coast_points_land = grid_tgt[coast_pnt_bool_land]

# Do the same with sea
coast_pnt_bool_sea = np.logical_and(coast_pnt_bool, lsm_sea_bool_tgt)
coast_points_sea = grid_tgt[coast_pnt_bool_sea]

# xy is the source grid
xy = grid_src[lsm_sea_bool_src]
# uv is the target grid
uv = grid_tgt[coast_pnt_bool_land]

vtx, wts = interp_weights(xy, uv)

# 'values' is the coastline points to be interpolate
values = cube_src_sea_data

%%time
out_value_sea = interpolate(values, vtx, wts)

combined_data[coast_pnt_bool_sea] = out_value_sea
