from __future__ import division, print_function


from iris.aux_factory import HybridHeightFactory
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

import numpy as np

# The sample data is taken from a place includes both water and sea
# (coastline at somewhere in Sri Lanka)
# For APS2, the index points lsm_aps2.data[411:415, 231:234], shape=(4, 3)
lsm_aps2_data = np.array(
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 0],
     [1, 1, 0]])

surface_alt_aps2_data = np.array(
        [[40., 0., 0.],
         [272.5, 49.5, 0.],
         [569., 139.375, 0.],
         [649.375, 185.875, 0.]])

# For APS3, the index points lsm_aps3.data[821:829, 461:467], shape=(8, 6)
lsm_aps3_data = np.array(
      [[1, 1, 0, 0, 0, 0],
       [1, 1, 1, 0, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0, 0],
       [1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 0]])

surface_alt_aps3_data = np.array(
        [[30.79901695, 24.64077759, 0., 0., 0., 0.],
         [40.51713181, 57.62937927, 44.1620636, 0., 0., 0.],
         [61.04179001, 77.5438385, 68.0868454, 27.50967216, 0., 0.],
         [197.31484985, 109.88461304, 81.08721161, 47.29803085, 0., 0.],
         [500.52633667, 204.61886597, 96.57264709, 57.00541687, 0., 0.],
         [801.8571167, 337.81097412, 131.99372864, 66.28527832, 0., 0.],
         [847.43878174, 408.94787598, 180.79109192, 91.68476105,
          42.89185333, 0.],
         [622.86254883, 365.65594482, 213.10107422, 122.55203247,
          55.53632736, 0.]])

t_scn_aps3_data = np.array(
     [[300.0625, 299.1875, 300.859375, 300.4375, 300.640625, 300.8125],
      [300.734375, 303.546875, 300.828125, 300.53125, 300.578125, 300.796875],
      [302.0625, 301.078125, 300.703125, 299.90625, 300.640625, 300.71875],
      [300.796875, 299.890625, 299.703125, 300.046875, 300.71875, 300.65625],
      [297.078125, 298.875, 299.09375, 300.078125, 300.71875, 300.65625],
      [296.125, 297.875, 298.5625, 299.328125, 300.59375, 300.734375],
      [294.953125, 296.90625, 298.125, 298.828125, 298.671875, 300.765625],
      [296.578125, 297.375, 297.171875, 298.34375, 297.65625, 300.75]])

sfc_prs_aps3_data = np.array(
      [[101078., 101066., 101068., 101078., 101095., 101110.],
       [101061., 101047., 101060., 101074., 101098., 101116.],
       [101065., 101063., 101057., 101075., 101100., 101121.],
       [101088., 101060., 101067., 101081., 101103., 101127.],
       [101080., 101069., 101083., 101091., 101111., 101135.],
       [101087., 101073., 101095., 101104., 101120., 101142.],
       [101100., 101072., 101105., 101119., 101133., 101153.],
       [101108., 101088., 101114., 101128., 101146., 101168.]])

precip_aps3_data = np.array(
      [[0.00000000e+00, 6.07900797e-04, 9.05660615e-04,
        7.16217433e-04, 6.88065994e-04, 4.22674314e-04],
       [0.00000000e+00, 2.47466551e-04, 1.71492408e-03,
        1.02195099e-03, 9.36624213e-04, 5.31554862e-04],
       [0.00000000e+00, 9.13838708e-04, 2.80686256e-03,
        1.44079693e-03, 1.08191311e-03, 7.92450033e-04],
       [0.00000000e+00, 5.87481663e-04, 3.16777859e-03,
        1.97154691e-03, 1.01608064e-03, 7.13987132e-04],
       [0.00000000e+00, 9.86691211e-06, 1.04245740e-03,
        2.38504367e-03, 9.71641691e-04, 4.36859315e-04],
       [0.00000000e+00, 2.04424682e-05, 1.05835806e-03,
        2.17763191e-03, 1.08107582e-03, 2.96673398e-04],
       [1.48722883e-08, 1.04974047e-04, 1.34315611e-03,
        3.25316976e-03, 1.05785504e-03, 2.65039941e-04],
       [0.00000000e+00, 3.39914513e-04, 1.40707767e-03,
        2.86105606e-03, 3.17726928e-04, 1.35791352e-04]])

dpt_scn_aps3_data = np.array(
    [[293.515625, 294.03125, 294.5625, 294.53125, 294.171875, 294.],
     [293.1875, 290.59375, 292.15625, 294.078125, 294.09375, 293.6875],
     [292.59375, 291.71875, 291.390625, 291.890625, 293.828125, 293.640625],
     [291.25, 291.8125, 291.296875, 291.328125, 293.65625, 293.515625],
     [292.03125, 291.734375, 291.515625, 291.015625, 293.296875, 293.359375],
     [292.328125, 291.4375, 291.875, 291.3125, 292.90625, 293.171875],
     [289.390625, 291.3125, 291.71875, 291.234375, 291.6875, 293.0625],
     [290.03125, 291.171875, 292.140625, 291.34375, 291.953125, 292.953125]])

q_scn_aps3_data = np.array(
    [[0.01489258, 0.01538086, 0.01586914, 0.01586914, 0.01538086, 0.01538086],
     [0.01464844, 0.01245117, 0.01367188, 0.01538086, 0.01538086, 0.01489258],
     [0.01416016, 0.01342773, 0.01318359, 0.01342773, 0.01513672, 0.01489258],
     [0.01318359, 0.01342773, 0.01293945, 0.01293945, 0.01489258, 0.01489258],
     [0.0144043, 0.01367188, 0.01318359, 0.01269531, 0.01464844, 0.01464844],
     [0.01513672, 0.01367188, 0.01367188, 0.01293945, 0.0144043, 0.01464844],
     [0.01269531, 0.01367188, 0.01367188, 0.01293945, 0.01342773, 0.0144043],
     [0.01269531, 0.01342773, 0.01391602, 0.01318359, 0.01367188, 0.0144043]])

# u10_um_aps3.data[821:829, 462:468] at different longitudes (and index)
u10_aps3_data = np.array(
      [[-3.75, -6.125, -7.5, -6.375, -4.625, -3.375],
       [-0.625, -4., -4.625, -6.375, -4.75, -3.25],
       [-1.625, -3.75, -3.625, -4.125, -4.625, -2.875],
       [-4., -3.875, -3.375, -4., -4.5, -2.375],
       [-2.375, -3., -3., -3.875, -4., -2.],
       [-0.875, -2.125, -2.5, -3.375, -3.5, -1.5],
       [-0.5, -1.75, -2., -1.875, -2., -1.],
       [-1.75, -2.25, -1.75, -2.125, -2., -0.625]])

# v10_um_aps3.data[821:829, 461:467] at different latitude (same index)
v10_aps3_data = np.array(
      [[2., -0.25, -3.875, -5., -5.625, -5.5],
       [3.625, 2., -2.375, -4.875, -5.75, -5.625],
       [3.875, -4.375, -1.75, -3.25, -6.125, -5.75],
       [3.125, -2.5, -2.625, -2.5, -6.5, -6.125],
       [2.125, -2.75, -3.125, -2.75, -6.75, -6.25],
       [0.625, -2.625, -3.375, -2.875, -6.125, -6.25],
       [-1.625, -2.25, -3.5, -2.875, -3.75, -6.375],
       [-2., -2.625, -3.125, -2.875, -2.75, -6.125]])

# From air temperature model data[:3, 821:827, 461:467]
temp_model_data = np.array(
      [[[300.25,  299.625,  300.625,  300.125,  300.375,  300.625],
        [300.75,  303.75,  300.625,  300.25,  300.375,  300.625],
        [302.,  301.25,  300.625,  300.125,  300.375,  300.5],
        [302.125,  300.25,  300.125,  300.125,  300.5,  300.375],
        [299.75,  299.5,  299.625,  300.125,  300.5,  300.5],
        [297.,  298.625,  299.,  299.625,  300.375,  300.5],
        [295.5,  297.5,  298.5,  299.25,  299.,  300.625],
        [296.875,  297.625,  297.625,  298.625,  298.5,  300.625]],

       [[300.,  299.5,  300.375,  299.875,  300.125,  300.25],
        [300.5,  303.375,  300.25,  300.,  300.,  300.25],
        [301.625,  300.875,  300.375,  299.75,  300.125,  300.125],
        [302.,  300.,  299.875,  299.875,  300.125,  300.125],
        [299.625,  299.375,  299.5,  299.875,  300.25,  300.125],
        [296.75,  298.375,  298.875,  299.375,  300.125,  300.25],
        [295.375,  297.25,  298.25,  299.,  298.875,  300.25],
        [296.5,  297.375,  297.5,  298.5,  298.625,  300.25]],

       [[299.5,  299.375,  299.875,  299.375,  299.625,  299.75],
        [300.125,  302.875,  299.75,  299.5,  299.5,  299.75],
        [301.125,  300.375,  299.875,  299.375,  299.625,  299.75],
        [301.75,  299.625,  299.5,  299.5,  299.75,  299.625],
        [299.375,  299.,  299.125,  299.375,  299.75,  299.625],
        [296.375,  298.,  298.5,  299.,  299.625,  299.75],
        [295.,  296.875,  297.875,  298.625,  298.75,  299.75],
        [296.125,  297.,  297.25,  298.125,  298.625,  299.75]]])


def empty_3d_cube_aps2(data, name=None, unit=None, stash=None, **kwargs):
    """
    Prepare some iris cubes at APS2 grids for testing
    """
    if data is None:
        data = np.empty([2, 2])

    cube = Cube(data)

    time = AuxCoord([0], 'time', units='hours since epoch')

    latitude = DimCoord([6.328125, 6.5625, 6.796875, 7.03125],
                        standard_name='latitude', units='degrees')

    longitude = DimCoord([81.211053, 81.562616, 81.914179],
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

    latitude = DimCoord([6.26953125, 6.38671875, 6.50390625, 6.62109375,
                         6.73828125, 6.85546875, 6.97265625, 7.08984375],
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


def empty_model_level_cube(data, name=None, unit=None, stash=None, **kwargs):
    """
    Create a model_level cube from input data.
    """
    if data is None:
        data = np.empty([1, 3, 8, 6])
    assert data.shape == (3, 8, 6)
    # Make axis=0 for time dim_coord
    new_data = data[np.newaxis, :]
    cube = Cube(new_data)

    # time = AuxCoord([0], 'time', units='hours since epoch')
    time = DimCoord([0], 'time', units='hours since epoch')

    # model = DimCoord([1, 2, 3], 'model_level_number',
    #                  attributes={'positive': 'up'})
    model = DimCoord([1, 2, 3], 'air_pressure',
                     attributes={'positive': 'up'})
    latitude = DimCoord([6.26953125, 6.38671875, 6.50390625, 6.62109375,
                         6.73828125, 6.85546875, 6.97265625, 7.08984375],
                        standard_name='latitude', units='degrees')

    longitude = DimCoord([81.12304688, 81.29882812, 81.47460938,
                          81.65039062, 81.82617188, 82.00195312],
                         standard_name='longitude', units='degrees')

    level_heights = np.array([20., 53.336, 100.])
    level_height = DimCoord(level_heights, long_name='level_height', units='m')
    surface = AuxCoord(topo_aps3.data, 'surface_altitude', units='m')

    sigma = AuxCoord([0.99772321, 0.99393402, 0.98864199], long_name='sigma')

    cube.add_dim_coord(time, 0)
    cube.add_dim_coord(model, 1)
    cube.add_dim_coord(latitude, 2)
    cube.add_dim_coord(longitude, 3)

    cube.add_aux_coord(level_height, 1)
    cube.add_aux_coord(sigma, 1)
    cube.add_aux_coord(surface, (2, 3))

    # Now that we have all of the necessary information, construct a
    # HybridHeight derived "altitude" coordinate.
    cube.add_aux_factory(HybridHeightFactory(level_height, sigma, surface))

    if name:
        cube.long_name = name
    if unit:
        cube.units = unit
    if stash:
        cube.attributes['STASH'] = stash

    return cube


# Prepare some sample cubes
topo_aps2 = empty_3d_cube_aps2(
    surface_alt_aps2_data, 'surface_altitude', 'm', 'm01s00i033')
topo_aps3 = empty_3d_cube_aps3(
    surface_alt_aps3_data, 'surface_altitude', 'm', 'm01s00i033')
lsm_aps2 = empty_3d_cube_aps2(
    lsm_aps2_data, 'land_area_fraction', '1', )
lsm_aps3 = empty_3d_cube_aps3(
    lsm_aps3_data, 'land_binary_mask', '1')
t_scn_aps3 = empty_3d_cube_aps3(
    t_scn_aps3_data, 'air_temperature', 'K')
sfc_prs_aps3 = empty_3d_cube_aps3(
    sfc_prs_aps3_data, 'air_pressure_at_sea_level', 'Pa')
precip_aps3 = empty_3d_cube_aps3(
    precip_aps3_data, 'precipitation_amount', 'kg m-2', 'm01s05i226')
dpt_scn_aps3 = empty_3d_cube_aps3(
    dpt_scn_aps3_data, 'dew_point_temperature', 'K', 'm01s03i250')
q_scn_aps3 = empty_3d_cube_aps3(
    q_scn_aps3_data, 'specific_humidity', '1', 'm01s03i237')

# Note that u10 is sitting on different longitude (x)
u10_aps3 = empty_3d_cube_aps3(
    u10_aps3_data, 'x_wind', 'm s-1', 'm01s03i209')
u10_aps3.coord('longitude').points = np.array(
    [81.2109375, 81.38671875, 81.5625,
     81.73828125, 81.9140625, 82.08984375])

# Note that v10 is sitting on different latitude (y)
v10_aps3 = empty_3d_cube_aps3(
    v10_aps3_data, 'y_wind', 'm s-1', 'm01s03i210')
v10_aps3.coord('latitude').points = np.array(
    [6.2109375, 6.328125, 6.4453125, 6.5625,
     6.6796875, 6.796875, 6.9140625, 7.03125])

# Prepare the temperature model cube
temp_model = empty_model_level_cube(
        temp_model_data, name='air_temperature', unit='K',
        stash='m01s16i004')
