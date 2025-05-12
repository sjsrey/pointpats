import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point
from pointpats import ellipse

@pytest.fixture
def sample_coords():
    return np.array([
        [66.22, 32.54],
        [22.52, 22.39],
        [31.01, 81.21],
        [9.47, 31.02],
        [30.78, 60.10],
        [75.21, 58.93],
        [79.26, 7.68],
        [8.23, 39.93],
        [98.73, 77.17],
        [89.78, 42.53],
        [65.19, 92.08],
    ])

@pytest.fixture
def sample_weights():
    return np.arange(1, 12)

@pytest.fixture
def sample_geoseries(sample_coords):
    return gpd.GeoSeries([Point(x, y) for x, y in sample_coords])

def test_ellipse_numpy_no_weights(sample_coords):
    major, minor, rotation = ellipse(sample_coords)
    assert isinstance(major, float)
    assert isinstance(minor, float)
    assert isinstance(rotation, float)
    assert major > 0
    assert minor > 0
    assert -np.pi <= rotation <= np.pi

def test_ellipse_numpy_with_weights(sample_coords, sample_weights):
    major, minor, rotation = ellipse(sample_coords, weights=sample_weights)
    assert isinstance(major, float)
    assert isinstance(minor, float)
    assert isinstance(rotation, float)

def test_ellipse_list_input(sample_coords):
    coords_list = sample_coords.tolist()
    major, minor, rotation = ellipse(coords_list)
    assert isinstance(major, float)
    assert isinstance(minor, float)
    assert isinstance(rotation, float)

def test_ellipse_list_with_weights(sample_coords, sample_weights):
    coords_list = sample_coords.tolist()
    weights_list = sample_weights.tolist()
    major, minor, rotation = ellipse(coords_list, weights=weights_list)
    assert isinstance(major, float)
    assert isinstance(minor, float)
    assert isinstance(rotation, float)

def test_ellipse_geoseries(sample_geoseries):
    ellipse_ = ellipse(sample_geoseries)
    assert ellipse_.geom_type == "Polygon"
    assert ellipse_.is_valid

def test_ellipse_geoseries_with_weights(sample_geoseries, sample_weights):
    ellipse_ = ellipse(sample_geoseries, weights=sample_weights)
    assert ellipse_.geom_type == "Polygon"
    assert ellipse_.is_valid

def test_invalid_method(sample_coords):
    with pytest.raises(ValueError, match="`method` must be either 'crimestat' or 'yuill'"):
        ellipse(sample_coords, method="invalidmethod")

def test_weights_length_mismatch(sample_coords):
    wrong_weights = np.arange(5)
    with pytest.raises(ValueError):
        ellipse(sample_coords, weights=wrong_weights)
