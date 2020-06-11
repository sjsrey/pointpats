import numpy
from scipy import spatial
from functools import singledispatch
from libpysal.cg import alpha_shape_auto
from libpysal.cg.kdtree import Arc_KDTree


# ------------------------------------------------------------#
# Utilities and dispatching                                   #
# ------------------------------------------------------------#

TREE_TYPES = (spatial.KDTree, spatial.cKDTree, Arc_KDTree)
try:
    from sklearn.neighbors import KDTree, BallTree

    TREE_TYPES = (*TREE_TYPES, KDTree, BallTree)
except ModuleNotFoundError:
    pass

HULL_TYPES = (
    numpy.ndarray,
    spatial.qhull.ConvexHull,
)

## Define default dispatches and special dispatches without GEOS

### AREA
@singledispatch
def area(shape):
    """
    If a shape has an area attribute, return it. 
    Works for: 
        shapely.geometry.Polygon
    """
    return shape.area


@area.register
def _(shape: spatial.qhull.ConvexHull):
    """
    If a shape is a convex hull from scipy, 
    assure it's 2-dimensional and then use its volume. 
    """
    assert shape.points.shape[1] == 2
    return shape.volume


@area.register
def _(shape: numpy.ndarray):
    """
    If a shape describes a bounding box, compute length times width
    """
    assert len(shape) == 4, "shape is not a bounding box!"
    width, height = shape[2] - shape[0], shape[3] - shape[1]
    return numpy.abs(width * height)


### bounding box
@singledispatch
def bbox(shape):
    """
    If a shape has bounds, use those.
    Works for:
        shapely.geometry.Polygon
    """
    return bbox(numpy.asarray(shape))


@bbox.register
def _(shape: numpy.ndarray):
    """
    If a shape is an array of points, compute the minima/maxima 
    or let it pass through if it's 1 dimensional & length 4
    """
    if (shape.ndim == 1) & (len(shape) == 4):
        return shape
    return numpy.array([*shape.min(axis=0), *shape.max(axis=0)])


@bbox.register
def _(shape: spatial.qhull.ConvexHull):
    """
    For scipy.spatial.ConvexHulls, compute the bounding box from
    their boundary points.
    """
    return bbox(shape.points[shape.vertices])


### contains


@singledispatch
def contains(shape, x, y):
    """
    Try to use the shape's contains method directly on XY.
    Does not currently work on anything. 
    """
    raise NotImplementedError()
    return shape.contains((x, y))


@contains.register
def _(shape: numpy.ndarray, x: float, y: float):
    """
    If provided an ndarray, assume it's a bbox
    and return whether the point falls inside
    """
    xmin, xmax = shape[0], shape[2]
    ymin, ymax = shape[1], shape[3]
    in_x = (xmin <= x) and (x <= xmax)
    in_y = (ymin <= y) and (y <= ymax)
    return in_x & in_y


@contains.register
def _(shape: spatial.Delaunay, x: float, y: float):
    """
    For points and a delaunay triangulation, use the find_simplex
    method to identify whether a point is inside the triangulation.

    If the returned simplex index is -1, then the point is not
    within a simplex of the triangulation. 
    """
    return shape.find_simplex((x, y)) > 0


@contains.register
def _(shape: spatial.qhull.ConvexHull, x: float, y: float):
    """
    For convex hulls, convert their exterior first into a Delaunay triangulation
    and then use the delaunay dispatcher.
    """
    exterior = shape.points[shape.vertices]
    delaunay = spatial.Delaunay(exterior)
    return contains(delaunay, x, y)


### centroid
@singledispatch
def centroid(shape):
    """
    Assume the input is a shape with a centroid method:
    """
    return shape.centroid


@centroid.register
def _(shape: numpy.ndarray):
    """
    Handle point arrays or bounding boxes
    """
    from .centrography import mean_center

    if shape.ndim == 2:
        return mean_center(shape).squeeze()
    elif shape.ndim == 1:
        assert shape.shape == (4,)
        xmin, ymin, xmax, ymax = shape
        return numpy.column_stack(
            (numpy.mean((xmin, xmax)), numpy.mean((ymin, ymax)))
        ).squeeze()
    else:
        raise TypeError(
            f"Centroids are only implemented in 2 dimensions,"
            f" but input has {shape.ndim} dimensinos"
        )


@centroid.register
def _(shape: spatial.qhull.ConvexHull):
    """
    Treat convex hulls as arrays of points
    """
    return centroid(shape.points[shape.vertices])


try:
    from shapely.geometry.base import BaseGeometry as _BaseGeometry
    from shapely.geometry import (
        Polygon as _ShapelyPolygon,
        MultiPolygon as _ShapelyMultiPolygon,
    )
    from shapely.geometry import Point as _ShapelyPoint

    HULL_TYPES = (*HULL_TYPES, _ShapelyPolygon, _ShapelyMultiPolygon)
    HAS_SHAPELY = True

    @contains.register
    def _(shape: _BaseGeometry, x: float, y: float):
        """
        If we know we're working with a shapely polygon, 
        then use the contains method & cast input coords to a shapely point
        """
        return shape.contains(_ShapelyPoint((x, y)))

    @bbox.register
    def _(shape: _BaseGeometry):
        """
        If a shape is an array of points, compute the minima/maxima 
        or let it pass through if it's 1 dimensional & length 4
        """
        return numpy.asarray(list(shape.bounds))

    @centroid.register
    def _(shape: _BaseGeometry):
        """
        Handle shapely, which requires explicit centroid extraction
        """
        return numpy.asarray(list(shape.centroid.coords)).squeeze()


except ModuleNotFoundError:
    HAS_SHAPELY = False


try:
    import pygeos

    HAS_PYGEOS = True
    HULL_TYPES = (*HULL_TYPES, pygeos.Geometry)

    @area.register
    def _(shape: pygeos.Geometry):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.area
        """
        return pygeos.area(shape)

    @contains.register
    def _(shape: pygeos.Geometry, x: float, y: float):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.within casting the points to a pygeos object too
        """
        return pygeos.within(pygeos.points((x, y)), shape)

    @bbox.register
    def _(shape: pygeos.Geometry):
        """
        If we know we're working with a pygeos polygon, 
        then use pygeos.bounds
        """
        return pygeos.bounds(shape)

    @centroid.register
    def _(shape: pygeos.Geometry):
        """
        if we know we're working with a pygeos polygon,
        then use pygeos.centroid
        """
        return pygeos.coordinates.get_coordinates(pygeos.centroid(shape)).squeeze()


except ModuleNotFoundError:
    HAS_PYGEOS = False

# ------------------------------------------------------------#
# Constructors for trees, prepared inputs, & neighbors        #
# ------------------------------------------------------------#


def build_best_tree(coordinates, metric):
    """
    Build the best query tree that can support the application.
    Chooses from:
    1. sklearn.KDTree if available and metric is simple
    2. sklearn.BallTree if available and metric is complicated
    3. scipy.spatial.cKDTree if nothing else
    """
    coordinates = numpy.asarray(coordinates)
    tree = spatial.cKDTree
    try:
        from sklearn.neighbors import KDTree, BallTree

        if metric in KDTree.valid_metrics:
            tree = lambda coordinates: KDTree(coordinates, metric=metric)
        elif metric in BallTree.valid_metrics:
            tree = lambda coordinates: BallTree(coordinates, metric=metric)
        elif callable(metric):
            warnings.warn(
                "Distance metrics defined in pure Python may "
                " have unacceptable performance!",
                stacklevel=2,
            )
            tree = lambda coordinates: BallTree(coordinates, metric=metric)
        else:
            raise KeyError(
                f"Metric {metric} not found in set of available types."
                f"BallTree metrics: {BallTree.valid_metrics}, and"
                f"scikit KDTree metrics: {KDTree.valid_metrics}."
            )
    except ModuleNotFoundError as e:
        if metric not in ("l2", "euclidean"):
            raise KeyError(
                f"Metric {metric} requested, but this requires"
                f" scikit-learn to use. Without scikit-learn, only"
                f" euclidean distance metric is supported."
            )
    return tree(coordinates)


def k_neighbors(tree, coordinates, k, **kwargs):
    """
    Query a kdtree for k neighbors, handling the self-neighbor case
    in the case of coincident points. 
    """
    distances, indices = tree.query(coordinates, k=k + 1, **kwargs)
    n, ks = distances.shape
    assert ks == k + 1
    full_indices = numpy.arange(n)
    other_index_mask = indices != full_indices.reshape(n, 1)
    has_k_indices = other_index_mask.sum(axis=1) == (k + 1)
    other_index_mask[has_k_indices, -1] = False
    distances = distances[other_index_mask].reshape(n, k)
    indices = indices[other_index_mask].reshape(n, k)
    return distances, indices


def prepare_hull(coordinates, hull=None):
    """
    Construct a hull from the coordinates given a hull type
    Will either return:
        - a bounding box array of [xmin, ymin, xmax, ymax]
        - a scipy.spatial.ConvexHull object from the Qhull library
        - a shapely shape using alpha_shape_auto
    """
    if isinstance(hull, numpy.ndarray):
        assert len(hull) == 4, f"bounding box provided is not shaped correctly! {hull}"
        assert hull.ndim == 1, f"bounding box provided is not shaped correctly! {hull}"
        return hull
    if (hull is None) or (hull == "bbox"):
        return bbox(coordinates)
    if HAS_SHAPELY:  # protect the isinstance check if import has failed
        if isinstance(hull, (_ShapelyPolygon, _ShapelyMultiPolygon)):
            return hull
    if HAS_PYGEOS:
        if isinstance(hull, pygeos.Geometry):
            return hull
    if isinstance(hull, str):
        if hull.startswith("convex"):
            return spatial.ConvexHull(coordinates)
        elif hull.startswith("alpha") or hull.startswith("α"):
            return alpha_shape_auto(coordinates)
    elif isinstance(hull, spatial.qhull.ConvexHull):
        return hull
    raise ValueError(
        f"Hull type {hull} not in the set of valid options:"
        f" (None, 'bbox', 'convex', 'alpha', 'α', "
        f" shapely.geometry.Polygon, pygeos.Geometry)"
    )