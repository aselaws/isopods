"""
"""
# --- Imports ---
import numpy as np
import rasterio.features  # for converting array to shapes
import shapely.geometry  # make polygons from rasterio shapes
import geojson  # for preparing polygons for saving to file
from .convert import convert_grid_values
from .grid import transform_coords_to_grid


# --- Functions ---
def convert_grid_to_polys(arr: np.array) -> dict:
    """Make dict of polygons of discrete regions in grid geometry.
    
    The rasterio shape conversion needs the region labels to have
    float32 or int32 dtype. If this isn't the case in the input arr,
    then the labels are temporarily switched to arbitrary float32.
    The output dict uses the same labels as in the input arr.

    Inputs
    ------
    arr - np.array. 2D grid of region labels.

    Returns
    -------
    dict_polys - dict. One key per unique label in arr. Each value
                 is a shapely Polygon or MultiPolygon.
    """

    # rasterio shape conversion needs float32 or int32 dtype.
    try:
        arr_float32 = arr.astype(np.float32)
        converted_vals = False
    except ValueError:
        # If not numeric, convert to float with this lookup:
        converted_vals = True
        # Looking to find a shape for each val in the array:
        try:
            vals = np.unique(arr.compressed())  # masked array
        except AttributeError:
            vals = np.unique(arr)  # normal array
        dict_val_floats = dict(
            zip(vals, np.arange(len(vals), dtype=np.float32)))
        arr_float32 = convert_grid_values(arr, dict_val_floats)

    shapes_generator = _make_shapes_from_arr(arr_float32)
    dict_polys = _convert_shapes_to_poly_dict(shapes_generator)

    if converted_vals:
        # Convert floats back to original labels.
        dict_polys = dict([(val, dict_polys[f_val])
                           for val, f_val in dict_val_floats.items()])
    else:
        pass
    return dict_polys


def _make_shapes_from_arr(arr: np.array, **kwargs) -> Generator:
    """Convert arr of regions into dict of shapes and labels.

    Wrapper for rasterio.features.shapes.

    Inputs
    ------
    arr - np.array.

    Returns
    -------
    shapes - Generator.    
    """
    # Convert the array into shapes. Adjacent areas of the array with the
    # same value (e.g. belong to same region) will be gathered as one
    # shape.
    shapes = rasterio.features.shapes(np.float32(arr), **kwargs)
    return shapes


def _convert_shapes_to_poly_dict(shapes_generator: Generator) -> dict:
    """Convert shapes dictionary to polygon dict.

    Create a dict with one key per region in the grid and each
    value being a shapely geometry.

    Gather the geometry carefully in case of non-contiguous regions.
    shapes only gathers areas that are adjacent in the array.
    For example, ambulance area shapes will have one entry for Devon
    and another for the island of Lundy. Both will have the same value
    (belong to the same region). Take care to retain both entries.

    Inputs
    ------
    shapes_generator - Generator. Each item is a tuple of
                       (polygon dict, region label). The output from
                       rasterio.features.shapes.

    Returns
    -------
    dict_polys - dict. One key per region in the grid. Each value
                 is a shapely Polygon or MultiPolygon.
    """
    # Convert generator to list so we can access it more than once:
    shapes = [s for s in shapes_generator]
    # Labels of regions present here:
    region_vals, val_counts = np.unique(
        [s[1] for s in shapes], return_counts=True)
    # Set up a dictionary. Each region has an empty array the length of
    # the number of shapes that belong to it.
    dict_shapes = dict([(r, np.empty(n, dtype=object)) for r, n in
                        zip(region_vals, val_counts)])
    fill_counts = dict([(r, 0) for r in region_vals])

    # Fill the dictionary with polygons, grouping by region.
    # shapes can have multiple entries for one region
    # when the region consists of multiple non-adjacent shapes,
    # so place multiple polys for same region in the list.
    for s in shapes:
        region_val = s[1]
        coord_lists = s[0]['coordinates']
        p = _make_poly_from_coords_lists(coord_lists, offset=0.0)
        # Place this poly in the next empty slot for its region:
        ind = fill_counts[region_val]
        dict_shapes[region_val][ind] = p
        fill_counts[region_val] += 1

    # Convert dict of lists into a dict of polygons and multipolygons.
    # Lists with only one polygon are released from the list.
    # Combine lists of polygons into a single MultiPolygon.
    dict_polys = dict([(region_val, _convert_poly_list_to_poly(p_list))
                       for region_val, p_list in dict_shapes.items()])
    return dict_polys


def _make_poly_from_coords_lists(
        coord_lists: list | tuple,
        offset: float | int = 0.5
        ) -> shapely.geometry.Polygon:
    """Make Polygon from lists of exterior and interior coordinates.
    
    Inputs
    ------
    coord_lists - list | tuple. Sets of coordinates for one polygon.
                  The first list is the exterior coordinates and any
                  extra lists are for interior holes.
    offset      - float | int. Offset to convert result from the
                  centres to the edges of cells. Should be 0.5 when
                  coords are in pixel units.

    Returns
    -------
    p - shapely.geometry.Polygon. Polygon from input coords.
    """
    # Pick out coordinates for the exterior of the polygon.
    # Apply offset so lines are drawn at edge of pixel,
    # not through middle.
    ext = coord_lists[0]
    if offset == 0.0:
        pass
    else:
        ext = list(np.array(ext) - offset)
    # Pick out any holes in the polygon if applicable:
    n_poly = len(coord_lists)
    if n_poly == 1:
        holes = None
    else:
        holes = coord_lists[1:]
        # Apply same offset as for exterior coords:
        if offset == 0.0:
            pass
        else:
            holes = [list(np.array(h) - offset) for h in holes]
    # Convert list(s) of coordinates to shapely Polygon:
    p = shapely.geometry.Polygon(ext, holes)
    return p


def _convert_poly_list_to_poly(
        p_list: list | tuple
        ) -> shapely.geometry.Polygon | shapely.geometry.MultiPolygon:
    """Make a Polygon or MultiPolygon from this list of polygons.

    Helper for _convert_shapes_to_poly_dict().

    Inputs
    ------
    p_list - list | tuple. List of polygons. Will be length 1 when
             the region is a single landmass and longer
             when the region is non-contiguous.

    Returns
    -------
    p - shapely.geometry.Polygon | shapely.geometry.MultiPolygon.
        A single geometry from the input polygon(s).    
    """
    if len(p_list) > 1:
        # Combine list of polygons into MultiPolygon.
        p = shapely.geometry.MultiPolygon(list(p_list))
    elif len(p_list) == 1:
        # Use the single polygon given.
        p = p_list[0]
    else:
        # This shouldn't happen!
        p = None
    return p


def convert_dict_polys_to_featurecollection(
        dict_polys: dict
        ) -> geojson.FeatureCollection:
    """Convert dict to FeatureCollection for saving to file.

    Inputs
    ------
    dict_polys - dict. One key per region, each value is a shapely
                 geometry.

    Returns
    -------
    fc - geojson.FeatureCollection. A single collection of all the
         input geometries with the region labels carried through.
    """
    features = np.empty(len(dict_polys), dtype=object)
    ind = 0
    for region, poly in dict_polys.items():
        f = geojson.Feature(geometry=shapely.geometry.mapping(poly),
                            properties={'label': region})
        features[ind] = f
        ind += 1
    fc = geojson.FeatureCollection(list(features))
    return fc


def convert_featurecollection_to_dict_polys(
        fc: geojson.FeatureCollection
        ) -> dict:
    """Convert FeatureCollection to dict for easier use.

    Easier to load FeatureCollection from file than a dict,
    and easier to use a dict than a FeatureCollection.

    Inputs
    ------
    fc - geojson.FeatureCollection. The "label" property is made in
         the convert_dict_polys_to_featurecollection() function.

    Returns
    -------
    d - dict. One key per region, one geometry per key.
    """
    d = {}
    for f in fc['features']:
        poly = shapely.geometry.shape(f['geometry'])
        region = f['properties']['label']
        d[region] = poly
    return d


def update_poly_coords(
        p: shapely.geometry.Polygon,
        *,
        cell_width: float | int,
        cell_height: float | int,
        n_rows: float | int,
        xmin: float | int,
        ymax: float | int,
        **kwargs
        ) -> shapely.geometry.Polygon:
    """Scale a polygon's coordinates from grid to reference system.

    Wrapper for transform_coords_to_grid().

    Convert polygon coordinates in terms of the grid (x are column
    indices, y are row indices) to the coordinate reference system
    (e.g. British National Grid).

    shapely v2.1.0 (April 2025) has a new interleaved kwarg in
    shapely.transform() which would allow for coordinate
    transformation, but this function uses another method to allow
    compatibility with older versions of shapely.

    Inputs
    ------
    p           - shapely.geometry.Polygon.
    cell_width  - float | int. Width of cells for this grid.
    cell_height - float | int. Height of cells for this grid.
    n_rows      - float | int. Number of rows in the grid.
    xmin        - float | int. x bound of the grid.
    ymax        - float | int. y bound of the grid.
    **kwargs    - dict. These go unused.

    Returns
    -------
    p - shapely.geometry.Polygon. New version of the input polygon
        with transformed coordinates.
    """
    ext = np.array([list(c) for c in p.exterior.coords])
    ext = transform_coords_to_grid(
        ext, cell_width=cell_width, cell_height=cell_height,
        n_rows=n_rows, xmin=xmin, ymax=ymax
    )
    holes = [list(r.coords) for r in p.interiors]
    holes_t = np.empty(len(holes), dtype=object)
    for i, h in enumerate(holes):
        coords = np.array([list(c) for c in h])
        coords = transform_coords_to_grid(
            coords, cell_width=cell_width, cell_height=cell_height,
            n_rows=n_rows, xmin=xmin, ymax=ymax
        )
        holes_t[i] = coords
    holes = None if len(holes_t) < 1 else holes_t
    p = shapely.geometry.Polygon(ext, holes)
    return p


def transform_poly_dict_coords(
        dict_polys: dict,
        *,
        cell_width: float | int,
        cell_height: float | int,
        n_rows: float | int,
        xmin: float | int,
        ymax: float | int,
        **kwargs
        ) -> dict:
    """Transform coordinates of each geometry in this dict.

    Wrapper for update_poly_coords().

    Assumes that input dict is has geometry in grid coordinates
    (x is column index, y is row index) and that input parameters
    will transform these to the intended reference system, e.g.
    British National Grid.

    Inputs
    ------
    dict_polys  - dict. One key per region label, values are
                  shapely geometry.
    cell_width  - float | int. Width of cells for this grid.
    cell_height - float | int. Height of cells for this grid.
    n_rows      - float | int. Number of rows in the grid.
    xmin        - float | int. x bound of the grid.
    ymax        - float | int. y bound of the grid.
    **kwargs    - dict. These go unused.

    Returns
    -------
    dict_polys_new - dict. Same keys as input dict_polys, values
                     have their geometry transformed to new system.
    """
    dict_polys_new = {}
    for label, p in dict_polys.items():
        if type(p) == shapely.geometry.multipolygon.MultiPolygon:
            new_mp_list = np.empty(len(p.geoms), dtype=object)
            for i, poly in enumerate(p.geoms):
                poly = update_poly_coords(
                    poly, cell_width=cell_width, cell_height=cell_height,
                    n_rows=n_rows, xmin=xmin, ymax=ymax
                )
                new_mp_list[i] = poly
            new_p = shapely.geometry.MultiPolygon(list(new_mp_list))
        elif type(p) == shapely.geometry.polygon.Polygon:
            new_p = update_poly_coords(
                p, cell_width=cell_width, cell_height=cell_height,
                n_rows=n_rows, xmin=xmin, ymax=ymax
            )
        else:
            # This shouldn't happen!
            new_p = None
        dict_polys_new[label] = new_p
    return dict_polys_new


def convert_poly_to_coords(p: shapely.geometry.Polygon) -> np.array:
    """Flatten Polygon exterior and hole coords for plotly.
    
    Inputs
    ------
    p - shapely.geometry.Polygon.

    Returns
    -------
    coords - np.array. All coordinates of the input polygon, first
             the exterior then any holes, where the sections are
             split using None. First column is x, second is y.
    """
    ext = np.array([list(c) for c in p.exterior.coords])
    holes = [np.array([list(c) for c in list(r.coords)])
             for r in p.interiors]
    if len(holes) == 0:
        coords = ext
    else:
        # Separate coord lists with this gap:
        gap = np.full(ext.shape[-1], None)
        # Set up array of coords to concat.
        # First item ext coords, then alternate gaps and holes.
        list_concat = np.empty(1 + len(holes)*2, dtype=object)
        g_inds = np.arange(1, len(list_concat), 2)
        h_inds = np.arange(2, len(list_concat), 2)
        list_concat[0] = ext
        list_concat[g_inds] = [gap]
        list_concat[h_inds] = holes
        coords = np.vstack(list_concat)
    return coords


def convert_multipoly_to_coords(
        mp: shapely.geometry.MultiPolygon
        ) -> np.array:
    """Flatten MultiPolygon exterior and hole coords for plotly.
    
    Wrapper for convert_poly_to_coords().
    
    Inputs
    ------
    mp - shapely.geometry.MultiPolygon.

    Returns
    -------
    coords - np.array. All coordinates of all input polygons, split
             using None. For each polygon's coordinates order, see
             convert_poly_to_coords(). First column is x, second is y.
    """
    coords_lists = [convert_poly_to_coords(p) for p in mp.geoms]
    # Separate coord lists with this gap:
    gap = np.full(coords_lists[0].shape[-1], None)
    # Set up array of coords to concat. Alternate polys and gaps.
    list_concat = np.empty((len(mp.geoms) * 2) - 1, dtype=object)
    p_inds = np.arange(0, len(list_concat), 2)
    g_inds = np.arange(1, len(list_concat), 2)
    list_concat[p_inds] = coords_lists
    list_concat[g_inds] = [gap]
    coords = np.vstack(list_concat)
    return coords


def convert_dict_polys_to_coords(
        dict_polys: dict[any, shapely.Geometry]
        ) -> dict[any, np.array]:
    """Convert each Multi/Polygon in dict to coordinates for plotly.

    Wrapper for convert_poly_to_coords().

    Inputs
    ------
    dict_polys - dict[any, shapely.Geometry].

    Returns
    -------
    dict_coords - dict[any, np.array].
    """
    dict_coords = {}
    for label, shape in dict_polys.items():
        if shape.geom_type == 'Polygon':
            coords = convert_poly_to_coords(shape)
        elif shape.geom_type == 'MultiPolygon':
            coords = convert_multipoly_to_coords(shape)
        elif shape.geom_type == 'GeometryCollection':
            print('Do something')
            coords = None
        else:
            # What are we doing here, a warning or an error? TO DO.
            coords = None
        dict_coords[label] = coords
    return dict_coords
