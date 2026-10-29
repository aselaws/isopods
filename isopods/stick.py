"""Assign shapes to grid cells and calculate the cell area covered.

TO DO - consider changing main stick function so that cell bounds
are calculated in advance rather than during the loop.
More potential in the future for non-rectangular cells or not
calculating all cells in the grid if some are known to be empty.
Would also need to store cell area rather than proportion as
cell area could vary.
"""
# --- Imports ---
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from .grid import set_up_cell_limits


# --- Functions ---
def stick_shapes_to_grid(
        gdf: gpd.GeoDataFrame,
        *,
        cell_width: float | int,
        cell_height: float | int,
        n_cols: float | int,
        n_rows: float | int,
        xmin: float | int,
        ymin: float | int,
        print_progress: bool = True,
        _test_grid_layout: bool = False,
        **kwargs
        ) -> pd.DataFrame:
    """Record which shapes are in each grid cell.

    For each cell in turn, clip the input geometry to the cell bounds.
    Record the shapes in the clipped area and the area that each
    covers.

    Resulting dataframe only stores data for non-empty cells.

    **kwargs allows for calling the function with a dictionary
    that contains unnecessary keys,
    e.g. transform_coords_to_grid(coords, **transform_dict).
    The *identifer makes sure that cell properties are passed
    as keywords and so their order doesn't matter.

    TO DO - make it clear whether col and row are starting in top
    left or wherever.
    TO DO - define cells in advance?

    Inputs
    ------
    gdf               - gpd.GeoDataFrame. Contains shapes to be stuck
                        to the grid.
    cell_width        - float | int. Width of cells for this grid.
    cell_height       - float | int. Height of cells for this grid.
    n_cols            - float | int. Number of columns in the grid.
    n_rows            - float | int. Number of rows in the grid.
    xmin              - float | int. x bound of the grid.
    ymin              - float | int. y bound of the grid.
    print_progress    - bool. Optional. Print how many cells so far
                        have been checked.
    _test_grid_layout - bool. Optional. Don't use shapes, just assign
                        row and column to each cell.

    Returns
    -------
    df_props - pd.DataFrame. Contains a row for each cell that contains
               any shape. Columns record cell location (row, column),
               contents (shapes, proportions of filled area taken up
               by each shape), proportion of filled area.
    """

    cell_area = cell_width * cell_height
    cell_props = []
    cell_label = 0
    for col in range(n_cols):
        for row in range(n_rows):
            if print_progress:
                perc_done = 100.0 * (cell_label + 1) / (n_rows * n_cols)
                print(f'{perc_done:5.1f}% of cells checked', end='\r')
            else:
                pass

            if _test_grid_layout:
                cell_props.append((cell_label, row, col))
            else:
                # Limit gdf to this pixel:
                bbox = set_up_cell_limits(
                    col, row, xmin, ymin, cell_width, cell_height
                    )
                gdf_here = gpd.clip(gdf, bbox)

                if len(gdf_here) > 0:
                    # Sanity check - are there intersecting shapes?
                    pairs = find_overlapping_polygons(gdf_here)
                    warn_overlap_here = 1 if len(pairs) > 0 else 0

                    prop_cell_covered = round(
                        gdf_here.area.sum() / cell_area, 3)
                    series_area_props = find_area_props(gdf_here).round(3)
                    # If this index is returned as a pandas StringArray
                    # (experimental feature at time of writing)
                    # then turn it into a standard array:
                    r = series_area_props.index.values
                    try:
                        r = r.to_numpy().tolist()
                    except AttributeError:
                        pass
                    # Majority assumes that series is sorted by
                    # descending prop.
                    cell_props.append((
                        cell_label,
                        row,
                        col,
                        r,
                        series_area_props.values,
                        prop_cell_covered,
                        series_area_props.index.values[0],  # majority
                        warn_overlap_here,
                    ))
                else:
                    pass
            cell_label += 1

    # Convert the results to dataframe:
    if _test_grid_layout:
        cols = ('cell_label', 'cell_row', 'cell_col')
    else:
        cols = ('cell_label', 'cell_row', 'cell_col',
                'label_props', 'area_props', 'prop_cell_covered',
                'label_majority', 'overlapping_polys')
    df_props = pd.DataFrame(cell_props, columns=cols).set_index('cell_label')

    n_overlapping = df_props['overlapping_polys'].sum()
    if n_overlapping > 0:
        warnings.warn((
            'Overlapping polygons present '
            f'in {n_overlapping:.0f} cells: '
            'overlap area will be counted twice.'
        ), UserWarning)
    else:
        pass
    return df_props


def find_overlapping_polygons(gdf: gpd.GeoDataFrame) -> np.array:
    """Find pairs of shapes here that overlap.

    Use this to check whether any shapes overlap and so whether to
    check results about the areas of the cell covered by the shapes.
    For example, the total area covered could be much greater than
    the area of the cell if lots of regions are double-counted.

    Assumes that shape labels are in the gdf index.

    Inputs
    ------
    gdf - gpd.GeoDataFrame. Contains shapes that share an area.
          Index is the labels for the shapes.

    Returns
    -------
    pairs - np.array. Each element is a list of the names of a pair
            of shapes that overlap.
    """
    # predicate='overlaps' means if two shapes touch boundaries,
    # then it doesn't count as overlapping. They are only flagged
    # if their interiors overlap.
    c = gdf.index.name
    gdf_overlap = gpd.sjoin(gdf, gdf, predicate='overlaps').reset_index()
    mask = gdf_overlap[f'{c}_left'] != gdf_overlap[f'{c}_right']
    gdf_overlap = gdf_overlap[mask]

    pairs = gdf_overlap[[f'{c}_left', f'{c}_right']].values
    # Remove duplicates in different orders:
    pairs = np.unique(np.sort(pairs))
    return pairs


def find_area_props(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Calculate proportion of this gdf covered by each shape.

    Note that these are proportions of the total shape area in gdf,
    not the total area covered by the bounding box of the gdf.
    Empty space within the bounding box and not covered by a shape
    is not included in these calculations.

    Results are sorted by decreasing proportion.

    Inputs
    ------
    gdf - gpd.GeoDataFrame. Contains shapes that share an area.
          Index is the labels for the shapes.

    Returns
    -------
    s_area_props - pd.Series. One row for each shape in the gdf.
                   Values are the proportion of the total covered
                   area that each shape covers.
    """
    # Find proportion of the constituent LSOAs by area:
    s_area_props = gdf.area / gdf.area.sum()
    # Sort by decreasing proportion:
    s_area_props = s_area_props.sort_values(ascending=False)
    return s_area_props
