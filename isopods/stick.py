"""
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
        # td: dict,                              # to do - update this in notebooks
        *,
        cell_width: float|int,
        cell_height: float|int,
        n_cols: float|int,
        n_rows: float|int,
        xmin: float|int,
        ymin: float|int,
        print_progress: bool = True,
        _test_grid_layout: bool = False,
        **kwargs
        ) -> pd.DataFrame:
    """
    Assign shapes to grid cells.
    
    Set up grid. Bounds of each cell.

    td is transform_dict.

    df_props only stores data for non-empty cells.

    TO DO - make it clear whether col and row are starting in top
    left or wherever.
    """    
    
    cell_area = cell_width * cell_height
    cell_props = []
    cell_label = 0
    raise_overlap_warning = False
    for col in range(n_cols):
        for row in range(n_rows):
            if print_progress:
                perc_done = (
                    100.0 * (cell_label + 1) /
                    (n_rows * n_cols)
                )
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
                    if len(pairs) > 0:
                        raise_overlap_warning = True
                        warn_overlap_here = 1
                    else:
                        warn_overlap_here = 0
                    
                    prop_cell_covered = round(gdf_here.area.sum() / cell_area, 3)
                    series_area_props = find_area_props(gdf_here).round(3)
                    # If this index is returned as a pandas StringArray
                    # (experimental feature at time of writing)
                    # then turn it into a standard array:
                    r = series_area_props.index.values
                    try:
                        r = r.to_numpy().tolist()
                    except AttributeError:
                        pass
                    # Majority assumes that series is sorted by descending prop.
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

    if raise_overlap_warning:
        n_overlapping = df_props['overlapping_polys'].sum()
        warnings.warn((
            'Overlapping polygons present '
            f'in {n_overlapping:.0f} cells: '
            'overlap area will be counted twice.'
        ), UserWarning)
    else:
        pass
    return df_props


def find_overlapping_polygons(gdf: gpd.GeoDataFrame) -> np.array:
    """
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
    """

    Results sorted by decreasing proportion.

    Proportion of the total shape area in gdf_here,
    not its bounding box for example. Empty space ignored.
    """
    # Find proportion of the constituent LSOAs by area:
    s_area_props = gdf.area / gdf.area.sum()
    # Sort by decreasing proportion:
    s_area_props = s_area_props.sort_values(ascending=False)
    return s_area_props