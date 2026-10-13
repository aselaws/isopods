"""
"""
# --- Imports ---
import numpy as np


# --- Functions ---
def make_grid_bounds_containing_bbox(
        bbox_shapes: list | tuple,
        cell_width: float | int,
        cell_height: float | int
        ) -> list[float | int]:
    """
    """
    xmin, ymin, xmax, ymax = bbox_shapes
    bbox = [
        (xmin // cell_width) * cell_width,            # xmin
        (ymin // cell_height) * cell_height,          # ymin
        ((xmax // cell_width) + 1.0) * cell_width,    # xmax
        ((ymax // cell_height) + 1.0) * cell_height,  # ymax
        ]
    return bbox


def make_transform_dict(
        bbox: list[float | int] | tuple[float | int, ...],
        cell_width: float | int,
        cell_height: float | int,
        crs_name: str = '',
        crs_epsg: str = ''
        ) -> dict:
    """
    bbox - left, bottom, right, top

    Should this be calculating number of columns and rows
    or calculating the extent or what?
    Should bbox passed in be snappped to sensible grid
    boundaries in advance?
    """
    xmin, ymin, xmax, ymax = bbox
    n_cols = int(np.ceil((xmax - xmin) // cell_width))
    n_rows = int(np.ceil((ymax - ymin) // cell_height))

    # Store the parameters for scaling the raster image
    # to match the original vector image:
    transform_dict = {
        # For the vector image:
        'xmin': xmin,
        # 'ymin': ymin,
        # 'xmax': xmax,
        'ymax': ymax,
        # For the raster image:
        'cell_width': cell_width,
        'cell_height': cell_height,
        'n_cols': n_cols,
        'n_rows': n_rows,
        'xmax': xmin + (cell_width * n_cols),
        'ymin': ymax - (cell_height * n_rows),
    }
    if len(crs_name) > 0:
        transform_dict['crs_name'] = crs_name
    else:
        pass
    if len(crs_epsg) > 0:
        transform_dict['epsg'] = crs_epsg
    else:
        pass
    return transform_dict


def set_up_cell_limits(
        col: int,
        row: int,
        xmin: float | int,
        ymin: float | int,
        cell_width: float | int,
        cell_height: float | int
        ) -> tuple[float | int, ...]:
    """Set up scaled bounds of this cell (row, col).


    """
    # Usual bbox is xmin, ymin, xmax, ymax:
    bbox = (
        xmin + cell_width * col,         # left
        ymin + cell_height * row,        # bottom
        xmin + cell_width * (col + 1),   # right
        ymin + cell_height * (row + 1),  # top
        )
    return bbox


def transform_coords_to_grid(
        coords: np.array,
        *,
        cell_width: float | int,
        cell_height: float | int,
        n_rows: float | int,
        xmin: float | int,
        ymax: float | int,
        **kwargs
        ) -> np.array:
    """Change pixel coordinates to match grid.

    d is transform_dict."""
    # Current coordinates:
    x = coords[:, 0]
    y = coords[:, 1]
    # Transformed coordinates:
    x = xmin + x * cell_width
    y = ymax - (n_rows - y) * cell_height
    # Gather transformed coords:
    new_coords = np.stack((x, y), axis=1)
    return new_coords


def calculate_cell_centres(
        zmin: float | int,
        zmax: float | int,
        cell_size: float | int
        ) -> np.array:
    z = np.arange(zmin, zmax, cell_size) + 0.5 * cell_size
    return z
