"""
Define grids in the reference frame of the shapes.

TO DO - should transform dict function accept various inputs and
calculate the missing bits? Currently accepts bounds and cell size
and calculates numbers of columns/rows, but perhaps offer the
other way round (calculate cell size given target numbers of rows).

TO DO - potentially one day allow for non-rectangular grid cells.
"""
# --- Imports ---
import numpy as np


# --- Functions ---
def make_grid_bounds_containing_bbox(
        bbox_shapes: list[float | int] | tuple[float | int, ...],
        cell_width: float | int,
        cell_height: float | int
        ) -> list[float | int]:
    """Calculate nice grid limits that contain this bbox.

    Find grid limits that contain the bounding box of some set of
    shapes and that start and end at nice round numbers. For
    example, for a cell size of 1000, the limits will be rounded
    down (for xmin, ymin) or up (xmax, ymax) to the nearest multiple
    of 1000.

    Inputs
    ------
    bbox_shapes - list | tuple. Bounding box (xmin, ymin, xmax, ymax).
    cell_width  - float | int. Width of cells for this grid.
    cell_height - float | int. Height of cells for this grid.

    Returns
    -------
    bbox - list | tuple. Bounding box (xmin, ymin, xmax, ymax) that
           just contains the input bbox and has its limits rounded
           to multiples of cell width and cell height.
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
        **kwargs
        ) -> dict:
    """Gather information on grid dimensions into a dictionary.

    Dict contains grid extent, cell size, and number of rows and
    columns. Any kwargs provided are also stored. Recommended kwargs
    are crs name and EPSG.

    Inputs
    ------
    bbox        - list | tuple. Bounding box (xmin, ymin, xmax, ymax)
                  for the grid itself, not the shapes it will contain.
    cell_width  - float | int. Width of cells for this grid.
    cell_height - float | int. Height of cells for this grid.
    crs_name    - str. Optional. Name of the crs projection that this
                  grid is intended to work for.
    crs_epsg    - str | int. Optional. Name of the ESPG that this
                  grid is intended to work for.

    Returns
    -------
    transform_dict - dict. Dictionary of grid coordinate information.
    """
    xmin, ymin, xmax, ymax = bbox
    # Calculate numbers of rows and columns that fit in this extent:
    n_cols = int(np.ceil((xmax - xmin) // cell_width))
    n_rows = int(np.ceil((ymax - ymin) // cell_height))
    # Recalculate grid limits to ensure that an integer number of cells
    # fit within the limits.
    xmax = xmin + (cell_width * n_cols)
    ymin = ymax - (cell_height * n_rows)

    transform_dict = {
        'xmin': xmin,
        'ymax': ymax,
        'cell_width': cell_width,
        'cell_height': cell_height,
        'n_cols': n_cols,
        'n_rows': n_rows,
        'xmax': xmax,
        'ymin': ymin,
    }
    for k, v in kwargs:
        transform_dict[k] = v
    return transform_dict


def set_up_cell_limits(
        col: int,
        row: int,
        xmin: float | int,
        ymin: float | int,
        cell_width: float | int,
        cell_height: float | int
        ) -> tuple[float | int, ...]:
    """Calculate bounds of rectangular cell in this row and column.

    Assume that this cell starts and ends an integer number of cell
    widths and heights from the starting point of the grid.

    Inputs
    ------
    col         - int. Column index for this cell.
    row         - int. Row index for this cell.
    xmin        - float | int. x bound of the grid.
    ymin        - float | int. y bound of the grid.
    cell_width  - float | int. Width of cells for this grid.
    cell_height - float | int. Height of cells for this grid.

    Returns
    -------
    bbox - tuple. Bounding box for this cell.
    """
    # Usual bbox is xmin, ymin, xmax, ymax:
    bbox = (
        xmin + cell_width * col,         # xmin, left
        ymin + cell_height * row,        # ymin, bottom
        xmin + cell_width * (col + 1),   # xmax, right
        ymin + cell_height * (row + 1),  # ymax, top
        )
    return bbox


def transform_coords_to_grid(
        coords: np.ndarray,
        *,
        cell_width: float | int,
        cell_height: float | int,
        n_rows: float | int,
        xmin: float | int,
        ymax: float | int,
        **kwargs
        ) -> np.ndarray:
    """Transform coordinates from pixel to projected coordinates.

    Assumes that the first cell is the 0th row and 0th column.

    **kwargs allows for calling the function with a dictionary
    that contains unnecessary keys,
    e.g. transform_coords_to_grid(coords, **transform_dict).
    The *identifer makes sure that cell properties are passed
    as keywords and so their order doesn't matter.

    Inputs
    ------
    coords      - np.ndarray. Stacked coordinates in pixel units.
                  First column is x values, second is y.
    cell_width  - float | int. Width of cells for this grid.
    cell_height - float | int. Height of cells for this grid.
    n_rows      - float | int. Number of rows in the grid.
    xmin        - float | int. x bound of the grid.
    ymax        - float | int. y bound of the grid.
    **kwargs    - dict. These go unused.

    Returns
    -------
    new_coords - np.ndarray. Transformed coordinates. First column
                 is x values, second is y.
    """
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
    """Calculate centres of this line of cells given start and size.

    Assumes that all cells are the same width and that the required
    centre is halfway between the start and end of a cell.

    Inputs
    ------
    zmin      - float | int. Start of the first cell.
    zmax      - float | int. End of the final cell.
    cell_size - float | int. Size of each cell.

    Returns
    -------
    z - np.array. Midpoints of all cells in this line.
    """
    z = np.arange(zmin, zmax, cell_size) + 0.5 * cell_size
    return z
