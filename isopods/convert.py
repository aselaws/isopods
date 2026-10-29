"""Convert the data contained in a grid.
"""
# --- Imports ---
import warnings
import numpy as np
import pandas as pd


# --- Functions ---
def convert_df_to_2darray(
        df: pd.DataFrame,
        data_col: str,
        *,
        n_rows: float | int,
        n_cols: float | int,
        empty_value: any = None,
        **kwargs
        ) -> np.ma.masked_array:
    """Convert dataframe of grid info to grid layout.

    Create a new array to hold the grid data using the input numbers
    of rows and columns and with the data type appropriate for the
    data column in the dataframe. Fill the grid using the cell info
    and data from the dataframe. Default value for empty cells depends
    on data type. The resulting grid is a masked array to flag cells
    that do not contain any shapes.

    Input dataframe of grid info is output from
    stick.stick_shapes_to_grid().

    **kwargs allows for calling the function with a dictionary
    that contains unnecessary keys,
    e.g. transform_coords_to_grid(coords, **transform_dict).
    The *identifer makes sure that cell properties are passed
    as keywords and so their order doesn't matter.

    Inputs
    ------
    df          - pd.DataFrame. One row per cell that contains a shape.
                  Columns include row and column index and the data
                  to be placed into the grid.
    data_col    - str. Column in df to place in the grid.
    n_cols      - float | int. Number of columns in the grid.
    n_rows      - float | int. Number of rows in the grid.
    empty_value - any. Optional value for cells that contain no shapes.

    Returns
    -------
    marr - np.ma.masked_array. data attribute holds the df[data_col]
           values, and mask attribute flags whether cells contain any
           shapes.
    """
    d = df[data_col].dtype
    if isinstance(d, pd.StringDtype):
        d = object
    elif isinstance(d, pd.Float64Dtype):
        d = float
    else:
        pass

    if empty_value is None:
        empty_value = _generate_empty_value(d)
    else:
        pass

    # Make a 2D array with all pixels, not just valid ones:
    arr = np.full((n_rows, n_cols), empty_value, dtype=d)
    # Update the values of valid pixels:
    arr[df['cell_row'], df['cell_col']] = df[data_col].values
    # Make mask to record invalid cells:
    mask = np.full(arr.shape, 1, dtype=int)
    mask[df['cell_row'], df['cell_col']] = 0
    # Combine arr and mask into masked array:
    marr = np.ma.masked_array(arr, mask=mask)
    return marr


def convert_grid_values(
        arr_old: np.array | np.ma.masked_array,
        dict_values: dict,
        empty_value: any = None
        ) -> np.array | np.ma.masked_array:
    """Convert grid data using a dictionary.

    This function checks that the data in the input
    dictionary has consistent data types.

    TO DO - check if all options in arr_values are present
    in keys of dict_colours.

    Inputs
    ------
    arr_old     - np.array | np.ma.masked_array. Existing grid to use
                  as base for new grid data.
    dict_values - dict. Map of old grid data to new grid data.
    empty_value - any. Optional value for cells that contain no
                  data in the new grid.

    Returns
    -------
    arr_new - np.array | np.ma.masked_array. Type is same as arr_old.
              New grid of data made by mapping dict_values to the old
              grid. If masked_array then the mask is taken from the
              old grid.
    """
    # Check: data type of incoming values.
    _check_incoming_dtype(list(dict_values.values()))
    value_type = _get_incoming_dtype(list(dict_values.values()))
    listlike = [tuple, list, np.ndarray]
    values_are_colours = any(
        [np.issubdtype(value_type, t) for t in listlike])
    value_example = list(dict_values.values())[0]

    if values_are_colours:
        _check_colours_valid(list(dict_values.values()))
        arr_shape = (*arr_old.shape, len(value_example))
        arr_dtype = type(value_example[0])
    else:
        arr_shape = arr_old.shape
        if np.issubdtype(value_type, str):
            # Make sure the array isn't initialised with a
            # string dtype that can't hold as many characters
            # as the length of the incoming strings.
            arr_dtype = object
        else:
            arr_dtype = value_type

    if empty_value is None:
        if values_are_colours:
            empty_value = _generate_empty_colour(value_example)
        else:
            empty_value = _generate_empty_value(value_type)
    else:
        pass

    # New data:
    arr_new = np.full(arr_shape, empty_value, dtype=arr_dtype)
    for old_value, new_value in dict_values.items():
        inds = (arr_old == old_value)
        arr_new[inds] = new_value
    # New mask:
    if isinstance(arr_old, np.ma.MaskedArray):
        if values_are_colours:
            mask_new = np.stack([arr_old.mask] * len(value_example), axis=2)
        else:
            mask_new = arr_old.mask
        arr_new = np.ma.masked_array(arr_new, mask=mask_new)
    else:
        pass
    return arr_new


def _check_incoming_dtype(values: list | tuple):
    """Check that all values have same/similar dtype(s).

    If only one dtype or all values are numeric, check passes.
    Otherwise, raise error to flag mixed data types.

    Inputs
    ------
    values - list | tuple.
    """
    types_here = list(set([type(v) for v in values]))
    if len(types_here) == 1:
        pass
    else:
        numeric_types = [int, float]
        all_numeric = all([
            any([np.issubdtype(t, n) for n in numeric_types])
            for t in types_here
            ])
        if all_numeric:
            pass
        else:
            # If they're mixed dtypes, bad.
            raise TypeError(
                f'New values should be same type. Found {types_here}.')


def _get_incoming_dtype(values: list | tuple) -> any:
    """Find the dtype of these values.

    Do a series of checks. If it's string, use object. It it's mixed
    numeric, use float. If objects are present, use object. Otherwise
    use the most common dtype in the list.

    Inputs
    ------
    values - list | tuple.

    Returns
    -------
    d - any. dtype that best describes the input values.
    """
    types_here = list(set([type(v) for v in values]))
    if len(types_here) == 1:
        d = types_here[0]
    elif any([isinstance(d, pd.StringDtype) for d in types_here]):
        d = pd.StringDtype
    elif any([isinstance(d, pd.Float64Dtype) for d in types_here]):
        d = float
    else:
        # Check for mix of float and int.
        f = any([np.issubdtype(t, float) for t in types_here])
        i = any([np.issubdtype(t, int) for t in types_here])
        if f & i:
            # Pick out the first float type:
            d = [t for t in types_here if np.issubdtype(t, float)][0]
        else:
            # This isn't ideal. If object present, return it.
            # Otherwise use the most common.
            t_o = [t for t in types_here if np.issubdtype(t, object)]
            if len(t_o) > 0:
                d = t_o[0]
            else:
                # Mode. From https://stackoverflow.com/a/28129716
                d = max(set(t_o), key=t_o.count)

    if isinstance(d, pd.StringDtype):
        d = object
    return d


def _generate_empty_value(d: any) -> any:
    """Pick value for empty cells with input data type.

    For colours, use _generate_empty_colour.

    Inputs
    ------
    d - any. A data type.

    Returns
    -------
    empty_value - any. Value to use for cells with no shapes in.
    """
    if isinstance(d, pd.StringDtype):
        empty_value = ''
    elif isinstance(d, pd.Float64Dtype):
        empty_value = np.nan
    elif np.issubdtype(d, object) | np.issubdtype(d, str):
        empty_value = ''
    elif np.issubdtype(d, np.integer):
        empty_value = -99
    else:
        empty_value = np.nan
    return empty_value


def _generate_empty_colour(colour_example: list | tuple) -> list | tuple:
    """Pick colour tuple for empty cells.

    For other values, use _generate_empty_value.

    Inputs
    ------
    colour_example - list | tuple. A colour tuple.

    Returns
    -------
    empty_value - any. Colour tuple to use for empty cells.
    """
    # Define colour for empty cells in same format as example colour.
    c_len = len(colour_example)
    f = np.issubdtype(type(colour_example[0]), float)
    v = 1.0 if f else 255
    a = 0.0 if f else 0
    if c_len > 3:
        colour_empty = (v, v, v, a)  # RGBA transparent
    else:
        colour_empty = (v, v, v)  # RGB white
    return colour_empty


def _check_colours_valid(colours: list | tuple):
    """Check that all colour tuples in list have same format.

    Check that all inputs colours are tuples, are of same length,
    and are in RGB or RGBA format. If any not, raise error.

    TO DO - could check whether they're all 0.0-1.0 or 0-255.

    Inputs
    ------
    colours - list | tuple. Many colour tuples.
    """
    all_c_len = []
    for colour_example in colours:
        listlike = (list, tuple, np.ndarray)
        colour_is_listlike = any([
            np.issubdtype(type(colour_example), t) for t in listlike])
        if colour_is_listlike:
            c_len = len(colour_example)
            if c_len in [3, 4]:
                all_c_len.append(c_len)
            else:
                raise TypeError(('Colour tuples must be of length 3 or 4. '
                                 f'Found {colour_example}.'))
        else:
            raise TypeError(('Colours must be RGB or RGBA tuples. '
                             f'Found {colour_example}.'))
    if len(set(all_c_len)) > 1:
        raise TypeError(('All colour tuples must be of same length.'))
    else:
        pass


def calculate_scaled_cell_values(
        df_props: pd.DataFrame,
        dict_region_value: dict,
        col_label_props: str = 'label_props',
        col_area_props: str = 'area_props',
        r: int = 3
        ) -> pd.Series:
    """Calculate new cell values from combination of cell contents.

    Expect index to be cell label.
    Assumes that area props sum to 1.

    Inputs
    ------
    df_props          - pd.DataFrame. One row per cell in the grid,
                        columns containing lists of regions present,
                        and the areas they take up.
    dict_region_value - dict. Value associated with each region.
                        These values will be used in the scaled sum.
    col_label_props   - str. Name of column containing region names.
    col_area_props    - str. Name of column containing area
                        proportions.
    r                 - int. Decimal places to round area proportions.

    Returns
    -------
    d - pd.Series. One row per cell in the grid. Values are the scaled
        sum of the contents of each cell.
    """
    df = pd.concat(
        [df_props[[c]].explode(c) for c in [col_label_props, col_area_props]],
        axis='columns'
    )
    # Place all values in the df:
    df['label_value'] = df[col_label_props].map(dict_region_value)
    # Scale all values by their contribution to the cell:
    df['area_value_scaled'] = df['label_value'] * df[col_area_props]
    # Sum scaled values for each cell:
    index_name = df.index.name
    d = df['area_value_scaled'].reset_index().groupby(index_name).sum()
    d = d.round(r)
    # Make sure the dtype is numeric, not object.
    # Not sure why the column is object by default.
    d = d.convert_dtypes()
    return d


def pick_midpoint(arr: np.array, label: any) -> tuple[int, int]:
    """Find a representative point for cells that equal "label".

    Assumes that a region is all of the cells with the value "label".
    The midpoint is the middlemost column of the middlemost row in the
    grid for cells that match "label".

    Inputs
    ------
    arr   - np.array. 2D grid of values.
    label - any. Some value contained in the grid.

    Returns
    -------
    (x_med, y_med) - tuple[int, int]. Coordinates of representative
                     middle pixel of the region matching the input
                     label.
    """
    ys, xs = np.where(arr == label)
    if len(ys) == 0:
        # This label isn't in the array.
        warnings.warn('Requested label for region midpoint is not in grid.',
                      UserWarning)
        y_med, x_med = (99999, 99999)
    else:
        # Pick the middle row:
        i = int(0.5*len(ys))
        y_med = ys[i]
        # Pick the middle column in that row:
        j = int(0.5*len(xs[ys == y_med]))
        x_med = xs[ys == y_med][j]
    return (x_med, y_med)


def pick_midpoints_for_regions(
        arr: np.ndarray, labels_unique: list | tuple
        ) -> np.ndarray:
    """Wrapper for pick_midpoint().

    Inputs
    ------
    arr           - np.ndarray. 2D grid of values.
    labels_unique - list | tuple. Labels of areas to find midpoints
                    for.

    Returns
    -------
    coords - np.ndarray. (x, y) coordinate for the midpoint of the
             region with each input label.
    """
    coords = np.empty((len(labels_unique), 2), dtype=int)
    for i, label in enumerate(labels_unique):
        coords[i] = pick_midpoint(arr, label)
    return coords
