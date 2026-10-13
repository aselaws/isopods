"""
"""
# --- Imports ---
import numpy as np
import pandas as pd

# --- Functions ---
def convert_df_to_2darray(
        df: pd.DataFrame,
        data_col: str,
        # transform_dict: dict,                        # CHANGE THIS to input rows and cols directly
        *,
        n_rows: float|int,
        n_cols: float|int,
        empty_value: any = None,
        **kwargs
    ) -> np.ma.masked_array:
    """
    Default value for empty cells depends on data type.
    """
    d = df[data_col].dtype
    if type(d) == pd.StringDtype:
        d = object
    elif type(d) == pd.Float64Dtype:
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
        arr_old: np.array|np.ma.masked_array,
        dict_values: dict,
        empty_value: any = None
        ) -> np.array|np.ma.masked_array:
    """
    TO DO - check if these look like colours. Check data types of
    incoming dict values.

    TO DO - check if all options in arr_values are present
    in keys of dict_colours.
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


def _check_incoming_dtype(values: list|tuple):
    """Check that all values have same/similar dtype(s)."""
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
            raise TypeError(('New values should be same type. '
                             f'Found {types_here}.'))


def _get_incoming_dtype(values: list|tuple):  # -> dtype:
    """
    """
    types_here = list(set([type(v) for v in values]))
    if len(types_here) == 1:
        d = types_here[0]
    elif any([(type(d) == pd.StringDtype) for d in types_here]):
        d = pd.StringDtype
    elif any([(type(d) == pd.Float64Dtype) for d in types_here]):
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

    if type(d) == pd.StringDtype:
        d = object
    return d


def _generate_empty_value(d):#: dtype) -> any:
    """
    For colours, use _generate_empty_colour.
    """
    if type(d) == pd.StringDtype:
        empty_value = ''
    elif type(d) == pd.Float64Dtype:
        empty_value = np.nan
    elif np.issubdtype(d, object) | np.issubdtype(d, str):
        empty_value = ''
    elif np.issubdtype(d, np.integer):
        empty_value = -99
    else:
        empty_value = np.nan
    return empty_value


def _generate_empty_colour(colour_example: list|tuple) -> list|tuple:
    """
    """
    # Define colour for empty cells in same format as dict_colours.
    c_len = len(colour_example)
    f = np.issubdtype(type(colour_example[0]), float)
    v = 1.0 if f else 255
    a = 0.0 if f else 0
    if c_len > 3:
        colour_empty = (v, v, v, a)  # RGBA transparent
    else:
        colour_empty = (v, v, v)  # RGB white
    return colour_empty
    

def _check_colours_valid(colours: list|tuple):
    """
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
    ) -> pd.DataFrame:
    """
    Expect index to be cell label.

    Assumes that area props sum to 1.
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
    """
    """
    ys, xs = np.where(arr == label)
    # Pick the middle row:
    i = int(0.5*len(ys))
    y_med = ys[i]
    # Pick the middle column in that row:
    j = int(0.5*len(xs[ys == y_med]))
    x_med = xs[ys == y_med][j]
    return (x_med, y_med)


def pick_midpoints_for_regions(
        arr: np.ndarray, labels_unique: list|tuple
        ):
    """Wrapper for pick_midpoint().
    """
    coords = np.empty((len(labels_unique), 2), dtype=int)
    for i, label in enumerate(labels_unique):
        coords[i] = pick_midpoint(arr, label)
    return coords