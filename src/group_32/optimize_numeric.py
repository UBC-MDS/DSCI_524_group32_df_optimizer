import pandas as pd

def optimize_numeric(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast integer and float columns to the smallest suitable numeric dtype.

    This internal helper function processes all numeric columns in the DataFrame,
    attempting to reduce their memory footprint through intelligent downcasting.
    The function uses pandas' built-in downcast functionality to safely convert
    numeric types to smaller representations while preserving values within acceptable precision limits.

    For integer columns:
    - int64 → int32, int16, or int8 (depending on value range).
    - uint64 → uint32, uint16, or uint8 (for unsigned integers).

    For float columns:
    - float64 → float32 (potentially introducing minor precision loss).

    The downcasting process examines the actual range of values in each column
    and selects the smallest dtype that can accommodate all values. For example,
    a column with values ranging from 0-255 stored as int64 (8 bytes) can be
    safely downcast to uint8 (1 byte), achieving 87.5% memory reduction.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing numeric columns to be optimized.
    verbose : bool, default True
        If True, prints memory usage statistics before and after optimization.

    Returns
    -------
    pd.DataFrame
        The DataFrame with optimized numeric column dtypes.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> df = pd.DataFrame({
    ...     "int_col": np.array([1, 2, 3], dtype=np.int64),
    ...     "big_int_col": np.array([1000, 2000, 3000], dtype=np.int64),
    ...     "float_col": np.array([0.1, 0.2, 0.3], dtype=np.float64),
    ...     "non_numeric": ["a", "b", "c"]
    ... })
    >>>
    >>> df.dtypes
    int_col          int64
    big_int_col      int64
    float_col      float64
    non_numeric     object
    dtype: object
    >>>
    >>> optimized_df = optimize_numeric(df)
    Memory reduced from 0.XX MB to 0.XX MB (XX.X% reduction)
    >>> optimized_df.dtypes
    int_col            int8
    big_int_col       int16
    float_col       float32
    non_numeric       object
    dtype: object

    Notes
    -----
    - Uses try-except to skip columns that cannot be safely downcast.
    - Float downcasting from float64 to float32 may introduce minor precision loss.
    - Integer downcasting is lossless when values fit in the target range.
    - Failed conversions are silently skipped, preserving original dtypes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Calculate initial memory usage
    initial_memory = df.memory_usage(deep=True).sum()
    
    out = df.copy(deep=True)

    # Process all numeric columns in one pass
    numeric_cols = out.select_dtypes(include=['int', 'uint', 'float']).columns
      
    for col in numeric_cols:
        try:
            # Check if integer type
            if out[col].dtype.kind in ['i', 'u']:  # 'i' for signed int, 'u' for unsigned
                out[col] = pd.to_numeric(out[col], downcast='integer')
            # Check if float type
            elif out[col].dtype.kind == 'f':
                out[col] = pd.to_numeric(out[col], downcast='float')
        except Exception:
            # Silently skip columns that cannot be downcast
            pass

    # Calculate final memory usage and report
    if verbose:
        final_memory = out.memory_usage(deep=True).sum()
        reduction_pct = ((initial_memory - final_memory) / initial_memory) * 100
        print(f"Memory reduced from {initial_memory / 1024**2:.2f} MB to "
              f"{final_memory / 1024**2:.2f} MB ({reduction_pct:.1f}% reduction)")

    return out

