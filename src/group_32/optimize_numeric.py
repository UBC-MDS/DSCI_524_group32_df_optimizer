import pandas as pd

def optimize_numeric(df: pd.DataFrame) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        The DataFrame with optimized numeric column dtypes.

    Notes
    -----
    - Uses 'errors=ignore' to skip columns that cannot be safely downcast.
    - Float downcasting from float64 to float32 may introduce minor precision loss.
    - Integer downcasting is lossless when values fit in the target range.
    - Prints confirmation message upon successful completion.
    """
    pass