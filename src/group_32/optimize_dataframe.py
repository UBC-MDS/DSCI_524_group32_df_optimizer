from __future__ import annotations

import pandas as pd

from .optimize_numeric import optimize_numeric
from .optimize_categorical import optimize_categorical
from .optimize_special import optimize_special

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates an optimized copy of a pandas DataFrame by applying
    a series of memory-reduction strategies while preserving the original data.

    It serves as the main wrapper function for the package and coordinates
    multiple optimization steps:
    - Numeric columns are downcast to smaller, safe data types where possible
    - Low-cardinality string columns are converted to pandas 'category' dtype
    - Special columns (e.g., IDs, coordinates, high-cardinality text) are
      identified and reported but not modified

    The original DataFrame is never mutated; all operations are performed
    on a copy.

    Parameters
    ----------
    df : pd.DataFrame
        The input pandas DataFrame to be optimized.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with optimized data types and reduced memory usage.

    Notes
    -----
    - This function acts as a wrapper that calls lower-level helper functions
      such as numeric and categorical optimizers.
    - The optimization process is designed to be transparent and reproducible;
      a summary of changes and memory savings may be printed.
    - This function prioritizes safety over aggressiveness and avoids modifying
      columns that could lead to unexpected behavior.
    - The optimized DataFrame should behave identically to the original in
      downstream analysis, aside from potential minor float precision changes
      if numeric downcasting is applied.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "status": ["pending", "shipped", "pending"],
    ...     "quantity": [1, 2, 3]
    ... })
    >>> optimized_df = optimize_dataframe(df)
    >>> optimized_df.dtypes["status"]
    CategoricalDtype(categories=['pending', 'shipped'], ordered=False)
    >>> optimized_df.dtypes["quantity"]
    dtype('int8')

    >>> df = pd.DataFrame({
    ...     "region": ["US", "CA", "US", "US"],
    ...     "price": [10.5, 12.0, 9.99, 11.25]
    ... })
    >>> optimized_df = optimize_dataframe(df)
    >>> optimized_df.dtypes["region"]
    CategoricalDtype(categories=['CA', 'US'], ordered=False)
    >>> optimized_df.dtypes["price"]
    dtype('float32')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Always work on copies (helpers already copy, but keep wrapper intent explicit)
    out = df.copy(deep=True)

    # Apply safe optimizations
    out = optimize_numeric(out)
    out = optimize_categorical(out, max_unique_ratio=0.5)

    # Report special columns (no mutation)
    optimize_special(out)

    return out
