import re
import pandas as pd


def optimize_special(df: pd.DataFrame) -> None:
    """
    Identify and report columns requiring special handling based on content patterns.

    This helper function performs pattern-based analysis to detect columns that
    typically require domain-specific handling and should not undergo standard
    optimizations. The function uses regular expressions to match common naming
    conventions and analyzes column characteristics (cardinality, data type) to
    classify columns into several special categories:

    1. **Unique Identifiers**:
       Columns with names such as 'id', 'uuid', or 'key' that have very high
       cardinality (close to one unique value per row). These columns are reported
       but not modified to preserve referential integrity.

    2. **Geographic Coordinates**:
       Columns named 'latitude', 'longitude', 'lat', or 'lon' that typically store
       geographic coordinate data. These are usually already optimized to a float
       dtype by numeric optimization and are flagged for user awareness.

    3. **Text Entities**:
       High-cardinality string columns that do not match ID patterns, often
       representing names, addresses, or free-form text. These remain as object
       dtype because categorical conversion would be inefficient.

    4. **Categorical or Ordinal Data**:
       Columns that are already of pandas 'category' dtype, potentially representing
       nominal or ordinal categories.

    This function does not modify the DataFrame. It prints informative messages
    to help users understand their data structure and the optimization decisions
    made by the package.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze for special column patterns.

    Returns
    -------
    None
        This function prints its findings to stdout and does not return a value.

    Notes
    -----
    - Uses regex patterns to match common naming conventions
    - Analyzes both column names and data characteristics
    - Provides tag output for easy visual scanning:
      <Unique ID>
      <Coordinates>
      <Text Entity>
      <Categorical/Ordinal>
    - Detection heuristics may not catch all special cases; review output carefully

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'customer_id': range(1000),
    ...     'latitude': [37.7749] * 1000,
    ...     'longitude': [-122.4194] * 1000,
    ...     'full_name': [f'Person {i}' for i in range(1000)],
    ...     'membership_level': pd.Categorical(
    ...         ['gold', 'silver', 'bronze'] * 333 + ['gold']
    ...     )
    ... })
    >>> 
    >>> optimize_special(df)
    
    --- Special Column Analysis ---
    customer_id: Identified as potential Unique ID (high cardinality).
    latitude: Identified as geographic coordinate column.
    longitude: Identified as geographic coordinate column.
    full_name: Identified as high-cardinality text column.
    membership_level: Identified as categorical or ordinal data (category dtype).

    >>> df2 = pd.DataFrame({
    ...     'uuid': ['a1b2c3'] * 500 + ['d4e5f6'] * 500,
    ...     'order_key': range(1000),
    ...     'lat': [40.7128] * 1000,
    ...     'delivery_address': [f'{i} Main St' for i in range(1000)]
    ... })
    >>> 
    >>> optimize_special(df2)
    
    --- Special Column Analysis ---
    order_key: Identified as potential Unique ID (high cardinality).
    lat: Identified as geographic coordinate column.
    delivery_address: Identified as high-cardinality text column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    print("\n--- Special Column Analysis ---")

    n_rows = len(df)
    if n_rows == 0:
        print("(DataFrame is empty)")
        return None

    # Common patterns for identifiers (case-insensitive)
    id_regex = re.compile(r"(?:^|_)(id|uuid|key)(?:$|_)", flags=re.IGNORECASE)

    # Common coordinate column names
    coord_names = {"lat", "latitude", "lon", "long", "longitude"}

    for col in df.columns:
        name = str(col)
        s = df[col]

        # Already categorical
        if pd.api.types.is_categorical_dtype(s):
            print(f"<Categorical/Ordinal>{name}: Identified as categorical or ordinal data (category dtype).")
            continue

        # Geographic coordinates (by name)
        if name.strip().lower() in coord_names:
            print(f"<Coordinates>{name}: Identified as geographic coordinate column.")
            continue

        # Cardinality-based checks
        nunique = s.nunique(dropna=False)
        unique_ratio = nunique / n_rows if n_rows > 0 else 0

        # Potential unique identifier
        if id_regex.search(name) and unique_ratio >= 0.9:
            print(f"<Unique ID>{name}: Identified as potential Unique ID (high cardinality).")
            continue

        # High-cardinality text column
        if pd.api.types.is_object_dtype(s) and unique_ratio > 0.5 and not id_regex.search(name):
            print(f"<Text Entity>{name}: Identified as high-cardinality text column.")
            continue

    return None
