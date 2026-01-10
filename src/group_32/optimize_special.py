def _analyze_special_columns(df: pd.DataFrame):
    """
    Identify and report columns requiring special handling based on content patterns.

    This internal helper function performs pattern-based analysis to detect columns
    that typically require domain-specific handling and should not undergo standard
    optimizations. The function uses regular expressions to match common naming
    conventions and analyzes column characteristics (cardinality, data type) to
    classify columns into several special categories:

    1. **Unique Identifiers**: Columns with names like 'id', 'uuid', 'customer_key'
       that have high cardinality (one unique value per row). These should remain
       in their original format to preserve referential integrity.

    2. **Geographic Coordinates**: Columns named 'latitude', 'longitude', 'lat', 'lon'
       containing floating-point coordinates. These are already optimized to float32
       by the numeric optimization step and are flagged for user awareness.

    3. **Text Entities**: High-cardinality string columns that don't match ID patterns,
       typically representing names, addresses, or free-form text. These remain as
       object dtype because categorical conversion would be inefficient.

    4. **Categorical/Ordinal Data**: Columns already converted to category dtype,
       potentially representing ordered categories (e.g., 'small', 'medium', 'large')
       or nominal categories (e.g., 'red', 'blue', 'green').

    The function does not modify any columns but provides informative output to help
    users understand their data structure and the optimization decisions made.

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
    - Provides emoji-coded output for easy visual scanning:
      :pushpin: = Unique ID
      :earth_africa: = Coordinates
      :memo: = Text Entity
      :1234: = Categorical/Ordinal
    - Detection heuristics may not catch all special cases; review output carefully

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'customer_id': range(1000),  # unique IDs
    ...     'latitude': [37.7749] * 1000,
    ...     'longitude': [-122.4194] * 1000,
    ...     'full_name': [f'Person {i}' for i in range(1000)],  # high cardinality text
    ...     'membership_level': pd.Categorical(['gold', 'silver', 'bronze'] * 333 + ['gold'])
    ... })
    >>> 
    >>> _analyze_special_columns(df)
    
    --- Special Column Analysis ---
    📌 customer_id: Identified as potential **Unique ID**. High cardinality (not optimized to 'category').
    🌍 latitude: Identified as **Latitude/Longitude**. Already optimized to a float dtype.
    🌍 longitude: Identified as **Latitude/Longitude**. Already optimized to a float dtype.
    📝 full_name: Identified as **Text Entity (Name/Address)**. Stays as string/object due to high variability.
    🔢 membership_level: **Categorical/Ordinal** (Type is 'category').
    
    >>> # Another example with different patterns
    >>> df2 = pd.DataFrame({
    ...     'uuid': ['a1b2c3'] * 500 + ['d4e5f6'] * 500,
    ...     'order_key': range(1000),
    ...     'lat': [40.7128] * 1000,
    ...     'delivery_address': [f'{i} Main St' for i in range(1000)]
    ... })
    >>> 
    >>> _analyze_special_columns(df2)
    
    --- Special Column Analysis ---
    📌 order_key: Identified as potential **Unique ID**. High cardinality (not optimized to 'category').
    🌍 lat: Identified as **Latitude/Longitude**. Already optimized to a float dtype.
    📝 delivery_address: Identified as **Text Entity (Name/Address)**. Stays as string/object due to high variability.
    
    """