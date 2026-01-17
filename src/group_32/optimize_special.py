import pandas as pd

def optimize_special(df: pd.DataFrame)-> None:
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
    ...     'customer_id': range(1000),  # unique IDs
    ...     'latitude': [37.7749] * 1000,
    ...     'longitude': [-122.4194] * 1000,
    ...     'full_name': [f'Person {i}' for i in range(1000)],  # high cardinality text
    ...     'membership_level': pd.Categorical(['gold', 'silver', 'bronze'] * 333 + ['gold'])
    ... })
    >>> 
    >>> optimize_special(df)
    
    --- Special Column Analysis ---
    customer_id: Identified as potential **Unique ID**. High cardinality (not optimized to 'category').
    latitude: Identified as **Latitude/Longitude**. Already optimized to a float dtype.
    longitude: Identified as **Latitude/Longitude**. Already optimized to a float dtype.
    full_name: Identified as **Text Entity (Name/Address)**. Stays as string/object due to high variability.
    membership_level: **Categorical/Ordinal** (Type is 'category').
    
    >>> # Another example with different patterns
    >>> df2 = pd.DataFrame({
    ...     'uuid': ['a1b2c3'] * 500 + ['d4e5f6'] * 500,
    ...     'order_key': range(1000),
    ...     'lat': [40.7128] * 1000,
    ...     'delivery_address': [f'{i} Main St' for i in range(1000)]
    ... })
    >>> 
    >>> optimize_special(df2)
    
    --- Special Column Analysis ---
    order_key: Identified as potential **Unique ID**. High cardinality (not optimized to 'category').
    lat: Identified as **Latitude/Longitude**. Already optimized to a float dtype.
    delivery_address: Identified as **Text Entity (Name/Address)**. Stays as string/object due to high variability.
    
    """
    #Check input type is pd.DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    

    print("\n--- Special Column Analysis ---")

   #Check if it contains data
    n_rows = len(df)
    if n_rows == 0:
        print("(DataFrame is empty)")
        return
   

    # Common patterns for IDs/keys (case-insensitive)
    id_regex = re.compile(r"(?:^|_)(id|uuid|key)(?:$|_)", flags=re.IGNORECASE)

    # Coordinate names
    coord_names = {"lat", "latitude", "lon", "long", "longitude"}


    for col in df.columns:
        name = str(col)
        s = df[col]

        # Category dtype
        if pd.api.types.is_categorical_dtype(s):
            print(f"<Categorical/Ordinal> {name}: **Categorical/Ordinal** (Type is 'category').")
            continue

        # Coordinates: based on column name
        if name.strip().lower() in coord_names:
            print(f"<Coordinates> {name}: Identified as **Latitude/Longitude**. Already optimized to a float dtype (if numeric optimization was run).")
            continue

        # Cardinality heuristics
        nunique = s.nunique(dropna=False)
        uniq_ratio = nunique / n_rows

        # Unique ID: name pattern + high cardinality
        if id_regex.search(name) and uniq_ratio >= 0.9:
            print(f"<Unique ID> {name}: Identified as potential **Unique ID**. High cardinality (not optimized to 'category').")
            continue

        # Text entity: object dtype + high cardinality (but not an ID)
        if pd.api.types.is_object_dtype(s) and (uniq_ratio > 0.5) and (not id_regex.search(name)):
            print(f"<Text Entity> {name}: Identified as **Text Entity (Name/Address)**. Stays as string/object due to high variability.")
            continue

    # No return value by design
    return None