def optimize_categorical(df: pd.DataFrame, max_unique_ratio: float = 0.5) -> pd.DataFrame:
    """
    This function makes a copy of a DataFrame (df) and looks through all the string columns. 
    It counts the number of unique strings and calculates the ratio of the unique values to the total number of rows. 
    If this ratio is below a chosen threshold in this function, it converts the column to category dtype.
    This function returns a dataframe with the converted columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing string columns.

    max_unique_ratio : float, default=0.5
        The maximum ratio of (unique_values / total_rows) for categorical conversion.
        - 0.5 means convert if unique values are less than 50% of total rows
        - Lower values (e.g., 0.3) make conversion more conservative
        - Higher values (e.g., 0.7) make conversion more aggressive

    Returns
    -------
    pd.DataFrame
        The DataFrame with eligible string columns converted to category dtype.

    Example
    -------
    >>> import pandas as pd
    >>> data = {
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['Alice', 'Bobby', 'Charles', 'Ally', 'Bob', 'Charlie', 'David', 'Alex', 'Ben', 'Cherry'],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC', 'LA', 'Chicago', 'NYC', 'LA'],
        'status': ['active', 'inactive', 'active', 'active', 'inactive', 'active', 'inactive', 'active', 'active', 'inactive']}
    >>> df = pd.DataFrame(data)
    >>> df.dtypes
    user_id  int64
    name     object
    city     object
    status   object
    dtype: object
    
    >>> optimized_df = optimize_categorical(df, max_unique_ratio=0.5)
    >>> optimized_df.dtypes
    user_id  int64
    name     object
    city     category
    status   category
    dtype: object
    
    >>> optimized_df.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype   
    ---  ------  --------------  -----   
     0   user_id 10 non-null    int64
     1   name    10 non-null    object
     2   city    10 non-null    category
     3   status  10 non-null    category   
    dtypes: category(2), int64(1), object(1)

    Notes
    -----
    - Only processes columns with 'object' dtype (typically strings)
    - Prints the number of columns successfully converted
    - Conversion is reversible using .astype('object') if needed
    - Categorical columns maintain the same values and ordering
    """