import pandas as pd
import numpy as np


def optimize_categorical(df: pd.DataFrame, max_unique_ratio: float = 0.5) -> pd.DataFrame:
    """
    This function makes a copy of a DataFrame (df) and looks through all the string columns. 
    It counts the number of unique strings and calculates the ratio of the unique values to the total number of rows. 
    If this ratio is below a chosen threshold in this function, it converts the column to category dtype.
    This function returns a dataframe with the converted columns and prints the number of columns that have been updated.
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
    >>> Converted 2 column(s) to 'category' dtype.
    
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
    #check if df input is a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    #check if threshold input is a int or float or invalid
    if not isinstance(max_unique_ratio, (int, float)) or np.isnan(max_unique_ratio):
        raise TypeError("max_unique_ratio must be a number")

    #check is threshold is negtive or larger than 1
    if max_unique_ratio < 0 or max_unique_ratio > 1:
        raise TypeError("max_unique_ratio must be between 0 and 1 (inclusive)!")

    df_copy = df.copy()

    n_rows = len(df_copy)
    if n_rows == 0:

        return df_copy

    converted_columns = 0

    for col in df_copy.select_dtypes(include=["object"]).columns:
        n_col = df_copy[col]

        if n_col.isnull().all(): #if the column is empty, terminate the current loop
            break

        n_unique = n_col.nunique(dropna=False)
        ratio = n_unique / n_rows
        
        if ratio <= max_unique_ratio:
            df_copy[col] = n_col.astype("category")
            converted_columns  += 1

    if converted_columns  > 0:
        print(f"Converted {converted_columns} column(s) to 'category' dtype.")

    return df_copy

    