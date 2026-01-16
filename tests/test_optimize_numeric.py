import pandas as pd
import numpy as np
from group_32.optimize_numeric import optimize_numeric
import pytest


def test_integer_columns_are_downcasted_only():
    """
    Ensure that integer columns are downcasted, and that the function
    does not modify non-numeric columns in any way.
    """
    df = pd.DataFrame({
        "int_col": np.array([1, 2, 3], dtype=np.int64),
        "cat_col": ["a", "b", "c"]
    })

    result = optimize_numeric(df)

    # Integer column should remain integer
    assert pd.api.types.is_integer_dtype(result["int_col"])
    pd.testing.assert_series_equal(result["int_col"], df["int_col"], check_dtype=False)

    # Non-numeric column should be unchanged
    pd.testing.assert_series_equal(result["cat_col"], df["cat_col"])


def test_float_columns_are_downcasted_only():
    """
    Verify that float columns are optimized while non-numeric
    columns are not affected.
    """
    df = pd.DataFrame({
        "float_col": np.array([1.5, 2.5, 3.5], dtype=np.float64),
        "cat_col": ["x", "y", "z"]
    })

    result = optimize_numeric(df)

    assert pd.api.types.is_float_dtype(result["float_col"])
    np.testing.assert_allclose(result["float_col"], df["float_col"], rtol=1e-6)
    pd.testing.assert_series_equal(result["cat_col"], df["cat_col"])


def test_mixed_numeric_columns_are_handled_independently():
    """
    Confirm that integer and float columns are both optimized
    independently within the same DataFrame.
    """
    df = pd.DataFrame({
        "int_col": np.array([10, 20, 30], dtype=np.int64),
        "float_col": np.array([1.1, 2.2, 3.3], dtype=np.float64)
    })

    result = optimize_numeric(df)

    assert pd.api.types.is_integer_dtype(result["int_col"])
    assert pd.api.types.is_float_dtype(result["float_col"])
    pd.testing.assert_series_equal(result["int_col"], df["int_col"], check_dtype=False)
    np.testing.assert_allclose(result["float_col"], df["float_col"], rtol=1e-6)


def test_missing_values_preserved_for_numeric_columns():
    """
    Ensure that missing values in numeric columns are preserved
    after optimization.
    """
    df = pd.DataFrame({
        "int_col": [1, None, 3],
        "float_col": [1.0, np.nan, 3.0],
        "cat_col": ["a", "b", "c"]
    })

    result = optimize_numeric(df)

    assert result["int_col"].isna().sum() == df["int_col"].isna().sum()
    assert result["float_col"].isna().sum() == df["float_col"].isna().sum()
    pd.testing.assert_series_equal(result["cat_col"], df["cat_col"])


def test_no_numeric_columns_does_not_alter_dataframe():
    """
    Verify that the function does not alter the DataFrame
    when no numeric columns are present.
    """
    df = pd.DataFrame({
        "cat1": ["a", "b", "c"],
        "cat2": ["x", "y", "z"]
    })

    result = optimize_numeric(df)

    pd.testing.assert_frame_equal(result, df)


def test_negative_integers_downcasted_correctly():
    """
    Test that negative integers are downcasted to appropriate signed types.
    """
    df = pd.DataFrame({
        "neg_int": np.array([-10, -20, -30], dtype=np.int64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    # Should be int8 (range -128 to 127)
    assert result["neg_int"].dtype == np.int8


def test_boundary_values_int8():
    """
    Test that values at int8 boundaries (127, -128) are handled correctly.
    """
    df = pd.DataFrame({
        "boundary_int": np.array([127, -128, 0], dtype=np.int64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert result["boundary_int"].dtype == np.int8



def test_boolean_columns_not_affected():
    """
    Test that boolean columns are not modified by the function.
    """
    df = pd.DataFrame({
        "bool_col": [True, False, True],
        "int_col": np.array([1, 2, 3], dtype=np.int64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert result["bool_col"].dtype == bool
    pd.testing.assert_series_equal(result["bool_col"], df["bool_col"])



def test_very_large_integers_remain_int64():
    """
    Test that integers too large for int32 remain as int64.
    """
    df = pd.DataFrame({
        "huge_int": np.array([2147483648, 2147483649], dtype=np.int64)  # Beyond int32 max
    })
    
    result = optimize_numeric(df, verbose=False)
    
    # Should remain int64 since values exceed int32 range
    assert result["huge_int"].dtype == np.int64

def test_invalid_input_type():
    """
    Test that the function raises TypeError for non-DataFrame input.
    """
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        optimize_numeric([1, 2, 3])
    
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        optimize_numeric("not a dataframe")
    
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        optimize_numeric(None)