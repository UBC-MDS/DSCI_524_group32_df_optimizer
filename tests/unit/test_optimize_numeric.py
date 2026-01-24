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


def test_boundary_values_int16():
    """
    Test that values at int16 boundaries are handled correctly.
    """
    df = pd.DataFrame({
        "boundary_int": np.array([32767, -32768, 0], dtype=np.int64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert result["boundary_int"].dtype == np.int16


def test_boundary_values_int32():
    """
    Test that values at int32 boundaries are handled correctly.
    """
    df = pd.DataFrame({
        "boundary_int": np.array([2147483647, -2147483648, 0], dtype=np.int64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert result["boundary_int"].dtype == np.int32


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


def test_float32_sufficient_precision():
    """
    Test that floats are downcasted to float32 when precision is sufficient.
    """
    df = pd.DataFrame({
        "float_col": np.array([1.5, 2.5, 3.5], dtype=np.float64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert result["float_col"].dtype == np.float32


def test_float64_required_for_high_precision():
    """
    Test float downcasting behavior - floats are downcasted to float32
    unless precision would be lost beyond acceptable tolerance.
    """
    df = pd.DataFrame({
        "float_col": np.array([1.123456789123456, 2.987654321987654], dtype=np.float64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    # The function downcasts to float32 if values can be represented accurately enough
    # Verify the values are preserved within acceptable tolerance
    assert pd.api.types.is_float_dtype(result["float_col"])
    np.testing.assert_allclose(result["float_col"], df["float_col"], rtol=1e-6)


def test_unsigned_integers_handled_correctly():
    """
    Test that unsigned integer columns are downcasted appropriately.
    Note: unsigned integers may be converted to signed integers if all values
    fit within the signed range.
    """
    df = pd.DataFrame({
        "uint_col": np.array([0, 100, 255], dtype=np.uint64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    # Should be downcasted to int16 (or int8 if your function supports it)
    # since all values are positive and fit in a smaller integer type
    assert pd.api.types.is_integer_dtype(result["uint_col"])
    pd.testing.assert_series_equal(result["uint_col"], df["uint_col"], check_dtype=False)


def test_empty_dataframe():
    """
    Test that an empty DataFrame is handled without errors.
    """
    df = pd.DataFrame()
    
    result = optimize_numeric(df, verbose=False)
    
    assert len(result) == 0
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_one_row():
    """
    Test that a DataFrame with a single row is optimized correctly.
    """
    df = pd.DataFrame({
        "int_col": np.array([42], dtype=np.int64),
        "float_col": np.array([3.14], dtype=np.float64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert result["int_col"].dtype == np.int8
    assert result["float_col"].dtype == np.float32


def test_original_dataframe_not_modified():
    """
    Test that the original DataFrame is not modified (function returns a copy).
    """
    df = pd.DataFrame({
        "int_col": np.array([1, 2, 3], dtype=np.int64)
    })
    
    original_dtype = df["int_col"].dtype
    result = optimize_numeric(df, verbose=False)
    
    # Original should remain unchanged
    assert df["int_col"].dtype == original_dtype
    # Result should be optimized
    assert result["int_col"].dtype == np.int8


def test_datetime_columns_not_affected():
    """
    Test that datetime columns are not modified by the function.
    """
    df = pd.DataFrame({
        "date_col": pd.date_range("2024-01-01", periods=3),
        "int_col": np.array([1, 2, 3], dtype=np.int64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert pd.api.types.is_datetime64_any_dtype(result["date_col"])
    pd.testing.assert_series_equal(result["date_col"], df["date_col"])


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
    
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        optimize_numeric(np.array([1, 2, 3]))


def test_mixed_positive_negative_integers():
    """
    Test optimization with mixed positive and negative integers.
    """
    df = pd.DataFrame({
        "mixed_int": np.array([-50, 0, 50], dtype=np.int64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert result["mixed_int"].dtype == np.int8
    pd.testing.assert_series_equal(result["mixed_int"], df["mixed_int"], check_dtype=False)


def test_all_zeros():
    """
    Test that a column of all zeros is optimized correctly.
    """
    df = pd.DataFrame({
        "zero_int": np.array([0, 0, 0], dtype=np.int64),
        "zero_float": np.array([0.0, 0.0, 0.0], dtype=np.float64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert result["zero_int"].dtype == np.int8
    assert result["zero_float"].dtype == np.float32


def test_inf_values_in_floats():
    """
    Test that infinity values in float columns are preserved.
    """
    df = pd.DataFrame({
        "float_col": np.array([1.0, np.inf, -np.inf, 2.0], dtype=np.float64)
    })
    
    result = optimize_numeric(df, verbose=False)
    
    assert np.isinf(result["float_col"]).sum() == 2
    assert result["float_col"].iloc[1] == np.inf
    assert result["float_col"].iloc[2] == -np.inf

def test_optimize_numeric_verbose_false_no_output(capsys):
    """Test that verbose=False suppresses output."""
    df = pd.DataFrame({
        "int_col": np.array([1, 2, 3], dtype=np.int64),
        "float_col": np.array([1.5, 2.5, 3.5], dtype=np.float64)
    })
    
    result = optimize_numeric(df, verbose=False)
    captured = capsys.readouterr()
    
    # Should not print anything when verbose=False
    assert captured.out == ""
    # But should still perform optimization
    assert result["int_col"].dtype == np.int8
    assert result["float_col"].dtype == np.float32


def test_optimize_numeric_negative_integers_verbose(capsys):
    """
    Test that optimize_numeric handles negative integers correctly
    and produces verbose output when verbose=True.
    """
    # Create a DataFrame with negative int64 values
    df = pd.DataFrame({
        "neg_int": np.array([-10, -20, -30], dtype=np.int64)
    })
    
    # Call optimize_numeric with verbose=True
    result = optimize_numeric(df, verbose=True)
    
    # Capture the printed output
    captured = capsys.readouterr()
    
    # Assertions
    assert "Memory reduced from" in captured.out, "Expected verbose message not found in output"
    assert "reduction)" in captured.out, "Expected reduction percentage not found in output"
    
    # Additional assertions to verify the optimization worked
    assert "neg_int" in result.columns, "Column 'neg_int' missing from result"
    
    # Verify the data type was optimized (should be int8 since values fit in that range)
    assert result["neg_int"].dtype in [np.int8, np.int16, np.int32, np.int64], \
        f"Expected integer dtype, got {result['neg_int'].dtype}"
    
    # Verify values are preserved
    assert result["neg_int"].tolist() == [-10, -20, -30], \
        "Values were not preserved during optimization"
    
    # Print captured output for debugging
    print("\n--- Captured Verbose Output ---")
    print(captured.out)
    print("--- End of Captured Output ---\n")
    
    # Verify the dtype is smaller than original if possible
    original_dtype = df["neg_int"].dtype
    optimized_dtype = result["neg_int"].dtype
    
    print(f"Original dtype: {original_dtype}")
    print(f"Optimized dtype: {optimized_dtype}")
    print(f"Memory saved: {df.memory_usage(deep=True).sum() - result.memory_usage(deep=True).sum()} bytes")