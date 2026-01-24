from group_32.optimize_categorical import optimize_categorical

import pandas as pd
import pytest
import re

def test_optimize_categorical_converts_columns(capsys):

    df = pd.DataFrame(
        {
            "user_id": list(range(10)),
            "name": ["Alice", "Bobby", "Charles", "Ally", "Bob", "Charlie", "David", "Alex", "Ben", "Cherry"],
            "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "LA", "Chicago", "NYC", "LA"],
            "status": ["active", "inactive", "active", "active", "inactive", "active", "inactive", "active", "active", "inactive"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "binary_flag": [True, False, True, False, True, False, True, False, True, False]
        }
    )

    output = optimize_categorical(df, max_unique_ratio=0.5)

    # Test function converts appropriate columns
    assert str(output["city"].dtype) == "category"
    assert str(output["status"].dtype) == "category"

    # Test that other columns remain unchanged
    assert not pd.api.types.is_categorical_dtype(output["name"])
    assert df["user_id"].dtype == output["user_id"].dtype
    assert df["score"].dtype == output["score"].dtype
    assert df["binary_flag"].dtype == output["binary_flag"].dtype

    #Test print
    captured = capsys.readouterr()
    assert "Converted 2 column(s) to 'category' dtype." in captured.out

def test_optimize_categorical_threshold():
    df = pd.DataFrame(
        {
            "id": ["A", "B", "A", "C", "B", "D", "E", "F", "G", "H"],  # 8 unique / 10 = 0.8
            "hours": [23, 40, 12, 77, 85, 12, 64, 64, 46, 37.5],
            "company": ["Comp_A", "Comp_R", "Comp_A", "Comp_D", "Comp_G", "Comp_D", "Comp_A", "Comp_G", "Comp_R", "Comp_A"], #4 unique / 10 = 0.4
            "brand": ['A'] * 10 #1 unique / 10 = 0.1
        }
    )

    df_before = df.copy()

    output2 = optimize_categorical(df, max_unique_ratio=1)
    
    # Test threshold = 1 (convert all string columns)
    assert str(output2["id"].dtype) == "category"
    assert str(output2["company"].dtype) == "category"
    assert str(output2["brand"].dtype) == "category"

    # Test threshold = 0 (convert none)
    output3 = optimize_categorical(df, max_unique_ratio=0)
    assert output3.equals(df_before)

    # Test with None values
    df["brand"] = None

    #Test different thresholds
    output_low = optimize_categorical(df, max_unique_ratio=0.5)
    assert not pd.api.types.is_categorical_dtype(output_low["id"])

    output_high = optimize_categorical(df, max_unique_ratio=0.9)
    assert str(output_high["id"].dtype) == "category"

    output_low2 = optimize_categorical(df, max_unique_ratio=0.2)
    assert not pd.api.types.is_categorical_dtype(output_low2["id"])

    output1 = optimize_categorical(df, max_unique_ratio=0.1)
    assert not pd.api.types.is_categorical_dtype(output1["brand"])

    #threshold > 1 -> error
    with pytest.raises(TypeError, match = re.escape("max_unique_ratio must be between 0 and 1 (inclusive)!")):
        optimize_categorical(df, max_unique_ratio=2)

    #threshold < 0 -> error
    with pytest.raises(TypeError, match = re.escape("max_unique_ratio must be between 0 and 1 (inclusive)!")):
        optimize_categorical(df, max_unique_ratio=-0.5)

def test_optimize_categorical_invalid_inputs():
    df = pd.DataFrame(
        {"city": ["NYC", "LA", "NYC", "LA"]}
        )

    # Test invalid threshold values
    with pytest.raises(TypeError, match= re.escape("max_unique_ratio must be between 0 and 1 (inclusive)!")):
        optimize_categorical(df, max_unique_ratio=2)
    
    with pytest.raises(TypeError, match= re.escape("max_unique_ratio must be between 0 and 1 (inclusive)!")):
        optimize_categorical(df, max_unique_ratio=-0.5)
    
    # Test invalid type for threshold
    with pytest.raises(TypeError, match="max_unique_ratio must be a number"):
        optimize_categorical(df, max_unique_ratio="30%")
    
    # Test invalid df input
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        optimize_categorical(["A", "B", "C"], max_unique_ratio=0.8)

def test_optimize_categorical_not_df():
    df = ["A", "B", "C", "D", "E"]

    with pytest.raises(TypeError, match = "df must be a pandas DataFrame"):
        optimize_categorical(df, max_unique_ratio=0.8)

def test_optimize_categrical_empty_and_special_cases():
    
    # Test empty DataFrame
    df_empty = pd.DataFrame({"col1": [], "col2": []})

    output_empty = optimize_categorical(df_empty, max_unique_ratio=0.4)
    assert len(output_empty) == 0
    assert df_empty.columns.tolist() == output_empty.columns.tolist()
    
    # Test DataFrame with only non-string columns
    df_numeric = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.1, 2.2, 3.3],
        "bool_col": [True, False, True]
    })
    output_numeric = optimize_categorical(df_numeric, max_unique_ratio=0.5)
    assert df_numeric.equals(output_numeric)  # No changes expected
    
    # Test DataFrame with NaN values in string columns
    df_nan = pd.DataFrame({
        "category": ["A", "B", "A", None, "B", "A", None, "B", "A", "B"],
        "text": ["text"] * 10
    })
    output_nan = optimize_categorical(df_nan, max_unique_ratio=0.5)
    # Should handle NaN appropriately

    assert str(output_nan["category"].dtype) == "category"

def test_optimize_categorical_single_value_column(capsys):
    """Test that columns with only one unique value are converted to category."""
    df = pd.DataFrame({
        "all_same": ["A"] * 100,  # 1 unique / 100 rows = 0.01
        "also_same": ["XYZ"] * 100,  # 1 unique / 100 rows = 0.01
        "varied": [f"item_{i}" for i in range(100)]  # 100 unique / 100 rows = 1.0
    })
    
    output = optimize_categorical(df, max_unique_ratio=0.5)
    
    # Single-value columns should be converted (ratio = 0.01 << 0.5)
    assert str(output["all_same"].dtype) == "category"
    assert str(output["also_same"].dtype) == "category"
    
    # High-cardinality column should NOT be converted (ratio = 1.0 > 0.5)
    assert not pd.api.types.is_categorical_dtype(output["varied"])
    
    # Verify values are preserved
    assert output["all_same"].unique().tolist() == ["A"]
    assert output["also_same"].unique().tolist() == ["XYZ"]
    
    # Should report 2 columns converted
    captured = capsys.readouterr()
    assert "Converted 2 column(s) to 'category' dtype." in captured.out