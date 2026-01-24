import pandas as pd
import pytest
from group_32.optimize_special import optimize_special

def test_optimize_special_raises_type_error_for_non_dataframe():
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        optimize_special([1, 2, 3])


def test_optimize_special_empty_dataframe_prints_empty_message(capsys):
    df = pd.DataFrame()
    result = optimize_special(df)
    captured = capsys.readouterr().out
    assert result is None
    assert "Special Column Analysis" in captured
    assert "(DataFrame is empty)" in captured


def test_optimize_special_category_branch(capsys):
    # Triggers: if pd.api.types.is_categorical_dtype(s)
    df = pd.DataFrame(
        {
            "membership_level": pd.Categorical(["gold", "silver", "gold"])
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    assert "Special Column Analysis" in captured
    assert "membership_level: Identified as categorical or ordinal data (category dtype)." in captured


def test_optimize_special_coordinate_branch(capsys):
    # Triggers: if name.strip().lower() in coord_names
    df = pd.DataFrame(
        {
            "lat": [49.28, 49.29, 49.30],
            "longitude": [-123.12, -123.11, -123.10],
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    assert "lat: Identified as geographic coordinate column." in captured
    assert "longitude: Identified as geographic coordinate column." in captured


def test_optimize_special_unique_id_branch_high_cardinality(capsys):
    # Triggers: if id_regex.search(name) and unique_ratio >= 0.9
    df = pd.DataFrame(
        {
            "customer_id": range(10)  # 10 unique / 10 rows = 1.0 >= 0.9
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    assert "customer_id: Identified as potential Unique ID (high cardinality)." in captured


def test_optimize_special_text_entity_branch_high_cardinality_object(capsys):
    # Triggers: object dtype (object or string) + unique_ratio > 0.5 + not matching id_regex
    df = pd.DataFrame(
        {
            "full_name": [f"Person {i}" for i in range(10)]  # 10 unique / 10 rows = 1.0 > 0.5
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    assert "full_name: Identified as high-cardinality text column." in captured


def test_optimize_special_does_not_flag_low_cardinality_object(capsys):
    # Ensures the "text entity" branch does NOT trigger when unique_ratio <= 0.5
    df = pd.DataFrame(
        {
            "city": ["NYC", "LA", "NYC", "LA"]  # 2 unique / 4 rows = 0.5 (not > 0.5)
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    # It will print the header, but should not print a "city:" classification line
    assert "Special Column Analysis" in captured
    assert "city: Identified as high-cardinality text column." not in captured


def test_optimize_special_id_named_but_not_high_cardinality_not_flagged(capsys):
    # Ensures the "unique ID" branch does NOT trigger if ratio < 0.9
    df = pd.DataFrame(
        {
            "order_id": ["A", "A", "B", "B", "B"]  # 2 unique / 5 = 0.4 < 0.9
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    assert "Special Column Analysis" in captured
    assert "order_id: Identified as potential Unique ID (high cardinality)." not in captured


def test_optimize_special_dataframe_not_modified():
    # Since optimize_special is analysis only, confirm no mutation.
    df = pd.DataFrame(
        {
            "customer_id": range(5),
            "lat": [1.0] * 5,
            "name": ["a", "b", "c", "d", "e"],
            "membership_level": pd.Categorical(["x", "y", "x", "y", "x"]),
        }
    )
    df_before = df.copy(deep=True)
    optimize_special(df)
    pd.testing.assert_frame_equal(df, df_before)


#All-null column skip logic
def test_optimize_special_skips_all_null_columns(capsys):
    """Test that columns with all NaN/None values are silently skipped."""
    df = pd.DataFrame(
        {
            "all_null_col": [None, None, None, None],
            "valid_col": ["A", "B", "C", "D"]  # Low cardinality, won't be flagged
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    
    # Should print header but not mention all_null_col
    assert "Special Column Analysis" in captured
    assert "all_null_col" not in captured


# Coordinate column with whitespace in name
def test_optimize_special_coordinate_with_whitespace(capsys):
    """Test that coordinate column names with leading/trailing whitespace are recognized."""
    df = pd.DataFrame(
        {
            "  latitude  ": [37.7749, 37.7750, 37.7751],
            " lon ": [-122.4194, -122.4195, -122.4196]
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    
    assert "  latitude  : Identified as geographic coordinate column." in captured
    assert " lon : Identified as geographic coordinate column." in captured


#Unique ID threshold boundary - exactly at 0.9
def test_optimize_special_unique_id_boundary_at_threshold(capsys):
    """Test unique ID detection when ratio is exactly 0.9 (should trigger)."""
    df = pd.DataFrame(
        {
            "user_id": list(range(9)) + [8]  # 9 unique / 10 rows = 0.9
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    
    assert "user_id: Identified as potential Unique ID (high cardinality)." in captured


#Unique ID threshold boundary - just below 0.9
def test_optimize_special_unique_id_boundary_below_threshold(capsys):
    """Test unique ID detection when ratio is below 0.9 (should NOT trigger)."""
    df = pd.DataFrame(
        {
            "product_id": list(range(8)) + [7, 7]  # 8 unique / 10 rows = 0.8 < 0.9
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    
    # Should NOT be flagged as unique ID
    assert "product_id: Identified as potential Unique ID (high cardinality)." not in captured


#Text entity threshold boundary - exactly at 0.5
def test_optimize_special_text_entity_boundary_at_threshold(capsys):
    """Test text entity detection when ratio is exactly 0.5 (should NOT trigger since check is > 0.5)."""
    df = pd.DataFrame(
        {
            "description": ["A", "B", "C", "D", "E"] + ["A", "B", "C", "D", "E"]  # 5 unique / 10 rows = 0.5
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    
    # Should NOT be flagged as text entity (needs > 0.5, not >= 0.5)
    assert "description: Identified as high-cardinality text column." not in captured


#Text entity threshold boundary - just above 0.5
def test_optimize_special_text_entity_boundary_above_threshold(capsys):
    """Test text entity detection when ratio is just above 0.5 (should trigger)."""
    df = pd.DataFrame(
        {
            "comment": [f"Comment {i}" for i in range(6)] + ["Comment 0"] * 4  # 6 unique / 10 rows = 0.6 > 0.5
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    
    assert "comment: Identified as high-cardinality text column." in captured


#Numeric columns with high cardinality should NOT be flagged as text entities
def test_optimize_special_numeric_high_cardinality_not_flagged_as_text(capsys):
    """Test that numeric columns with high cardinality are not flagged as text entities."""
    df = pd.DataFrame(
        {
            "price": [100.0 + i for i in range(10)],  # 10 unique / 10 rows = 1.0, but numeric
            "quantity": range(10)  # Also numeric with high cardinality
        }
    )
    optimize_special(df)
    captured = capsys.readouterr().out
    
    # Should print header but NOT flag numeric columns as text entities
    assert "Special Column Analysis" in captured
    assert "price: Identified as high-cardinality text column." not in captured
    assert "quantity: Identified as high-cardinality text column." not in captured

def test_optimize_special_mixed_case_coordinate_names(capsys):
    """Test that coordinate columns with mixed case names are recognized."""
    df = pd.DataFrame({
        "Latitude": [49.28, 49.29, 49.30],
        "LONGITUDE": [-123.12, -123.11, -123.10],
        "LoN": [-122.41, -122.42, -122.43]
    })
    
    optimize_special(df)
    captured = capsys.readouterr().out