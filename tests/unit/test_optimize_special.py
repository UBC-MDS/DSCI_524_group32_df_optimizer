import numpy as np
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
    # Triggers: object dtype + unique_ratio > 0.5 + not matching id_regex
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
