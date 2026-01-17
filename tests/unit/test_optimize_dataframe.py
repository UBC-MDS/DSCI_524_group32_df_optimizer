import numpy as np
import pandas as pd
import pytest

from group_32.optimize_dataframe import optimize_dataframe


@pytest.fixture
def df_mixed():
    """A small mixed-type DataFrame that should trigger numeric + categorical + analysis."""
    return pd.DataFrame(
        {
            "region": ["US", "CA", "US", "US"],  # low-cardinality object -> category
            "quantity": np.array([1, 2, 3, 4], dtype=np.int64),  # -> smaller int
            "price": np.array([10.5, 12.0, 9.99, 11.25], dtype=np.float64),  # -> float32
        }
    )


@pytest.fixture
def df_empty():
    """Empty DataFrame should not crash."""
    return pd.DataFrame()


def test_optimize_dataframe_wrapper_applies_numeric_and_categorical_and_runs_analysis(df_mixed, capsys):
    df = df_mixed
    df_before = df.copy(deep=True)

    out = optimize_dataframe(df)
    captured = capsys.readouterr().out

    assert str(out["region"].dtype) == "category", "Expected 'region' to be converted to category."
    assert str(out["price"].dtype) == "float32", "Expected 'price' to be downcast to float32."
    assert str(out["quantity"].dtype) in {"int8", "int16", "int32"}, "Expected 'quantity' to be downcast to a smaller int dtype."

    # should not mutate original
    pd.testing.assert_frame_equal(df, df_before)

    # analysis should run (prints header)
    assert "Special Column Analysis" in captured, "Expected optimize_special to print analysis header."


def test_optimize_dataframe_raises_type_error_for_non_dataframe():
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        optimize_dataframe([1, 2, 3])


def test_optimize_dataframe_empty_dataframe_returns_empty(df_empty, capsys):
    out = optimize_dataframe(df_empty)
    captured = capsys.readouterr().out

    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_frame_equal(out, df_empty)
    assert "Special Column Analysis" in captured, "Expected analysis to run even for empty DataFrame."


def test_optimize_dataframe_no_object_columns_still_downcasts_numeric(capsys):
    df = pd.DataFrame(
        {
            "a": np.array([1, 2, 3, 4], dtype=np.int64),
            "b": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        }
    )
    df_before = df.copy(deep=True)

    out = optimize_dataframe(df)
    _ = capsys.readouterr().out

    assert str(out["b"].dtype) == "float32", "Expected float64 column to downcast to float32."
    assert str(out["a"].dtype) in {"int8", "int16", "int32"}, "Expected int64 column to downcast to a smaller int dtype."
    pd.testing.assert_frame_equal(df, df_before)


def test_optimize_dataframe_no_numeric_columns_still_converts_categorical(capsys):
    df = pd.DataFrame(
        {
            "region": ["US", "CA", "US", "US"],
            "status": ["pending", "shipped", "pending", "pending"],
        }
    )
    df_before = df.copy(deep=True)

    out = optimize_dataframe(df)
    _ = capsys.readouterr().out

    assert str(out["region"].dtype) == "category", "Expected 'region' to become category."
    assert str(out["status"].dtype) == "category", "Expected 'status' to become category."
    pd.testing.assert_frame_equal(df, df_before)


def test_optimize_dataframe_preserves_values(capsys):
    df = pd.DataFrame(
        {
            "quantity": np.array([1, 2, 3, 4], dtype=np.int64),
            "price": np.array([10.5, 12.0, 9.99, 11.25], dtype=np.float64),
            "region": ["US", "CA", "US", "US"],
        }
    )

    out = optimize_dataframe(df)
    _ = capsys.readouterr().out

    # Values should match original
    pd.testing.assert_series_equal(out["quantity"].astype("int64"), df["quantity"], check_names=True)
    np.testing.assert_allclose(out["price"].to_numpy(), df["price"].to_numpy(), rtol=1e-6, atol=1e-8)
    pd.testing.assert_series_equal(out["region"].astype("object"), df["region"], check_names=True)


def test_optimize_dataframe_calls_helpers(monkeypatch, df_mixed, capsys):
    """
    Branch-style test for wrapper logic: prove wrapper calls each helper exactly once.
    This is optional, but it's a strong Milestone 2 style test for a wrapper.
    """
    calls = {"num": 0, "cat": 0, "spec": 0}

    def fake_numeric(dfin):
        calls["num"] += 1
        return dfin

    def fake_categorical(dfin, max_unique_ratio=0.5):
        calls["cat"] += 1
        return dfin

    def fake_special(dfin):
        calls["spec"] += 1
        print("\n--- Special Column Analysis ---")  # keep wrapper test consistent with capsys expectations

    # Patch the names *as used inside optimize_dataframe module*
    monkeypatch.setattr("group_32.optimize_dataframe.optimize_numeric", fake_numeric)
    monkeypatch.setattr("group_32.optimize_dataframe.optimize_categorical", fake_categorical)
    monkeypatch.setattr("group_32.optimize_dataframe.optimize_special", fake_special)

    optimize_dataframe(df_mixed)
    captured = capsys.readouterr().out

    assert calls["num"] == 1, f"Expected optimize_numeric to be called once, got {calls['num']}."
    assert calls["cat"] == 1, f"Expected optimize_categorical to be called once, got {calls['cat']}."
    assert calls["spec"] == 1, f"Expected optimize_special to be called once, got {calls['spec']}."
    assert "Special Column Analysis" in captured
