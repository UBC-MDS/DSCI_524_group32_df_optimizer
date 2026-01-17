import numpy as np
import pandas as pd
import pytest

from group_32.optimize_categorical import optimize_categorical
from group_32.optimize_numeric import optimize_numeric
from group_32.optimize_dataframe import optimize_dataframe
from group_32.optimize_special import optimize_special


def test_optimize_dataframe_wrapper_applies_numeric_and_categorical_and_runs_analysis(capsys):
    df = pd.DataFrame(
        {
            "region": ["US", "CA", "US", "US"],  # low-cardinality object -> category
            "quantity": np.array([1, 2, 3, 4], dtype=np.int64),  # -> smaller int
            "price": np.array([10.5, 12.0, 9.99, 11.25], dtype=np.float64),  # -> float32
        }
    )
    df_before = df.copy(deep=True)

    out = optimize_dataframe(df)
    captured = capsys.readouterr().out

    # wrapper should call the optimizers
    assert str(out["region"].dtype) == "category"
    assert str(out["price"].dtype) == "float32"
    assert str(out["quantity"].dtype) in {"int8", "int16", "int32"}

    # should not mutate original
    pd.testing.assert_frame_equal(df, df_before)

    # analysis should run (prints header)
    assert "Special Column Analysis" in captured