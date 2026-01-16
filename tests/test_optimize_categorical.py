
from group_32.optimize_categorical import optimize_categorical
import pandas as pd
import pytest
import re


@pytest.fixture

def sample_data() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "user_id": list(range(10)),
            "name": ["Alice", "Bobby", "Charles", "Ally", "Bob", "Charlie", "David", "Alex", "Ben", "Cherry"],
            "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "LA", "Chicago", "NYC", "LA"],
            "status": ["active", "inactive", "active", "active", "inactive", "active", "inactive", "active", "active", "inactive"],
        }
    )

    return pd.DataFrame(df)

def test_optimize_categorical_converts_low_cardinality_object_columns(sample_data):

    output = optimize_categorical(sample_data, max_unique_ratio=0.5)

    # city: 3 unique / 10 = 0.3 -> convert
    assert str(output["city"].dtype) == "category"

    # status: 2 unique / 10 = 0.2 -> convert
    assert str(output["status"].dtype) == "category"

    # name: 10 unique / 10 = 1.0 -> do NOT convert
    assert output["name"].dtype == object

    # non-object stays unchanged
    assert sample_data["user_id"].dtype == output["user_id"].dtype


def test_optimize_categorical_threshold():
    df = pd.DataFrame(
        {
            "id": ["A", "B", "A", "C", "B", "D", "E", "F", "G", "H"],  # 8 unique / 10 = 0.8
            "hours": [23, 40, 12, 77, 85, 12, 64, 64, 46, 37.5],
            "company": ["Comp_A", "Comp_R", "Comp_A", "Comp_D", "Comp_G", "Comp_D", "Comp_A", "Comp_G", "Comp_R", "Comp_A"], #4 unique / 10 = 0.4
            "brand": ['A'] * 10
        }
    )

    df_before = df.copy(deep=True)
    df["brand"] = None

    output_low = optimize_categorical(df, max_unique_ratio=0.5)
    assert output_low["id"].dtype == object

    output_high = optimize_categorical(df, max_unique_ratio=0.9)
    assert str(output_high["id"].dtype) == "category"

    output_low2 = optimize_categorical(df, max_unique_ratio=0.2)
    assert output_low2["id"].dtype == object

    output = optimize_categorical(df, max_unique_ratio=0.1)
    assert output["brand"].dtype == object

    with pytest.raises(TypeError, match = re.escape("max_unique_ratio must be between 0 and 1 (inclusive)!")):
        optimize_categorical(df, max_unique_ratio=2)

    output = optimize_categorical(df, max_unique_ratio=0)
    pd.DataFrame.equals(output, df_before)

    with pytest.raises(TypeError, match = re.escape("max_unique_ratio must be between 0 and 1 (inclusive)!")):
        optimize_categorical(df, max_unique_ratio=-0.5)

def test_optimize_categorical_no_change():
    df = pd.DataFrame(
        {"city": ["NYC", "LA", "NYC", "LA"]}
        )
    df_before = df.copy(deep=True)

    out = optimize_categorical(df, max_unique_ratio=0.8)

    assert str(out["city"].dtype) == "category"

    assert df["city"].dtype == object

    pd.DataFrame.equals(df, df_before)

    with pytest.raises(TypeError, match = "max_unique_ratio must be a number"):
        optimize_categorical(df, max_unique_ratio= re.escape("30%"))

def test_optimize_categorical_not_df():
    df = ["A", "B", "C", "D", "E"]

    with pytest.raises(TypeError, match = "df must be a pandas DataFrame"):
        optimize_categorical(df, max_unique_ratio=0.8)

    