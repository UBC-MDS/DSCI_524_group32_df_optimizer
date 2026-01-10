# Welcome to Our DataFrame Memory Optimizer

This is a collaborative project developed as part of DSCI 524: Collaborate Softwware Development. 

## Project Summary

DataFrame Memory Optimizer is a lightweight Python package that reduces the memory footprint of pandas DataFrames by automatically selecting more efficient data types. It analyzes each column in a DataFrame and applies safe, transparent optimizations—such as numeric downcasting and categorical conversion—while preserving the original data values. This helps users work more efficiently with large datasets in data analysis, machine learning pipelines, and resource-constrained environments.

## Planned Functions 

`optimize_dataframe(df)`
The main user-facing function that takes a pandas DataFrame and returns an optimized copy with reduced memory usage. It coordinates numeric downcasting, categorical conversion, and reporting, while ensuring the original DataFrame is not modified.

`optimize_numeric(df)`
Optimizes numeric columns by downcasting integers to the smallest possible integer dtype and optionally converting floating-point columns to lower-precision floats when safe. This targets one of the largest sources of unnecessary memory usage in pandas.

`optimize_categorical(df, max_unique_ratio=0.5)`
Converts low-cardinality string/object columns to pandas’ category dtype based on a configurable uniqueness threshold. This can significantly reduce memory usage for repeated labels such as statuses, regions, or codes.

`analyze_special_columns(df)`
Identifies columns that may require special handling, such as ID columns, coordinates, or high-cardinality text fields. This function does not modify data but provides transparency by reporting which columns were optimized and which were intentionally left unchanged.

## How the package fits in the Python Ecosystem

Pandas provides low-level tools related to memory optimization, such as pd.to_numeric(..., downcast=...), DataFrame.convert_dtypes(), and DataFrame.memory_usage(deep=True). However, these tools require manual orchestration and do not provide a unified, opinionated workflow or clear diagnostics.

DataFrame Memory Optimizer builds on these ideas by combining them into a single, reusable interface that applies consistent heuristics and reports its decisions, making DataFrame memory optimization easier, safer, and more reproducible.

## Contributers

Mohammed Ibrahim

Roganci Fontelera

Wai Yan Lee

William Chong

## Copyright

- Copyright © 2026 UBC MDS.
- Free software distributed under the [MIT License](./LICENSE).
