# Welcome to Our DataFrame Memory Optimizer
|        |        |
|--------|--------|
| Testing | [![ci](https://github.com/UBC-MDS/DSCI_524_group32_df_optimizer/actions/workflows/build.yml/badge.svg)](https://github.com/UBC-MDS/DSCI_524_group32_df_optimizer/actions/workflows/build.yml) [![cd](https://github.com/UBC-MDS/DSCI_524_group32_df_optimizer/actions/workflows/deploy.yml/badge.svg)](https://github.com/UBC-MDS/DSCI_524_group32_df_optimizer/actions/workflows/deploy.yml) |
| Package | [![TestPyPI](https://img.shields.io/badge/dynamic/json?url=https://test.pypi.org/pypi/DSCI-524-group32-df-optimizer/json&query=%24.info.version&label=TestPyPI&color=green)](https://test.pypi.org/project/DSCI-524-group32-df-optimizer/) [![GitHub release](https://img.shields.io/github/v/tag/UBC-MDS/DSCI_524_group32_df_optimizer?color=green)](https://github.com/UBC-MDS/DSCI-524-group32-df-optimizer/releases) [![Python Version](https://img.shields.io/badge/dynamic/json?url=https://test.pypi.org/pypi/DSCI-524-group32-df-optimizer/json&query=%24.info.requires_python&label=Python&color=green)](https://test.pypi.org/project/DSCI-524-group32-df-optimizer/) |
| Meta   | [![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md) [![License](https://img.shields.io/badge/dynamic/json?url=https://test.pypi.org/pypi/DSCI-524-group32-df-optimizer/json&query=%24.info.license_expression&label=License&color=green)](https://github.com/UBC-MDS/DSCI-524-group32-df-optimizer/blob/main/LICENSE)|

This is a collaborative project developed as part of DSCI 524: Collaborative Software Development. 

## Project Summary

DataFrame Memory Optimizer is a lightweight Python package that reduces the memory footprint of pandas DataFrames by automatically selecting more efficient data types. It analyzes each column in a DataFrame and applies safe, transparent optimizations such as numeric downcasting and categorical conversion while preserving the original data values. This will allow users to work more efficiently with large datasets in data analysis, machine learning pipelines, and resource-constrained environments.

## Example on how to use our package 

To use the DataFrame Memory Optimizer, start by importing optimize_dataframe from the group_32 package. Pass any pandas DataFrame to this function, and it will return a new optimized copy while leaving the original DataFrame unchanged. The optimizer automatically applies safe numeric downcasting, converts low-cardinality text columns to categorical types, and reports columns that were intentionally left unchanged (such as identifiers or high-cardinality text). The optimized DataFrame behaves the same as the original in downstream analysis, but typically uses less memory.

```python
import pandas as pd
from group_32.optimize_dataframe import optimize_dataframe

# Create a simple DataFrame
df = pd.DataFrame({
    "region": ["US", "CA", "US", "US"],
    "quantity": [1, 2, 3, 4],
    "price": [10.5, 12.0, 9.99, 11.25]
})

# Optimize memory usage
optimized_df = optimize_dataframe(df)

# Inspect optimized dtypes
print(optimized_df.dtypes)
```

## Planned Functions 

The package is designed around a simple, reproducible workflow: users call optimize_dataframe() once, and the function applies a consistent set of memory-saving steps while keeping the original DataFrame unchanged.

`optimize_dataframe(df)`
The main user-facing entry point. It creates a copy of `df` and coordinates the full optimization pipeline by (1) downcasting numeric columns, (2) converting low-cardinality text columns to `category`, and (3) running a diagnostics pass to report columns that were intentionally left unchanged (e.g., IDs). Returns the optimized DataFrame.

`optimize_numeric(df)`
Numeric optimizer used by `optimize_dataframe()`. It reduces memory usage by downcasting integer columns to the smallest safe integer dtype and optionally downcasting floating-point columns (e.g., float64 → float32) with minimal expected precision impact.

`optimize_categorical(df, max_unique_ratio=0.5)`
Categorical optimizer used by `optimize_dataframe()`. It converts low-cardinality string/object columns to pandas category dtype based on the ratio of unique values to total rows. This is especially effective for repeated labels like status codes, regions, or categories.

`analyze_special_columns(df)`
Diagnostics and transparency helper used by `optimize_dataframe()`. It identifies columns that may require special handling (e.g., high-cardinality IDs, coordinate columns, free-text fields) and reports them to help users understand optimization decisions. This function does not modify the DataFrame.

# Developer Documentation
The following sections describe how developers, instructors, and TAs can set up the project locally, run tests, and build/deploy documentation.
## 1. Clone the repository and set Up the Development Environment
This project uses conda for environment management. To set up the development environment, run:
```bash
git clone git@github.com:UBC-MDS/DSCI_524_group32_df_optimizer.git
cd DSCI_524_group32_df_optimizer
conda env create -f environment.yml
conda activate dataframe-memory-optimizer
```
The environment.yml file contains all required dependencies for development, testing, and documentation.
## 2. Install the Package 
To install the package in development mode, navigate to the project root directory and run:
```bash
pip install -e .
```
This allows you to make changes to the source code and have them reflected immediately without reinstalling the package.
## 3. Run Tests
To run the test suite, use pytest. From the project root directory, execute:
```bash
pytest 
```
This will discover and run all tests located in the tests/ directory.
## 4. Build and Serve Documentation
The project uses quartodoc for documentation. To build and serve the documentation locally, run:
```bash
quarto render
```
This will generate the documentation in the docs/ directory. You can then open the generated HTML files in your web browser to view the documentation.
## 5. Deploy Documentation (Automated)
Documentation deployment is fully automated via GitHub Actions.

Deployment workflow :-

- Triggered automatically on pushes to the main branch
- Builds documentation using Quarto
- Deploys the site to GitHub Pages

The live documentation is available at:
```bash
https://ubc-mds.github.io/DSCI_524_group32_df_optimizer/
```
## How the Package Fits in the Python Ecosystem

Pandas provides low-level tools for memory optimization (e.g., pd.to_numeric(..., downcast=...), DataFrame.convert_dtypes()), but these require manual orchestration and offer limited transparency.

DataFrame Memory Optimizer combines these tools into a single, opinionated workflow with clear diagnostics, making memory optimization easier, safer, and more reproducible.
## Contributors

Mohammed Ibrahim

Roganci Fontelera

Wai Yan Lee

William Chong

## Copyright

- Copyright © 2026 William Chong, Wai Yan Lee, Roganci Fontelera, Mohammed Ibrahim.
- Free software distributed under the [MIT License](./LICENSE).
