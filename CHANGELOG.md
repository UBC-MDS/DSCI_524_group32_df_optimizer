# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.1] - 2026-01-10
- Initial release of the dataframe optimizer package.
- wrot documentation for functions and usage examples.

## [0.0.2] - 2026-01-17
- Core function to reduce pandas DataFrame memory footprint via dtype optimization.
- Input validation to ensure only pandas DataFrames are accepted.
- Unit tests covering common numeric and categorical optimization scenarios.

## [0.0.3] - 2026-01-24
- GitHub Actions workflow for automated testing on push and pull requests.
- Quarto-based documentation website with usage examples and API references.
- added tests for functions to test the workflow deployment automation.

## [0.1.5] - 2026-02-01
- Automated version release and deployment to TestPyPI via GitHub Actions upon merging to main branch.
- Updated documentation with installation instructions from TestPyPI.
- addressed issues from peer reviews and improved error handling.

## Peer review changes - 2026-01-29
- Added clickable link for documentation site in README.md
- Added installation instructions for TestPyPI in README.md