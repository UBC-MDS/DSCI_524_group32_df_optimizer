# Contributing

Contributions of all kinds are welcome here, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

## Example Contributions

You can contribute in many ways, for example:

* [Report bugs](#report-bugs)
* [Fix Bugs](#fix-bugs)
* [Implement Features](#implement-features)
* [Write Documentation](#write-documentation)
* [Submit Feedback](#submit-feedback)

### Report Bugs

Report bugs at https://github.com/UBC-MDS/DSCI_524_group32_df_optimizer/issues.

**If you are reporting a bug, please follow the template guidelines. The more
detailed your report, the easier and thus faster we can help you.**

### Fix Bugs

Look through the GitHub issues for bugs. Anything labelled with `bug` and
`help wanted` is open to whoever wants to implement it. When you decide to work on such
an issue, please assign yourself to it and add a comment that you'll be working on that,
too. If you see another issue without the `help wanted` label, just post a comment, the
maintainers are usually happy for any support that they can get.

### Implement Features

Look through the GitHub issues for features. Anything labelled with
`enhancement` and `help wanted` is open to whoever wants to implement it. As
for [fixing bugs](#fix-bugs), please assign yourself to the issue and add a comment that
you'll be working on that, too. If another enhancement catches your fancy, but it
doesn't have the `help wanted` label, just post a comment, the maintainers are usually
happy for any support that they can get.

### Write Documentation

Group 32 could always use more documentation, whether as
part of the official documentation, in docstrings, or even on the web in blog
posts, articles, and such. Just
[open an issue](https://github.com/UBC-MDS/DSCI_524_group32_df_optimizer/issues)
to let us know what you will be working on so that we can provide you with guidance.

### Submit Feedback

The best way to send feedback is to file an issue at
https://github.com/UBC-MDS/DSCI_524_group32_df_optimizer/issues. If your feedback fits the format of one of
the issue templates, please use that. Remember that this is a volunteer-driven
project and everybody has limited time.

## Get Started!

Ready to contribute? Here's how to set up DSCI_524_df_optimizer for
local development.

1. Fork the https://github.com/UBC-MDS/DSCI_524_group32_df_optimizer
   repository on GitHub.
2. Clone your fork locally (*if you want to work locally*)

    ```shell
    git clone git@github.com:UBC-MDS/DSCI_524_group32_df_optimizer.git
    ```

3. [Install hatch](https://hatch.pypa.io/latest/install/).

4. Create a branch for local development using the default branch (typically `main`) as a starting point. Use `fix` or `feat` as a prefix for your branch name.

    ```shell
    git checkout main
    git checkout -b fix-name-of-your-bugfix
    ```

    Now you can make your changes locally.

5. When you're done making changes, apply the quality assurance tools and check
   that your changes pass our test suite. This is all included with tox

    ```shell
    hatch run test:run
    ```

6. Commit your changes and push your branch to GitHub. Please use [semantic
   commit messages](https://www.conventionalcommits.org/).

    ```shell
    git add .
    git commit -m "fix: summarize your changes"
    git push -u origin fix-name-of-your-bugfix
    ```

7. Open the link displayed in the message when pushing your new branch in order
   to submit a pull request.

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your
   new functionality into a function with a docstring.
3. Your pull request will automatically be checked by the full test suite.
   It needs to pass all of them before it can be considered for merging.

## Development Tools and Practices

The current project applies modern software tools and organizational practices to ensure quality, reproducibility and effective collaboration between each member of the team.

### Used Tools and Infrastructures

- **GitHub** was used as main tool for version control and communication. In order to reduce errors, branch-method and  Pull Requests (PR) were created effectively.

- **GitHub Issues and Project Boards** managed the division of the tasks, ensuring an even distribution of the workload and tracking of the milestones projects.

- **Continuous Integration (CI)** was constantly implemented, running tests automatically and ensuring a correct functionality of the new branches before merging in the main.

- **pytest** automated testing helped validate the functionality of the functions

- **Environment Management** was ensured through `environment.yml` to ensure reproducibility across development environments

- **Documentation** was maintained using Quarto files in order to provide a clear guide of usage for future users.

- **Netfly** was used for the Milestone 4, to automatically deploy the project documentation website: whenever a change was pushed to the repository it triggered a new site build, ensuring a synchronized documentation.

- **Gitflow Workflow** principles were applied to structure development, improving code stability and supported parallel development.

### Organizational Practices

- The collaborators demonstrate a consistent usage of **branching** strategy that ensured a clear and well managed workflow. Before merging into `main`, at least one collaborator is required to review the PR and provide a constructive feedback or suggestion whenever needed.

- Clear guidelines of the code of conduct support and shape a clear collaboration.

### Scaling the Project

If this project were scaled to a larger or production-level application, additional tools and practices would be required. These include stronger code reviews, more tests, versioned releases, and better dependency management. Automated deployment and CI/CD pipelines would help maintain reliability as the project grows.
