# ChartExtractorSupplements

This repository houses two types of content: (1) jupyter notebooks that run experiments to improve ChartExtractor and (2) useful scripts for working with ChartExtractor.

### Getting Set Up

#### Where To Place Data

- When you pull this repository, there will be an empty directory called `Data` that contains a `.gitkeep` file. This file should remain in this directory, do not delete it.
- Add your data files to this directory. These files should and will be ignored by git.

#### Downloading Necessary Packages

- Install poetry using pip to start
  ```bash
  pip install poetry
  ```
- I have created the pyproject.toml files so you don't have to worry about any of that. Just do the below.
- Add configuration to have venv in project directory
  ```bash
  poetry config virtualenvs.in-project true
  ```
- Set up venv using poetry
  ```bash
  poetry install
  ```
- Now you should have a created venv that you can switch into with the following command and run the python scripts
  ```bash
  poetry shell
  ```
- As you develop you can add packages with the following command
  ```bash
  poetry add <package-name>
  ```
