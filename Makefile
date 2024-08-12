# Copyright 2024 Recursive AI

.DEFAULT_GOAL := help
.PHONY: install pre-commit pytest check pylint pyright bandit style black isort build publish help

install: ## Install/Upgrade all dependencies in editable mode
	pip install --upgrade -e '.[dev,pub,examples]'

pre-commit: ## Install pre-commit
	pre-commit install
	pre-commit run --all-files

## Testing ##
pytest: ## Execute unit tests with pytest
	python -m pytest -s -n auto

## Code checks ##
check: pylint pyright bandit ## Run pylint, pyright and bandit

pylint: ## Check code smells with pylint
	-python -m pylint --exit-zero src

pyright: ## Check types with pyright
	-python -m pyright

bandit: ## Check securty smells with bandit
	-python -m bandit -c pyproject.toml -r src

## Styling ##
style: black isort ## Run black and isort

black: ## Auto-format python code using black
	python -m black src

isort: ## Auto-format python code using isort
	python -m isort src

build: ## Build package into dist folder and check artifacts
	rm -f dist/*
	python -m build .
	python -m twine check dist/*

publish: ## Publish artifacts to private pypi repository
	python -m twine upload --repository pypi dist/*

help: # Run `make help` to get help on the make commands
	@echo "\033[36mAvailable commands:\033[0m"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
