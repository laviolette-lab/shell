.DEFAULT_GOAL := help

.PHONY: help test cov lint format fix types docs serve-docs build clean install pre-commit

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode with hatch
	pip install hatch
	hatch env create

test: ## Run tests
	hatch run test:test

cov: ## Run tests with coverage report
	hatch run test:cov

lint: ## Run Ruff linter
	hatch run lint:check

format: ## Format code with Ruff
	hatch run lint:format

fix: ## Auto-fix lint issues and format
	hatch run lint:all

types: ## Run mypy type checking
	hatch run types:check

docs: ## Build documentation
	hatch run docs:build-docs

serve-docs: ## Serve documentation locally
	hatch run docs:serve-docs

build: ## Build wheel and sdist
	hatch build

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ site/ htmlcov/
	rm -f coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

pre-commit: ## Install and run pre-commit hooks
	pre-commit install
	pre-commit run --all-files

docker-dev: ## Build and run dev Docker image
	docker build --target dev -t shell:dev .

docker-prod: ## Build production Docker image
	docker build --target prod -t shell:prod .
