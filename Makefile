POETRY ?= poetry

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make \033[36m<target>\033[0m\n"} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		/^[a-zA-Z0-9_.-]+:.*## / { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: check.lock
check.lock:
	@$(POETRY) check --lock

##@ Install
.PHONY: install
install: check.lock ## Install the package with its runtime dependencies
	@$(POETRY) install

.PHONY: install.dev
install.dev: check.lock ## Install with the dev dependency group
	@$(POETRY) install --with dev

.PHONY: install.dev.only
install.dev.only: check.lock ## Install only the dev dependency group
	@$(POETRY) install --only dev

.PHONY: install.doc
install.doc: check.lock ## Install only the docs dependency group
	@$(POETRY) install --only docs

##@ Test
.PHONY: unit-tests
unit-tests: install.dev ## Run unit tests (CPU only)
	@$(POETRY) run python -m pytest -v -m "not gpu and not multi_gpu" tests/unittests

.PHONY: gpu-unit-tests
gpu-unit-tests: install.dev ## Run unit tests on a single GPU
	@$(POETRY) run python -m pytest -v -m "gpu" tests/unittests

.PHONY: multi-gpu-unit-tests
multi-gpu-unit-tests: install.dev ## Run unit tests on multiple GPUs
	@$(POETRY) run python -m pytest -v -m "multi_gpu" tests/unittests

.PHONY: functional-tests
functional-tests: install.dev ## Run functional tests (CPU only)
	@$(POETRY) run python -m pytest -v -m "not gpu and not multi_gpu" --ref /localdrive10TB/users/clinicadl.ci/clinicadl_data_ci/data_ci tests/functional

.PHONY: gpu-functional-tests
gpu-functional-tests: install.dev ## Run functional tests on a single GPU
	@$(POETRY) run python -m pytest -v -m "gpu" --ref /localdrive10TB/users/clinicadl.ci/clinicadl_data_ci/data_ci tests/functional
