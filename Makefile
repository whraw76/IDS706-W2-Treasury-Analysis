PYTHON ?= python3
IMAGE ?= treasury-analysis:part-a
SOURCES = analysis.py compare_polars.py tests

.PHONY: install test format lint check run docker-build docker-run docker-test

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -v

format:
	$(PYTHON) -m black $(SOURCES)

lint:
	$(PYTHON) -m flake8 $(SOURCES)

check:
	$(PYTHON) -m black --check $(SOURCES)
	$(PYTHON) -m flake8 $(SOURCES)
	$(PYTHON) -m pytest -v

run:
	$(PYTHON) analysis.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	mkdir -p results
	docker run --rm -v "$(CURDIR)/results:/app/results" $(IMAGE)

docker-test:
	docker run --rm $(IMAGE) python -m pytest -v
