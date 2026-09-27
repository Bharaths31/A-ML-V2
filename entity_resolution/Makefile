# Business Entity Resolution — reproducible pipeline targets.
# Override PYTHON if you use a different interpreter/venv.
PYTHON ?= python3
DATA_ROOT ?= ../../dataset

.PHONY: install synthdata run test validate reproduce clean

install:
	$(PYTHON) -m pip install -r requirements.txt

synthdata:
	$(PYTHON) -m src.cli.run --stage synthdata --data-root $(DATA_ROOT) \
		--n-train 400 --n-test 200 --with-test-gt

run:
	$(PYTHON) -m src.cli.run --data-root $(DATA_ROOT)

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) -m src.cli.run --stage validate --data-root $(DATA_ROOT)

# One-command reproduction: install -> synth (only if data missing) -> run -> validate.
reproduce: install run validate

clean:
	rm -rf artifacts output .pytest_cache
