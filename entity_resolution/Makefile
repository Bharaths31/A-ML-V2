# V2 Business Entity Resolution — reproducible pipeline targets.
# Override PYTHON if you use a different interpreter/venv.
PYTHON ?= python3
DATA_ROOT ?= data
CONFIG ?= config.yaml
TEAM ?= team

.PHONY: setup profile train test predict validate package all clean

setup:
	$(PYTHON) setup.py

profile:
	$(PYTHON) master_profiler.py --stage all --data-root $(DATA_ROOT) --out profile

train:
	$(PYTHON) master_trainer.py --stage train --config $(CONFIG) --data-root $(DATA_ROOT)

test:
	$(PYTHON) -m pytest -q er_core/tests

predict:
	$(PYTHON) master_trainer.py --stage predict --config $(CONFIG) --data-root $(DATA_ROOT)

validate:
	$(PYTHON) submit.py --team-name $(TEAM) --validate-only --data-root $(DATA_ROOT)

package:
	$(PYTHON) submit.py --team-name $(TEAM) --data-root $(DATA_ROOT)

all: profile train predict package

clean:
	rm -rf cache profile output models __pycache__ .pytest_cache
