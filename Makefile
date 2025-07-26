# Simple developer conveniences

LOCK_CMD=pdm lock && pdm export -G rocm --no-hashes -o requirements-all.txt

.PHONY: lock build run shell test

lock:
	$(LOCK_CMD)

build:
	docker compose build

run:
	docker compose up

shell:
	./scripts/dev_shell.sh

test:
	pytest -q
