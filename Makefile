.PHONY: setup notebook compile lint

setup:
	python -m pip install -r requirements.txt

notebook:
	jupyter lab

compile:
	python -m compileall src

lint:
	python -m ruff check src
