all: deps

deps:
	pip install pipenv --upgrade
	pipenv install --dev --skip-lock

test:
	pipenv run pytest *.py
