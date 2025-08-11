.PHONY: lint test fmt train api ui feast-apply

lint:
	flake8 src
fmt:
	black src
test:
	pytest -q
train:
	python -m src.fraud.train --data data/raw/PS_20174392719_1491204439457_log.csv
api:
	uvicorn src.serving.fastapi_app:app --reload --port 8000
ui:
	streamlit run src/ui/streamlit_app.py
feast-apply:
	feast -c feature_repo apply
