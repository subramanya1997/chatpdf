.PHONY: install run clean

install:
	ARCHFLAGS="-arch arm64" poetry install --no-root

run:
	poetry run chainlit run app.py -w --port 7860

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete 