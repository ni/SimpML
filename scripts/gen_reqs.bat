@echo off

poetry export --without-hashes --only main -f requirements.txt -o requirements.txt
poetry export --without-hashes --only dev -f requirements.txt -o dev_requirements.txt
poetry export --without-hashes --only docs -f requirements.txt -o doc_requirements.txt
