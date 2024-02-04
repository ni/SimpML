@echo off

python -m pip install --user pipx==1.2.0

python -m pipx install poetry==1.5.1

python -m pipx install poethepoet==0.21.1

poetry install --no-root
