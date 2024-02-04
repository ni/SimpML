#!/bin/bash

set -e
trap 'echo "Command failed: $BASH_COMMAND"' ERR

if [ "$VIRTUAL_ENV" != "" ]; then
    # Virtual environments do not support '--user'
    extra_pip_flags=
else
    extra_pip_flags=--user
fi

python3 -m pip install $extra_pip_flags pipx==1.2.0

python3 -m pipx install poetry==1.5.1

python3 -m pipx install poethepoet==0.21.1

poetry install --no-root
