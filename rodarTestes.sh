#!/usr/bin/bash
# checks which python is installed and run pytest with correct parameters
if hash python3 2>/dev/null; then
    python3  -m pytest -v
    else
       python -m pytest -v
    fi