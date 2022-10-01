#!/bin/bash

pytest --html=report.html --self-contained-html &&
echo "Tests report at ./report.html" &&
pytest --cov-config=.coveragerc --cov-report html:cov_html --cov=. tests/ &&
echo "Coverage report at ./cov_html/index.html"