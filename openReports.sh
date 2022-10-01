#!/bin/bash

echo "Opening report pages in your prefered browser using python. You must have python or python3 at PATH.";

PythonCode="
import webbrowser
webbrowser.open('./report.html')
webbrowser.open_new_tab('./cov_html/index.html')
"

if hash python3 2>/dev/null; then
    echo "Python3 is installed"
        python3 -c "$PythonCode";
    else
       python -c "$PythonCode";
    fi
