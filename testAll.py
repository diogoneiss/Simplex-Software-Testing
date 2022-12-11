import subprocess
import difflib
import os
from pathlib import Path

# The folder in which the file you want to run is
# not needed, as the file is in current directory
src_path = os.path.join(os.path.dirname(__file__), "src")

# file to run
file_to_run = os.path.join(src_path, "main.py")

# Set the directories
input_directory = 'tests/cases/Testes'
output_directory = 'tests/cases/Saidas'

# Get all of the filenames in the directory
filenames = os.listdir(input_directory)

# Create an empty array to store the filenames
input_files = []

# Iterate through the filenames and append them to the array
for filename in filenames:
    input_files.append(str(Path(input_directory) / Path(filename)))

# Get all of the filenames in the directory
filenames = os.listdir(output_directory)

# Create an empty array to store the filenames
output_files = []

# Iterate through the filenames and append them to the array
for filename in filenames:
    output_files.append(str(Path(output_directory) / Path(filename)))

for i, input_file in enumerate(input_files):
    print(f"Running {input_file}...", end="")
    # Open the input file and read its contents
    with open(input_file, "r") as f:
        input_str = f.read()

    # Run the file and pass the input to it
    try:
        process = subprocess.run(
            ['python3', file_to_run],
            input=input_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
    except:
        process = subprocess.run(
            ['python', file_to_run],
            input=input_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )

    # The file you want to compare the output to
    expected_output_file = output_files[i]

    # Read the expected output file
    with open(expected_output_file, "r") as f:
        expected_output_str = f.read()

    output_lines = process.stdout.splitlines()
    expected_output_lines = expected_output_str.splitlines()

    convertedNumbers = [[]]
    convertedExpectedNumbers = [[]]

    for j in range(1, len(output_lines)):
        line = output_lines[j]
        convertedNumbers.append([round(float(x), 3) for x in line.split()])
        convertedExpectedNumbers.append([round(float(x), 3) for x in expected_output_lines[j].split()])

    output_lines = [output_lines[0]] + [' '.join([str(x) for x in line]) for line in convertedNumbers]
    expected_output_lines = [expected_output_lines[0]] + [' '.join([str(x) for x in line]) for line in
                                                          convertedExpectedNumbers]

    if expected_output_lines[0] != 'otima':
        # leave just first line
        output_lines = output_lines[:1]
        expected_output_lines = expected_output_lines[:1]

    # Compute the difference between the expected output and the actual output
    diff = difflib.ndiff(
        output_lines,
        expected_output_lines
    )

    changes = [l for l in diff if l.startswith('+ ') or l.startswith('- ')]
    if len(changes) == 0:
        print("Ok")
    else:
        print(" Mismatch. \nDifferences in output:")
        for line in changes:
            print(line)
