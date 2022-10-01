#!/bin/bash
if [ "$#" -lt 1 ]; then
    echo "You must enter one test file index as an argument, like 1";
    exit 1;
fi
for ITEM in "$@"
    do
        printf "______Running input case $ITEM _________________" ;

        printf "\n\n" ;
        python3 src/main.py < tests/cases/end2end/$ITEM ;
        printf "\n\n" ;
        printf "Input $ITEM ran, correct output is\n" ;
        cat tests/cases/end2end/res-$ITEM ;
        printf "\n"

    done
