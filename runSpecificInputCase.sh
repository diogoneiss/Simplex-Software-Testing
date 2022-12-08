#!/bin/bash

# sem argumentos, assinala todos os casos
if [ "$#" -lt 1 ]; then
    echo "No arguments provided, running all cases"
    # criar array de strings enumerando os casos
    argsCopy=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15")
else
    argsCopy=("$@")
fi

for ITEM in "${argsCopy[@]}";
    do
        # funciona pegando um arquivo de entrada teste, com o nome indicando o numero do caso, #
        # e printando o respectivo resultado no arquivo de saida, identificado por res-#
        # exemplo: a entrada chama 1 e a  saida chamará res-1
        printf "______Running input case $ITEM _________________" ;

        printf "\n\n" ;
        # passa a entrada como se estivesse digitando no terminal os valores, com o operador
        # de redirecionamento "<"
        python3 main.py < tests/cases/Testes/"$ITEM" ;
        # ^ troque acima o caminho da main e o caminho para cada arquivo de entrada
        printf "\n\n" ;
        printf "Input $ITEM ran, correct output is\n" ;

        cat tests/cases/Saidas/"$ITEM" ;
        # troaue acima pelo caminho do seu arquivo de saida
        printf "\n"

    done
