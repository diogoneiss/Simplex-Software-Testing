#!/bin/bash

# sem argumentos, assinala todos os casos
if [ "$#" -lt 1 ]; then
    echo "No arguments provided, running all cases"
    # criar array de strings enumerando os casos
    argsCopy=("1" "2" "3" "4" "5" "6" "7")
# se tiver argumentos copia pro array
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
        python3 src/main.py < tests/cases/end2end/"$ITEM" ;
        # ^ troque acima o caminho da main e o caminho para cada arquivo de entrada
        printf "\n\n" ;
        printf "Input $ITEM ran, correct output is\n" ;

        cat tests/cases/end2end/res-"$ITEM" ;
        # troaue acima pelo caminho do seu arquivo de saida
        printf "\n"

    done
