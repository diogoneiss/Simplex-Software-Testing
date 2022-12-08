import runpy
from pathlib import Path

"""


NOTA: talvez esse arquivo não funcione, testei apenas no meu computador... 

Se isso acontecer, por favor use o arquivo na pasta src/



Meus arquivos estão dentro da pasta src. Esse script apenas chama a main.py correta lá dentro.

Para entendimento, temos um arquivo main, que possui a classe SimplexRunner, que executa a fase 1 e 2

* auxiliar_lp.py: responsavel pelo auxiliar
* exceptions.py, que define as exceções utilizadas no programa no caso de inviavel ou ilimitada
* tableau.py, que lê o arquivo de entrada e cria o tableau no formato correto

Dentro da pasta Utils temos o arquivo linear_algebra.py, que possui funções úteis e modulares para lidar com vários aspectos do simplex.

Esse trabalho foi feito para a disciplina de teste de software, então temos uma pasta /tests, com vários testes em desenvolvimento dentro.

"""

import subprocess

if __name__ == "__main__":
    # nao sei se voce vai rodar com python ou python3
    try:
        subprocess.run(["python3", "src/main.py"])
    except:
        subprocess.run(["python", "src/main.py"])
