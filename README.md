
# Aplicação prática de testes de software no algoritmo de otimização Simplex

Carlos Magalhães e Diogo Neiss

## Sistema implementado

Escolhemos implementar o algoritmo de programação linear Simplex duas fases, usado amplamente para problemas de otimização e ensinado na disciplina de Pesquisa Operacional.


Ele trata problemas de otimização no formato padrão de igualdades, maximizando a função C^t * x sujeita a Ax <= b e x >= 0, com C sendo a função objetivo, A a matriz de coeficientes das restrições e b os valores das restrições, e buscando o vetor x que atinja o valor máximo.

Sua entrada é receber, em ordem
* dois inteiros n e m, o número de restrições e variáveis
* uma linha, de largura m, com a função objetivo correspondente ao vetor C
* n linhas, cada uma com m+1 inteiros, representando as m restrições, e o valor da restrição, b, como o inteiro adicional

O algoritmo avaliará se a solução da programação linear é ótima ou se o problema inserido é inviável ou ilimitado. Para cada um desses casos, um certificado é emitido, possibilitando multiplicar ambos os lados e verificar se o certificado de otimilidade, ilimitada ou inviável funciona nas condições necessárias.

## Tecnologias utilizadas

Utilizamos pytest para testes e numpy para as operações numéricas/álgebra linear necessárias. O código foi seperado em `src` e `tests`, com um arquivo `conftest.py` fazendo os ajustes pré testes necessários e configuração de fixtures globais.

Foram criados também alguns shell scripts para executar rotinas úteis, como rodar os testes, gerar relatórios de cobertura e até um primitivo teste de integração não verificável

O sistema foi desenvolvido com a IDE Pycharm, oferecendo suporte para debug e acompanhamento detalhados dos testes.

Foi implementado um esquema optativo de cobertura de testes e geração de relatórios de teste, rodável com o script `createReports.sh`, que criará os arquivos htmls.
O arquivo `openReports.sh` tentará abrir o browser e mostrar os artefatos gerados

## Como rodar

O sistema é preferencialmente rodado com entradas redirecionadas, criamos vários arquivos de entrada fixos e eles são inseridos no programa. Isso pode ser feito com o shellscript `runWithSpecificInputCase.sh`, que ou recebe o caso específico que você quer, de "01" até "11", ou roda todos.


