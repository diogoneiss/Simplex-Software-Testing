echo "Instalando pacotes localmente, assim dá pra importar na pasta testes"

pip install --editable .
python -m pytest