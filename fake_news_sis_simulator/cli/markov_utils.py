import numpy

def transforma_em_matriz(mapeamento, preenchidos):
    matriz = numpy.zeros((len(mapeamento), len(mapeamento)))
    for estado, conectados in preenchidos.items():
        for conectado, valor in conectados.items():
            linha_mapeada = mapeamento[estado]
            coluna_mapeada = mapeamento[conectado]
            matriz[linha_mapeada][coluna_mapeada] = valor

    return matriz


def preenche_diagonais(matriz):
    com_diagonal = numpy.copy(matriz)
    nlin, ncol = matriz.shape
    for i in range(nlin):
        com_diagonal[i][i] = -sum(matriz[i, :])
    return com_diagonal


def transforma_em_matriz_de_taxas(mapeamento, preenchido):
    matriz = transforma_em_matriz(mapeamento, preenchido)
    return preenche_diagonais(matriz)
