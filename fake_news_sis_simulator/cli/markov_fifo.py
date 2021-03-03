
from collections import defaultdict
from scipy.linalg import expm
import numpy

from markov_utils import transforma_em_matriz_de_taxas


def gera_estados_fifo(
        populacao, estado_atual=None, estados=None, debug=False
):

    if estado_atual is None:
        estado_atual = (populacao, 0, 0, 0)

    if estados is None:
        estados = set()

    if sum(estado_atual) not in range(populacao + 1):
        # print(f'Soma do estado não está no range({populacao}: {estado}')
        return set()

    if any(s < 0 for s in estado_atual):
        # print(f'Não pode números negativos: {estado}')
        return set()

        # Se tudo certo até aqui, adiciono a taxa
    estados.add(estado_atual)

    if debug:
        print(f'Estado {estado_atual} adicionado.')

    n00, n01, n10, n11 = estado_atual

    proximo_estado1 = (n00 - 1, n01 + 1, n10, n11)
    if proximo_estado1 not in estados:
        # print(f'Proximo estado 1: {proximo_estado1}')
        novos_estados = gera_estados_fifo(
            populacao,
            estado_atual=proximo_estado1,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        if debug:
            print(f'{estado_atual} deixou de visitar {proximo_estado1}')

    proximo_estado2 = (n00 + 1, n01, n10 - 1, n11)
    if proximo_estado2 not in estados:
        # print(f'Proximo estado 2: {proximo_estado2}')
        novos_estados = gera_estados_fifo(
            populacao,
            estado_atual=proximo_estado2,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        if debug:
            print(f'{estado_atual} deixou de visitar {proximo_estado2}')

    proximo_estado3 = (n00, n01 - 1, n10 + 1, n11)
    if proximo_estado3 not in estados:
        # print(f'Proximo estado 3: {proximo_estado3}')
        novos_estados = gera_estados_fifo(
            populacao,
            estado_atual=proximo_estado3,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        if debug:
            print(f'{estado_atual} deixou de visitar {proximo_estado3}')

    proximo_estado4 = (n00, n01 - 1, n10, n11 + 1)
    if proximo_estado4 not in estados:
        # print(f'Proximo estado 4: {proximo_estado4}')
        novos_estados = gera_estados_fifo(
            populacao,
            estado_atual=proximo_estado4,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        if debug:
            print(f'{estado_atual} deixou de visitar {proximo_estado4}')

    proximo_estado5 = (n00, n01, n10 + 1, n11 - 1)
    if proximo_estado5 not in estados:
        # print(f'Proximo estado 4: {proximo_estado4}')
        novos_estados = gera_estados_fifo(
            populacao,
            estado_atual=proximo_estado5,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        if debug:
            print(f'{estado_atual} deixou de visitar {proximo_estado5}')

    proximo_estado6 = (n00, n01 + 1, n10 - 1, n11)
    if proximo_estado6 not in estados:
        # print(f'Proximo estado 4: {proximo_estado4}')
        novos_estados = gera_estados_fifo(
            populacao,
            estado_atual=proximo_estado6,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        if debug:
            print(f'{estado_atual} deixou de visitar {proximo_estado6}')

    return estados



def preenche_matriz_fifo(
        populacao, lambda0, lambda1, mu0, mu1,  *,
        estado_anterior=None, estado_atual=None, taxa=None, estados=None, debug=False
):

    if estado_anterior is None:
        estado_anterior = (populacao, 0, 0, 0)

    if estado_atual is None:
        estado_atual = (populacao, 0, 0, 0)

    if estados is None:
        estados = defaultdict(lambda: defaultdict(int))

    if taxa is None:
        taxa = 0

    if sum(estado_atual) not in range(populacao + 1):
        if debug:
            print(f'Soma do estado não está no range({populacao}: {estado_atual}')
        return dict()

    if any(s < 0 for s in estado_atual):
        if debug:
            print(f'Não pode números negativos: {estado_atual}')
        return dict()

    # Se a posição na matriz já estiver preenchida, não preencho de novo.
    # Tudo que foi visitado já foi preenchido.
    if estado_anterior in estados and estado_atual in estados[estado_anterior]:
        return dict()
    estados[estado_anterior][estado_atual] = taxa

    if debug:
        print(f'Estado {estado_atual} adicionado.')

    n00, n01, n10, n11 = estado_atual

    proximo_estado1 = (n00 - 1, n01 + 1, n10, n11)
    # if proximo_estado1 not in estados:
        # print(f'Proximo estado 1: {proximo_estado1}')
    taxa = lambda1 + n00 * mu1 * (n01 + n10 + 2*n11)
    novos_estados = preenche_matriz_fifo(
        populacao,
        mu0=mu0,
        mu1=mu1,
        lambda0=lambda0,
        lambda1=lambda1,
        estado_anterior=estado_atual,
        estado_atual=proximo_estado1,
        taxa=taxa,
        estados=estados
    )
    estados.update(novos_estados)

    proximo_estado2 = (n00 + 1, n01, n10 - 1, n11)
    # if proximo_estado2 not in estados:
        # print(f'Proximo estado 2: {proximo_estado2}')
    taxa = lambda0 + n10 * mu0 * (2 * n00 + n01 + (n10 - 1))
    novos_estados = preenche_matriz_fifo(
        populacao,
        mu0=mu0,
        mu1=mu1,
        lambda0=lambda0,
        lambda1=lambda1,
        estado_anterior=estado_atual,
        estado_atual=proximo_estado2,
        taxa=taxa,
        estados=estados
    )
    estados.update(novos_estados)

    proximo_estado3 = (n00, n01 - 1, n10 + 1, n11)
    # if proximo_estado3 not in estados:
        # print(f'Proximo estado 3: {proximo_estado3}')
    taxa = lambda0 + n01 * mu0 * (2*n00 + (n01 - 1) + n10)
    novos_estados = preenche_matriz_fifo(
        populacao,
        mu0=mu0,
        mu1=mu1,
        lambda0=lambda0,
        lambda1=lambda1,
        estado_anterior=estado_atual,
        estado_atual=proximo_estado3,
        taxa=taxa,
        estados=estados
    )
    estados.update(novos_estados)

    proximo_estado4 = (n00, n01 - 1, n10, n11 + 1)
    # if proximo_estado4 not in estados:
        # print(f'Proximo estado 4: {proximo_estado4}')
    taxa = lambda1 + n01 * mu1 * ((n01 - 1) + n10 + 2*n11)
    novos_estados = preenche_matriz_fifo(
        populacao,
        mu0=mu0,
        mu1=mu1,
        lambda0=lambda0,
        lambda1=lambda1,
        estado_anterior=estado_atual,
        estado_atual=proximo_estado4,
        taxa=taxa,
        estados=estados
    )
    estados.update(novos_estados)

    proximo_estado5 = (n00, n01, n10 + 1, n11 - 1)
    # if proximo_estado4 not in estados:
    # print(f'Proximo estado 4: {proximo_estado4}')
    taxa = lambda0 + n11 * mu0 * (2 * n00 + n01 + n10)
    novos_estados = preenche_matriz_fifo(
        populacao,
        mu0=mu0,
        mu1=mu1,
        lambda0=lambda0,
        lambda1=lambda1,
        estado_anterior=estado_atual,
        estado_atual=proximo_estado5,
        taxa=taxa,
        estados=estados
    )
    estados.update(novos_estados)

    proximo_estado6 = (n00, n01 + 1, n10 - 1, n11)
    # if proximo_estado4 not in estados:
    # print(f'Proximo estado 4: {proximo_estado4}')
    taxa = lambda1 + n10 * mu1 * (n01 + (n10 - 1) + 2 * n11)
    novos_estados = preenche_matriz_fifo(
        populacao,
        mu0=mu0,
        mu1=mu1,
        lambda0=lambda0,
        lambda1=lambda1,
        estado_anterior=estado_atual,
        estado_atual=proximo_estado6,
        taxa=taxa,
        estados=estados
    )
    estados.update(novos_estados)

    return estados


def markov_fifo(
        populacao,
        estado_inicial,
        estado_final,
        mu0,
        mu1,
        lambda0,
        lambda1,
        *,
        max_time,
        time_inc,
        debug=False
):
    from matplotlib import pyplot as plt

    estados = gera_estados_fifo(populacao=populacao)
    # print('Estados gerados:', estados)

    mapeamento = {estado: i for i, estado in enumerate(estados)}
    mapeamento_reverso = {i: estado for i, estado in enumerate(estados)}
    if debug:
        print('Mapeamento:', mapeamento)

    estados_preenchidos = preenche_matriz_fifo(
        populacao=populacao,
        mu0=mu0,
        mu1=mu1,
        lambda0=lambda0,
        lambda1=lambda1,
    )
    if debug:
        print('Estados preenchidos:', estados_preenchidos)

    Q = transforma_em_matriz_de_taxas(
        mapeamento,
        estados_preenchidos
    )

    inistate = mapeamento[estado_inicial]

    timeline_probability_matrix = numpy.zeros(
        [
            len(numpy.arange(0, max_time, time_inc)),
            Q.shape[0]
        ]
    )

    for tt, t in enumerate(numpy.arange(0, max_time, time_inc)):
        P_t = expm(Q * t)
        timeline_probability_matrix[tt, :] = P_t[inistate, :]

    # Probabilidades de se ter nenhum infectado e todos completamente infectados
    plt.plot(
        numpy.arange(0, max_time, time_inc),
        timeline_probability_matrix[:, mapeamento[estado_inicial]],
        label=estado_inicial
    )
    plt.plot(
        numpy.arange(0, max_time, time_inc),
        timeline_probability_matrix[:, mapeamento[estado_final]],
        color='orange',
        label=estado_final
    )

    plt.legend()

    return timeline_probability_matrix


if __name__ == '__main__':

    timeline_probability_matrix = markov_fifo(
        populacao=5,
        estado_inicial=(4, 1, 0, 0),
        estado_final=(0, 0, 0, 5),
        mu0=0.7,
        mu1=0.25,
        lambda0=0.1,
        lambda1=0.2,
        max_time=10,
        time_inc=0.01
    )

    from matplotlib import pyplot as plt
    plt.show()
