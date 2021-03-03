
from collections import defaultdict
from scipy.linalg import expm
import numpy

from markov_utils import transforma_em_matriz_de_taxas

# while tempo_passado < tempo_total:
#     # TE = Tempo do estado N
#     te0 = numpy.random.exponential(1 / (0.5 * n1 * mu_0 * (f0(2) * n0 + n1)))
#     te1 = numpy.random.exponential(1 / (n2 * mu_0 * (f0(2) * n0 + n1)))
#     te2 = numpy.random.exponential(1 / (n0 * mu_1 * (n1 + f1(2) * n2)))
#     te3 = numpy.random.exponential(1 / (0.5 * n1 * mu_1 * (n1 + f1(2) * n2)))
#
#     tempo_passado += min(te0, te1, te2, te3)
#     estado_proximo = numpy.argmin(te0, te1, te2, te3)
#     if estado_proximo == 0:
#         # Estado 1: n00-1, n01 + 1, n10, n11
#         # Taxa: n00 mu_1 (n00 + n10  + f1(2)*n11)
#         n0 += 1
#         n1 -= 1
#     elif estado_proximo == 1:
#         # Estado 2: n00 + 1, n01, n10 - 1, n11
#         n1 += 1
#         n2 -= 1
#     elif estado_proximo == 2:
#         # Estado 3: n00, n01 - 1, n10 + 1, n11
#         # Taxa: n01 mu_0 (f0(2)*n00 + (n01 - 1) + n10)
#         n0 -= 1
#         n1 += 1
#     elif estado_proximo == 3:
#         # Estado 4: n00, n01 - 1, n10, n11 + 1
#         # Taxa: n01 * mu_1 ((n01 - 1) + n10 + f1(2)*n11)
#         n1 -= 1
#         n2 += 1
#
#     eixo_x.append(tempo_passado)
#     eixo_y.append((n0, n1, n2))
#
# import matplotlib.pyplot as plt
#
# plt.plot(eixo_x, eixo_y)

def gera_estados_rnd(
        populacao, estado_atual=None, estados=None
):

    if estado_atual is None:
        estado_atual = (populacao, 0, 0)

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

    # print(f'Estado {estado_atual} adicionado.')

    n0, n1, n2 = estado_atual

    proximo_estado1 = (n0 + 1, n1 - 1, n2)
    if proximo_estado1 not in estados:
        # print(f'Proximo estado 1: {proximo_estado1}')
        novos_estados = gera_estados_rnd(
            populacao,
            estado_atual=proximo_estado1,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        pass # print(f'{estado_atual} deixou de visitar {proximo_estado1}')

    proximo_estado2 = (n0 - 1, n1 + 1, n2)
    if proximo_estado2 not in estados:
        # print(f'Proximo estado 2: {proximo_estado2}')
        novos_estados = gera_estados_rnd(
            populacao,
            estado_atual=proximo_estado2,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        pass  #  print(f'{estado_atual} deixou de visitar {proximo_estado2}')

    proximo_estado3 = (n0, n1 + 1, n2 - 1)
    if proximo_estado3 not in estados:
        # print(f'Proximo estado 3: {proximo_estado3}')
        novos_estados = gera_estados_rnd(
            populacao,
            estado_atual=proximo_estado3,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        pass # print(f'{estado_atual} deixou de visitar {proximo_estado3}')

    proximo_estado4 = (n0, n1 - 1, n2 + 1)
    if proximo_estado4 not in estados:
        # print(f'Proximo estado 4: {proximo_estado4}')
        novos_estados = gera_estados_rnd(
            populacao,
            estado_atual=proximo_estado4,
            estados=estados
        )
        estados.update(novos_estados)
    else:
        pass # print(f'{estado_atual} deixou de visitar {proximo_estado4}')

    return estados



def preenche_matriz_rnd(
        populacao, lambda0, lambda1, mu0, mu1,  *,
        estado_anterior=None, estado_atual=None, taxa=None, estados=None
):

    if estado_anterior is None:
        estado_anterior = (populacao, 0, 0)

    if estado_atual is None:
        estado_atual = (populacao, 0, 0)

    if estados is None:
        estados = defaultdict(lambda: defaultdict(int))

    if taxa is None:
        taxa = 0

    if sum(estado_atual) not in range(populacao + 1):
        # print(f'Soma do estado não está no range({populacao}: {estado}')
        return dict()

    if any(s < 0 for s in estado_atual):
        # print(f'Não pode números negativos: {estado}')
        return dict()

    # Se a posição na matriz já estiver preenchida, não preencho de novo.
    # Tudo que foi visitado já foi preenchido.
    if estado_anterior in estados and estado_atual in estados[estado_anterior]:
        return dict()
    estados[estado_anterior][estado_atual] = taxa


    # print(f'Estado {estado_atual} adicionado.')

    n0, n1, n2 = estado_atual

    proximo_estado1 = (n0 + 1, n1 - 1, n2)
    # if proximo_estado1 not in estados:
        # print(f'Proximo estado 1: {proximo_estado1}')
    taxa = 0.5 * mu0 * (2 * n0 + n1)
    novos_estados = preenche_matriz_rnd(
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

    proximo_estado2 = (n0 - 1, n1 + 1, n2)     # 1, 1, 0 --> 0, 2, 0
    # if proximo_estado2 not in estados:
        # print(f'Proximo estado 2: {proximo_estado2}')
    taxa = n2 * mu0 * (2*n0 + n1)
    novos_estados = preenche_matriz_rnd(
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

    proximo_estado3 = (n0, n1 + 1, n2 - 1)
    # if proximo_estado3 not in estados:
        # print(f'Proximo estado 3: {proximo_estado3}')
    taxa = n0 * mu1 * (n1 + 2 * n2)
    novos_estados = preenche_matriz_rnd(
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

    proximo_estado4 = (n0, n1 - 1, n2 + 1)
    # if proximo_estado4 not in estados:
        # print(f'Proximo estado 4: {proximo_estado4}')
    taxa = 0.5 * n1 * mu1 * (n1 + 2 * n2)
    novos_estados = preenche_matriz_rnd(
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

    return estados



def markov_rnd(
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

    estados = gera_estados_rnd(populacao, )
    if debug:
        print('Estados gerados:', estados)

    mapeamento = {estado: i for i, estado in enumerate(estados)}
    mapeamento_reverso = {i: estado for i, estado in enumerate(estados)}
    if debug:
        print('Mapeamento:', mapeamento)

    estados_preenchidos = preenche_matriz_rnd(
        populacao,
        mu0,
        mu1,
        lambda0,
        lambda1,
    )
    # print('Estados preenchidos:', estados_preenchidos)

    Q = transforma_em_matriz_de_taxas(mapeamento, estados_preenchidos)

    # ncol, nlin = matriz_com_diagonal.shape
    # print('\t', '\t'.join(str(i) for i in mapeamento))
    # for l in range(nlin):
    #     print(f'{mapeamento_reverso[l]}', '\t\t'.join(f'{s:2.2f}' for s in matriz_com_diagonal[l, :]))
    #
    # print(matriz_com_diagonal)
    # print('[', end='')
    # for l in range(nlin):
    #     print('[', ', '.join(f'{s:2.3f}' for s in matriz_com_diagonal[l, :]), ']')
    # print(']')


    inistate = mapeamento[estado_inicial]

    timeline_probability_matrix = numpy.zeros(
        [
            len(numpy.arange(0, max_time, time_inc)),
            Q.shape[0]
        ]
    )

    for step, t in enumerate(numpy.arange(0, max_time, time_inc)):
        P_t = expm(Q * t)
        timeline_probability_matrix[step, :] = P_t[inistate, :]

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

    timeline_probability_matrix = markov_rnd(
        populacao=5,
        estado_inicial=(4, 1, 0),
        estado_final=(0, 0, 5),
        mu0=0.15,
        mu1=0.5,
        lambda0=0,
        lambda1=0,
        max_time=10,
        time_inc=0.01
    )

    from matplotlib import pyplot as plt
    plt.show()
