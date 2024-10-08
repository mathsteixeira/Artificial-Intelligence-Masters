import random
import time
import numpy as np

# Configurações de aprendizado e do ambiente
Alpha = 0.2  # Taxa de aprendizado
Gamma = 0.9  # Fator de desconto
Epsilon = 0.2  # Taxa de exploração
A = 4  # Número de ações (baixo, direita, cima, esquerda)
COL, LIN = 5, 5  # Dimensões do grid
S = COL * LIN  # Número de estados
EPISODIOS = 50  # Número de episódios

# Definição de células
LIVRE, OBSTACULO, SAIDA = 0, 1, 2

# Definição do grid com obstáculos e a saída
mapa = np.array([[1, 1, 1, 2, 1], [1, 0, 0, 0, 1], [1, 0, 1, 1, 1],
                 [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])

# Tabela Q (inicializada aleatoriamente entre 0 e 1)
Q = np.random.rand(S, A)

# Variáveis globais
x, y = 0, 0
rew = 0


# Funções auxiliares
def init_q(Q):
    for s in range(S):
        for a in range(A):
            Q[s, a] = random.random()


def inicio_aleatorio():
    global x, y
    while True:
        y = random.randint(0, LIN - 1)
        x = random.randint(0, COL - 1)
        if mapa[y, x] == LIVRE:
            break


def estado(x, y):
    return y * COL + x


def seleciona_acao(Q, s):
    if random.random() < Epsilon:
        return random.randint(0, A - 1)
    return np.argmax(Q[s])


def proximo_estado(a):
    global x, y, rew
    rew = 0
    if a == 0 and y + 1 < LIN and mapa[y + 1, x] != OBSTACULO:  # Baixo
        y += 1
    elif a == 1 and x + 1 < COL and mapa[y, x + 1] != OBSTACULO:  # Direita
        x += 1
    elif a == 2 and y - 1 >= 0 and mapa[y - 1, x] != OBSTACULO:  # Cima
        y -= 1
    elif a == 3 and x - 1 >= 0 and mapa[y, x - 1] != OBSTACULO:  # Esquerda
        x -= 1
    else:
        rew = 1  # Colidiu
    return estado(x, y)


def recompensa():
    if mapa[y, x] == SAIDA:
        return 100
    if rew == 1:
        return -5
    return -1


def atualiza_q(s, a, r, Q, next_s, next_a):
    Q[s, a] = Q[s, a] + Alpha * (r + Gamma * Q[next_s, next_a] - Q[s, a])


def desenha_mapa_politica(espaco, episodio):
    print("\n\n\n ===== Q-LEARNING ===== \n")
    print(f"Episódio: {episodio}\n")
    for linha in range(LIN):
        for coluna in range(COL):
            if mapa[linha, coluna] == LIVRE:
                esp = estado(coluna, linha)
                if espaco[esp] == 0:
                    print("v", end="")  # Baixo
                elif espaco[esp] == 1:
                    print(">", end="")  # Direita
                elif espaco[esp] == 2:
                    print("^", end="")  # Cima
                elif espaco[esp] == 3:
                    print("<", end="")  # Esquerda
            elif mapa[linha, coluna] == OBSTACULO:
                print("#", end="")
            elif mapa[linha, coluna] == SAIDA:
                print(" ", end="")
        print()


# Código principal (main)
random.seed(time.time())
init_q(Q)

# Exibindo a tabela Q inicial
print("Tabela Q inicial")
for s in range(S):
    for a in range(A):
        print(f"{Q[s, a]:.2f} ", end="")
    print()

# Loop principal de aprendizado
for episodio in range(EPISODIOS):
    inicio_aleatorio()
    s = estado(x, y)

    while mapa[y, x] != SAIDA:
        at = seleciona_acao(Q, s)
        s_proximo = proximo_estado(at)
        r = recompensa()
        a_proximo = seleciona_acao(Q, s_proximo)
        atualiza_q(s, at, r, Q, s_proximo, a_proximo)
        s = s_proximo

    # Desenha a política a cada 5 episódios
    if episodio % 5 == 0:
        politica = [np.argmax(Q[s]) for s in range(S)]
        desenha_mapa_politica(politica, episodio)
        time.sleep(1)

# Exibindo a tabela Q final
print("\nTabela Q final")
for s in range(S):
    for a in range(A):
        print(f"{Q[s, a]:.2f} ", end="")
    print()
