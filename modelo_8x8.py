# ============================================================
# MODELO AHP 8x8
# 8 critérios x 8 alternativas
# Gera: pesos dos critérios, pesos das alternativas por
# critério, RC de todas as matrizes e ranking final
# ============================================================

import numpy as np

criterios = [
    "Consumo Energético",
    "Emissões Atmosféricas",
    "Efluentes e Resíduos",
    "Retorno Econômico",
    "Investimento Necessário",
    "Conformidade Regulatória",
    "Digitalização e Automação",
    "Manutenção Preditiva"
]

alternativas = [
    "Caldeira de Recuperação",
    "Planta de Evaporação",
    "Forno de Cal",
    "Caldeira de Biomassa",
    "Digestor",
    "Torres de Branqueamento",
    "ETE",
    "Pátio de Madeiras"
]

RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12,
      6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}

def calcular_ahp(matriz, nome=""):
    n = len(matriz)
    col_sum = matriz.sum(axis=0)
    norm = matriz / col_sum
    pesos = norm.mean(axis=1)
    Aw = matriz @ pesos
    lambda_max = np.mean(Aw / pesos)
    CI = (lambda_max - n) / (n - 1)
    RC = CI / RI[n]
    print(f"\n  {nome}")
    print(f"  RC = {RC:.4f} {'✓ APROVADO' if RC < 0.10 else '✗ REPROVADO'}")
    for i, p in enumerate(pesos):
        label = criterios[i] if nome == "Critérios" else alternativas[i]
        print(f"    {label:<35} {p*100:.2f}%")
    return pesos, RC

# ── MATRIZ DE CRITÉRIOS ──────────────────────────────────────
mat_crit = np.array([
  [1,     2,     2,     3,     5,     6,     7,     8   ],
  [1/2,   1,     1,     2,     4,     5,     6,     7   ],
  [1/2,   1,     1,     2,     4,     5,     6,     7   ],
  [1/3,   1/2,   1/2,   1,     3,     4,     5,     6   ],
  [1/5,   1/4,   1/4,   1/3,   1,     2,     3,     4   ],
  [1/6,   1/5,   1/5,   1/4,   1/2,   1,     2,     3   ],
  [1/7,   1/6,   1/6,   1/5,   1/3,   1/2,   1,     2   ],
  [1/8,   1/7,   1/7,   1/6,   1/4,   1/3,   1/2,   1   ],
], dtype=float)

# ── MATRIZES DAS ALTERNATIVAS ────────────────────────────────
mats = {}

mats["Consumo Energético"] = np.array([
  [1,   2,   2,   3,   2,   5,   7,   8 ],
  [1/2, 1,   1,   2,   1,   4,   6,   7 ],
  [1/2, 1,   1,   2,   1,   4,   6,   7 ],
  [1/3, 1/2, 1/2, 1,   1/2, 3,   5,   6 ],
  [1/2, 1,   1,   2,   1,   4,   6,   7 ],
  [1/5, 1/4, 1/4, 1/3, 1/4, 1,   3,   4 ],
  [1/7, 1/6, 1/6, 1/5, 1/6, 1/3, 1,   2 ],
  [1/8, 1/7, 1/7, 1/6, 1/7, 1/4, 1/2, 1 ],
], dtype=float)

mats["Emissões Atmosféricas"] = np.array([
  [1,   1/2, 1/3, 1/2, 1,   3,   4,   5 ],
  [2,   1,   1/4, 1,   2,   4,   5,   6 ],
  [3,   4,   1,   3,   3,   6,   7,   8 ],
  [2,   1,   1/3, 1,   2,   4,   5,   6 ],
  [1,   1/2, 1/3, 1/2, 1,   3,   4,   5 ],
  [1/3, 1/4, 1/6, 1/4, 1/3, 1,   2,   3 ],
  [1/4, 1/5, 1/7, 1/5, 1/4, 1/2, 1,   2 ],
  [1/5, 1/6, 1/8, 1/6, 1/5, 1/3, 1/2, 1 ],
], dtype=float)

mats["Efluentes e Resíduos"] = np.array([
  [1,   2,   1,   2,   3,   1/2, 1/4, 3 ],
  [1/2, 1,   1/2, 1,   2,   1/3, 1/5, 2 ],
  [1,   2,   1,   2,   3,   1/2, 1/4, 3 ],
  [1/2, 1,   1/2, 1,   2,   1/3, 1/5, 2 ],
  [1/3, 1/2, 1/3, 1/2, 1,   1/4, 1/6, 1 ],
  [2,   3,   2,   3,   4,   1,   1/3, 4 ],
  [4,   5,   4,   5,   6,   3,   1,   6 ],
  [1/3, 1/2, 1/3, 1/2, 1,   1/4, 1/6, 1 ],
], dtype=float)

mats["Retorno Econômico"] = np.array([
  [1,   2,   3,   2,   2,   3,   6,   8 ],
  [1/2, 1,   2,   1,   1,   2,   5,   7 ],
  [1/3, 1/2, 1,   1/2, 1/2, 1,   4,   6 ],
  [1/2, 1,   2,   1,   1,   2,   5,   7 ],
  [1/2, 1,   2,   1,   1,   2,   5,   7 ],
  [1/3, 1/2, 1,   1/2, 1/2, 1,   4,   6 ],
  [1/6, 1/5, 1/4, 1/5, 1/5, 1/4, 1,   3 ],
  [1/8, 1/7, 1/6, 1/7, 1/7, 1/6, 1/3, 1 ],
], dtype=float)

mats["Investimento Necessário"] = np.array([
  [1,   2,   2,   3,   2,   4,   6,   7 ],
  [1/2, 1,   1,   2,   1,   3,   5,   6 ],
  [1/2, 1,   1,   2,   1,   3,   5,   6 ],
  [1/3, 1/2, 1/2, 1,   1/2, 2,   4,   5 ],
  [1/2, 1,   1,   2,   1,   3,   5,   6 ],
  [1/4, 1/3, 1/3, 1/2, 1/3, 1,   3,   4 ],
  [1/6, 1/5, 1/5, 1/4, 1/5, 1/3, 1,   2 ],
  [1/7, 1/6, 1/6, 1/5, 1/6, 1/4, 1/2, 1 ],
], dtype=float)

mats["Conformidade Regulatória"] = np.array([
  [1,   1,   1/2, 2,   2,   1/3, 3,   5 ],
  [1,   1,   1/2, 2,   2,   1/3, 3,   5 ],
  [2,   2,   1,   3,   3,   1/2, 4,   6 ],
  [1/2, 1/2, 1/3, 1,   1,   1/4, 2,   4 ],
  [1/2, 1/2, 1/3, 1,   1,   1/4, 2,   4 ],
  [3,   3,   2,   4,   4,   1,   5,   7 ],
  [1/3, 1/3, 1/4, 1/2, 1/2, 1/5, 1,   3 ],
  [1/5, 1/5, 1/6, 1/4, 1/4, 1/7, 1/3, 1 ],
], dtype=float)

mats["Digitalização e Automação"] = np.array([
  [1,   2,   2,   3,   2,   3,   4,   5 ],
  [1/2, 1,   1,   2,   1,   2,   3,   4 ],
  [1/2, 1,   1,   2,   1,   2,   3,   4 ],
  [1/3, 1/2, 1/2, 1,   1/2, 1,   2,   3 ],
  [1/2, 1,   1,   2,   1,   2,   3,   4 ],
  [1/3, 1/2, 1/2, 1,   1/2, 1,   2,   3 ],
  [1/4, 1/3, 1/3, 1/2, 1/3, 1/2, 1,   2 ],
  [1/5, 1/4, 1/4, 1/3, 1/4, 1/3, 1/2, 1 ],
], dtype=float)

mats["Manutenção Preditiva"] = np.array([
  [1,   2,   2,   3,   3,   4,   5,   6 ],
  [1/2, 1,   1,   2,   2,   3,   4,   5 ],
  [1/2, 1,   1,   2,   2,   3,   4,   5 ],
  [1/3, 1/2, 1/2, 1,   1,   2,   3,   4 ],
  [1/3, 1/2, 1/2, 1,   1,   2,   3,   4 ],
  [1/4, 1/3, 1/3, 1/2, 1/2, 1,   2,   3 ],
  [1/5, 1/4, 1/4, 1/3, 1/3, 1/2, 1,   2 ],
  [1/6, 1/5, 1/5, 1/4, 1/4, 1/3, 1/2, 1 ],
], dtype=float)

# ── CÁLCULO ──────────────────────────────────────────────────
print("=" * 55)
print("MODELO AHP 8x8")
print("=" * 55)

pesos_crit, _ = calcular_ahp(mat_crit, "Critérios")

matriz_pesos_alt = np.zeros((8, 8))
for j, crit in enumerate(criterios):
    pesos, _ = calcular_ahp(mats[crit], crit)
    matriz_pesos_alt[:, j] = pesos

pontuacao = matriz_pesos_alt @ pesos_crit
ranking = sorted(enumerate(pontuacao), key=lambda x: x[1], reverse=True)

print("\n" + "=" * 55)
print("RANKING FINAL")
print("=" * 55)
for pos, (idx, score) in enumerate(ranking):
    print(f"  {pos+1}°  {alternativas[idx]:<30} {score*100:.2f}%")
print(f"\n  Soma: {pontuacao.sum()*100:.1f}%")
