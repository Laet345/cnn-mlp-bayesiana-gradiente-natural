# Classificação de Imagens com Cabeça MLP Bayesiana e Gradiente Natural

Estudo prático e teórico sobre **quantificação de incerteza** e **detecção de dados fora da distribuição (OOD)** em visão computacional, comparando uma cabeça classificadora determinística contra abordagens bayesianas acopladas a um extrator convolucional pré-treinado.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laet345/cnn-mlp-bayesiana-gradiente-natural/blob/main/cnn-mlp-bayesiana-gradiente-natural.ipynb)

---

## Visão Geral da Arquitetura

O experimento utiliza uma abordagem híbrida em duas etapas sobre a base **Dogs vs. Cats**:

1. **Extrator Convolucional (Congelado):** uma CNN pré-treinada extrai um vetor latente de **128 atributos** ($\boldsymbol{\phi}(\mathbf{x})$) por imagem ($128 \times 128$ RGB), padronizado exclusivamente com as estatísticas do conjunto de treino ($Z_{\text{treino}}$) para evitar vazamento de dados.
2. **Cabeça Classificadora MLP Não Convexa (`128 -> 32 -> 16 -> 1`):** rede multicamada com ativações **ReLU** e saída sigmoide ($P = 4.673$ parâmetros), avaliada sob três regimes a partir da mesma inicialização de He:
   * **MLP Determinística (Adam):** estimativa pontual de pesos sem distribuição;
   * **MLP Bayesiana via Gradiente Natural Diagonal (VOGN):** inferência variacional gaussiana $q(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\mathbf{S}^{-1}))$ com custo linear $O(P)$ e passos regressivos ($\alpha_t, \beta_t: 0{,}1 \to 0{,}01$);
   * **MLP Bayesiana via Laplace + Newton-LM:** aproximação gaussiana local no ponto MAP usando a matriz densa de Gauss-Newton ($P \times P$).

---

## Principais Resultados e Diagnósticos

| Regime | Complexidade / Memória | Estabilidade em MLP ReLU | Incerteza Epistêmica (BALD) | Superconfiança (>90%) em OOD |
| :--- | :---: | :---: | :---: | :---: |
| **MLP Determinística (Adam)** | $O(P)$ | Alta | Não | Alta (palpite pontual rígido) |
| **MLP Bayesiana (GN Diagonal)** | $O(P)$ (~19 KB) | Alta | **Sim** (Monte Carlo) | **Cai de 82,7% para 26,6%** |
| **MLP Bayesiana (Laplace + Newton)** | $O(P^2)$ / $O(P^3)$ | **Falha** (Hessiana indefinida/singular) | Indisponível | Indisponível |

* **Por que Laplace/Newton falha na rede não convexa:** diagnósticos empíricos via *autodiff* (produtos Hessiana-vetor) demonstram que as ativações ReLU tornam a Hessiana exata indefinida (com milhares de autovalores negativos), além da simetria de permutação entre neurônios ocultos gerar $32! \times 16! \approx 10^{49}$ modos equivalentes impossíveis de capturar com uma única gaussiana local.
* **Detecção de Erros e OOD:** usando $K = 200$ amostras Monte Carlo da posterior treinada com Gradiente Natural, a **entropia preditiva** e a **informação mútua (BALD)** separam eficazmente previsões incorretas e imagens de domínios inéditos (atingindo **AUROC de 0,8531** na separação entre o conjunto de teste e 1.000 imagens de flores).

---

## Estrutura do Notebook

* **Seções 1 a 5:** Carregamento, pré-processamento e extração das representações convolucionais de 128 dimensões.
* **Seção 6 (`6.1` a `6.9`):** Definição da MLP não convexa, Jacobiano analítico, busca da precisão do prior ($\delta$) na validação, comparação de desempenho/calibração (ECE), autópsia espectral da Hessiana exata e experimentos de incerteza em erros e dados OOD (`flower_photos` e teste qualitativo).
* **Seção 7:** Síntese teórica, tabela comparativa e direções futuras (aproximações estruturadas K-FAC).

---

## Como Executar

1. Clique no botão **Open in Colab** no topo desta página (ou baixe o arquivo `cnn-mlp-bayesiana-gradiente-natural.ipynb`).
2. Para reproduzir a extração de características sem retreinar a CNN base, certifique-se de apontar o caminho do extrator pré-treinado (`cnn_base_cats_dogs.keras`) ou execute as células iniciais de treinamento da rede convolucional.

## Tecnologias Utilizadas

* **Python 3** | **TensorFlow / Keras** (autodiff, HVP e operações vetorizadas em GPU)
* **NumPy** & **Scikit-learn** (métricas de calibração ECE e AUROC)
* **Matplotlib** (visualização das distribuições preditivas e epistêmicas)