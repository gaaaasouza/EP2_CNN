# Análise e Otimização de Redes Neurais para Classificação de Dígitos (MNIST)

Este projeto explora a classificação de dígitos do dataset MNIST utilizando duas abordagens distintas de redes neurais, e implementa um sistema de otimização de hiperparâmetros para encontrar a melhor arquitetura de rede para cada caso.

As abordagens implementadas são:
1.  **CNN Padrão**: Uma Rede Neural Convolucional que opera diretamente sobre os pixels brutos das imagens.
2.  **Rede Densa com HOG**: Uma Rede Neural Densa (MLP) que opera sobre características extraídas das imagens utilizando o descritor HOG (Histogram of Oriented Gradients).

Para cada abordagem, o projeto treina e avalia modelos para classificação **multiclasse** (dígitos de 0 a 9) e **binária** (dígitos < 5 vs. dígitos >= 5).

## Estrutura do Projeto

-   `cnn_bruto.py`: Contém o código para a CNN padrão, incluindo as funções para treinamento multiclasse e binário, e salvamento de todos os artefatos (pesos, histórico, matriz de confusão, etc.).
-   `cnn_hog.py`: Contém o código para a rede densa que utiliza HOG, com estrutura similar ao `cnn_bruto.py`.
-   `grid_search.py`: Script orquestrador que realiza a busca em grade (Grid Search) para encontrar os melhores hiperparâmetros para os modelos multiclasse de ambas as abordagens.
-   `outputs_*`: Pastas geradas automaticamente para armazenar os resultados de cada execução, incluindo hiperparâmetros, pesos, histórico de treinamento, predições e imagens das matrizes de confusão.

## Pré-requisitos

Para executar este projeto, você precisará de Python 3 e das seguintes bibliotecas:

-   TensorFlow (Keras)
-   scikit-learn
-   scikit-image
-   NumPy
-   Matplotlib
-   Seaborn

## Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <NOME_DA_PASTA_DO_PROJETO>
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Para macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    Crie um arquivo chamado `requirements.txt` com o conteúdo abaixo:
    ```txt
    tensorflow
    scikit-learn
    scikit-image
    numpy
    matplotlib
    seaborn
    ```
    Em seguida, instale as bibliotecas com o pip:
    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

Existem duas maneiras principais de executar o código:

### 1. Execução Independente

Você pode treinar os modelos com seus parâmetros padrão para gerar os resultados e as matrizes de confusão.

-   **Para rodar o modelo CNN Bruto (multiclasse e binário):**
    ```bash
    python cnn_bruto.py
    ```
    Os resultados serão salvos nas pastas `outputs_bruto` e `outputs_bruto_binario`.

-   **Para rodar o modelo com HOG (multiclasse e binário):**
    ```bash
    python cnn_hog.py
    ```
    Os resultados serão salvos nas pastas `outputs_hog` e `outputs_hog_binario`.

### 2. Otimização com Grid Search

Para encontrar os melhores hiperparâmetros, execute o script de busca em grade.

```bash
python grid_search.py
```

**Atenção:** Este processo pode ser **extremamente demorado**, pois treinará dezenas de redes neurais.

O script irá:
1.  Testar todas as combinações de hiperparâmetros definidas em `param_grid_bruto` e `param_grid_hog`.
2.  Gerar relatórios em formato JSON (`grid_search_report_*.json`) com o ranking de todas as combinações.
3.  Após a busca, ele treinará uma última vez os modelos com os melhores parâmetros encontrados e salvará os resultados completos nas pastas `outputs_bruto_best` e `outputs_hog_best`.

## Resultados

Todos os artefatos gerados, como os pesos da rede, o histórico de acurácia/perda por época, as predições e as imagens das matrizes de confusão, são salvos automaticamente nas respectivas pastas de `outputs`.