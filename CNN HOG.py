import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import datasets, layers, models, Input
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def save_outputs(dir, hiperp, init_w, train_hist, train_err, output_CNN, final_w, pred_class, real_class):
    # Verificar existência de diretório
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Salvar os hiperparâmetros da rede
    with open(os.path.join(dir, 'hiperparametros.json'), 'w') as f:
        json.dump(hiperp, f)
    
    # Salvar pesos iniciais em um arquivo txt
    with open(os.path.join(dir,'pesos_iniciais.txt'), 'w') as f:
        for i, weight_matrix in enumerate(init_w):
            f.write(f"\nPesos da Camada {i+1}:\n")
            flattened_weights = weight_matrix.flatten()
            for weight in flattened_weights:
                f.write(f"{weight}\n")


    with open(os.path.join(dir, 'historico_treinamento.json'), 'w') as f:
        json.dump(train_hist, f)
    
    # Salvar o erro cometido pela rede neural em cada iteração do treinamento
    np.savetxt(os.path.join(dir, 'erros_treinamento.csv'), train_err, delimiter=',')

    # Salvar as saídas da rede neural para os dados de teste
    np.savetxt(os.path.join(dir, 'saidas_rede_neural.csv'), output_CNN, delimiter=',')

    with open(os.path.join(dir,'pesos_finais.txt'), 'w') as f:
        for i, weight_matrix in enumerate(final_w):
            f.write(f"\nPesos da Camada {i+1}:\n")
            flattened_weights = weight_matrix.flatten()
            for weight in flattened_weights:
                f.write(f"{weight}\n")

    # Salvar as classes preditas pelo modelo em comparação às reais
    with open(os.path.join(dir,'classes_previstas.txt'), 'w') as f:
        for number, real in zip(pred_class, real_class):
            f.write(f"Previu: {number}\tReal: {real}\n")

# Extrair características HOG
def extract_hog_features(images):
    print("Extraindo HOG...")
    hog_features = []
    for image in images:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# Método que cria o modelo CNN com HOG para avaliação multiclasse
def CNN_multi(train_images, train_labels, test_images, test_labels):
    # Extrair o desvritor HOG para imagens de treino e teste
    train_hog_features = extract_hog_features(train_images)
    test_hog_features = extract_hog_features(test_images)

    # Normalizar as características HOG
    scaler = StandardScaler()
    train_hog_features = scaler.fit_transform(train_hog_features)
    test_hog_features = scaler.transform(test_hog_features)

    # Dividir os dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(train_hog_features, train_labels, test_size=0.2, random_state=42)

    # Construir o modelo
    model = models.Sequential([
        Input(shape=(train_hog_features.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    diretorio_dos_arquivos_hog = 'outputs_hog' # Diretório onde os arquivos serão salvos
    initial_weights = model.get_weights() # Pesos iniciais para passsar para o arquivo

    # Compilar o modelo
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=5, 
                        validation_data=(X_val, y_val))

    # Avaliar o modelo
    test_loss, test_acc = model.evaluate(test_hog_features, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # Pegar dados para salvar em arquivos:
    #   Hiperparâmetros:
    l_rate = model.optimizer.learning_rate.numpy()
    hiperparametros = {'taxa_de_aprendizado': float(l_rate), 'epochs': 5}

    #   Pesos finais:
    post_train_weights = model.get_weights()

    #   Histórico das épocas:
    historico_treinamento = history.history

    #   Saídas (probabilidades) para cada neurônio/classe:
    saidas_rede_neural = model.predict(test_hog_features)

    #   Erros de cada época:
    erros_treinamento = historico_treinamento['loss']

    #   Classes previstas:
    predicted_labels = np.argmax(saidas_rede_neural, axis=1)
    
    save_outputs(diretorio_dos_arquivos_hog, hiperparametros, initial_weights, historico_treinamento, erros_treinamento, saidas_rede_neural, post_train_weights, predicted_labels, test_labels)


    # Calcular e plotar a matriz de confusão
    conf_matrix = confusion_matrix(test_labels, predicted_labels)

    # Calcular acurácia
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f'Acurácia: {accuracy:.2%}')

    # Visualizar a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão')
    plt.show()

# Método que cria o modelo CNN com HOG para avaliação binária
def CNN_bin(train_images, train_labels, test_images, test_labels):
    # Selecionar duas classes para classificação binária (exemplo: dígitos 0 e 1)
    binary_classes = [0, 1]
    train_filter = np.isin(train_labels, binary_classes)
    test_filter = np.isin(test_labels, binary_classes)

    # Extraindo dados das 2 classes, pegar as imagens que correspondem a 0s e 1s
    train_images, train_labels = train_images[train_filter], train_labels[train_filter]
    test_images, test_labels = test_images[test_filter], test_labels[test_filter]
    
    # Extrair o desvritor HOG para imagens de treino e teste
    train_hog_features = extract_hog_features(train_images)
    test_hog_features = extract_hog_features(test_images)

    # Normalizar as características HOG
    scaler = StandardScaler()
    train_hog_features = scaler.fit_transform(train_hog_features)
    test_hog_features = scaler.transform(test_hog_features)

    # Dividir os dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(train_hog_features, train_labels, test_size=0.2, random_state=42)

    # Construir o modelo para classificação binária
    model = models.Sequential([
        Input(shape=(train_hog_features.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Ajustar para uma saída binária (função sigmoide)
    ])

    diretorio_dos_arquivos_hog = 'outputs_hog_binario' # Diretório dos arquivos de saída
    initial_weights = model.get_weights() # Pesos iniciais da rede

    # Compilar o modelo
    model.compile(optimizer='adam',
                loss='binary_crossentropy',  # Ajustar para classificação binária
                metrics=['accuracy'])

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=5, 
                        validation_data=(X_val, y_val))

    # Avaliar o modelo
    test_loss, test_acc = model.evaluate(test_hog_features, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # Pegar dados para salvar em arquivos
    l_rate = model.optimizer.learning_rate.numpy()
    hiperparametros = {'taxa_de_aprendizado': float(l_rate), 'epochs': 5}
    post_train_weights = model.get_weights()
    historico_treinamento = history.history
    saidas_rede_neural = model.predict(test_hog_features)
    erros_treinamento = historico_treinamento['loss']
    predicted_labels = (saidas_rede_neural > 0.5).astype(int).flatten() # Por ser binária, a classificação escolhe a classe (neurônio) cuja probabilidade é maior que 50%
    
    save_outputs(diretorio_dos_arquivos_hog, hiperparametros, initial_weights, historico_treinamento, erros_treinamento, saidas_rede_neural, post_train_weights, predicted_labels, test_labels)

    # Calcular a matriz de confusão
    conf_matrix = confusion_matrix(test_labels, predicted_labels)

    # Calcular acurácia
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f'Acurácia: {accuracy:.2%}')

    # Visualizar a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão')
    plt.show()


if __name__ == "__main__":


    # Carregar o conjunto de dados MNIST
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Normalizar os dados
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # Adicionar um canal de cor
    print("Dados preparados!")

    print("Iniciando CNN MULTI-CLASSE com HOG")
    CNN_multi(train_images, train_labels, test_images, test_labels)    

    print("Iniciando CNN BINÁRIO com HOG")
    CNN_bin(train_images, train_labels, test_images, test_labels)