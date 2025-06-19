import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras import datasets, layers, models, Input
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


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

    # Salvar dados captados durante o treinamento
    with open(os.path.join(dir, 'historico_treinamento.json'), 'w') as f:
        json.dump(train_hist, f)
    
    # Salvar o erro cometido pela rede neural em cada iteração do treinamento
    np.savetxt(os.path.join(dir, 'erros_treinamento.csv'), train_err, delimiter=',')

    # Salvar as saídas da rede neural para os dados de teste
    np.savetxt(os.path.join(dir, 'saidas_rede_neural.csv'), output_CNN, delimiter=',')

    # Salvar os pesos finais obtidos pela rede
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




# Método para criar CNN e fazer avaliação multiclasse
def CNN_multi(train_images, train_labels, test_images, test_labels):

    # Construir o modelo
    model = models.Sequential([
        Input(shape=(28, 28, 1)),  # Usar Input aqui
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), # Reduz pela metade
        layers.Flatten(), # Converter em um vetor unidimensional
        layers.Dense(64, activation='relu'),
        # A última camada densa deve ter um número de unidades igual ao número de classes
        layers.Dense(10, activation='softmax')
    ])

    diretorio_dos_arquivos_bruto = 'outputs_bruto' # Diretório para armazenar arquivos de saída
    init_weights = model.get_weights() # Obter todos os pesos do modelo

    # Compilar o modelo
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Treinar o modelo
    history = model.fit(train_images, train_labels, epochs=5, 
                        validation_data=(test_images, test_labels))

    # Avaliar o modelo
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    # Salvar dados em arquivos

    #   Hiperparâmetros:
    learning_rate = model.optimizer.learning_rate.numpy()
    hiperparametros = {'taxa_de_aprendizado': float(learning_rate), 'epochs': 5}

    #   Pesos finais:
    post_train_weights = model.get_weights()

    #   Dados referentes a cada época:
    historico_treinamento = history.history

    #   Gera as predições com base em probabilidades, ou seja, a saída de cada classe corresponde à probabilidade de a entrada pertencer a ela
    saidas_rede_neural = model.predict(test_images) 
    #   A decisão final da classe é feita com base no neurônio de maior valor de saída (escolhe a classe mais provável) 
    predicted_labels = np.argmax(saidas_rede_neural, axis=1)

    #   Erros de cada época:
    erros_treinamento = historico_treinamento['loss']

    save_outputs(diretorio_dos_arquivos_bruto, hiperparametros, init_weights, historico_treinamento, erros_treinamento, saidas_rede_neural, post_train_weights, predicted_labels, test_labels)

    # Calcular e plotar a matriz de confusão:
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



# Método para criar CNN e fazer avaliação binária
def CNN_bin(train_images, train_labels, test_images, test_labels):
    # Filtrar apenas as classes 0 e 1 para classificação binária
    binary_train_mask = (train_labels == 0) | (train_labels == 1)
    binary_test_mask = (test_labels == 0) | (test_labels == 1)
    # Extraindo dados das 2 classes, pegar as imagens que correspondem a 0s e 1s
    train_images_binary = train_images[binary_train_mask]
    train_labels_binary = train_labels[binary_train_mask]
    test_images_binary = test_images[binary_test_mask]
    test_labels_binary = test_labels[binary_test_mask]

    # Construir o modelo para classificação binária
    binary_model = models.Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Adicione mais camadas convolucionais conforme necessário
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 1 unidade para classificação binária
    ])

    init_weights_bin = binary_model.get_weights()
    # Compilar o modelo
    diretorio_dos_arquivos_bruto = 'outputs_bruto_binario'

    binary_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    # Treinar o modelo para classificação binária
    binary_history = binary_model.fit(train_images_binary, train_labels_binary, epochs=5,
                                    validation_data=(test_images_binary, test_labels_binary))

    # Avaliar o modelo binário
    binary_test_loss, binary_test_acc = binary_model.evaluate(test_images_binary, test_labels_binary, verbose=2)

    print(f'\nBinary Test accuracy: {binary_test_acc}\nBinary Test loss: {binary_test_loss}')

    # Salvar dados em aquivos
    learn_rate_bin = binary_model.optimizer.learning_rate.numpy()
    binary_hiperparametros = {'taxa_de_aprendizado': float(learn_rate_bin), 'epochs': 5}
    binary_post_train_weights = binary_model.get_weights()
    binary_historico_treinamento = binary_history.history
    erros_treinamento_bin = binary_historico_treinamento['loss']
    binary_saidas_rede_neural = binary_model.predict(test_images_binary)
    binary_predicted_labels = (binary_saidas_rede_neural > 0.5).astype(int).flatten()
    save_outputs(diretorio_dos_arquivos_bruto, binary_hiperparametros, init_weights_bin, binary_historico_treinamento, erros_treinamento_bin, binary_saidas_rede_neural, binary_post_train_weights, binary_predicted_labels, test_labels_binary)

    # Matriz confusão
    binary_conf_matrix = confusion_matrix(test_labels_binary, binary_predicted_labels)

    # Calcular acurácia binária
    binary_accuracy = accuracy_score(test_labels_binary, binary_predicted_labels)
    print(f'Acurácia Binária: {binary_accuracy:.2%}')

    # Visualizar a matriz de confusão binária
    plt.figure(figsize=(10, 8))
    sns.heatmap(binary_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão Binária')
    plt.show()


if __name__ == "__main__":

    # Carregar o conjunto de dados MNIST
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    print("Conjunto MNIST carregado!")

    # Preparando dados
    # Normalizar dados
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # Adicionar um canal de cor
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]
    print("Dados preparados!")

    print("Iniciando CNN MULTI-CLASSE")
    CNN_multi(train_images, train_labels, test_images, test_labels)    

    print("Iniciando CNN BINÁRIO")
    CNN_bin(train_images, train_labels, test_images, test_labels)