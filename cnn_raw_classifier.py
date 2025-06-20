import numpy as np
from typing import Tuple
from keras import models, layers, Input
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_manager import DataManager
from file_manager import FileManager
from model_evaluator import ModelEvaluator

class CNNRawClassifier:
    """Classificador CNN com dados brutos"""
    
    def __init__(self, data_manager: DataManager, file_manager: FileManager, evaluator: ModelEvaluator):
        self.data_manager = data_manager
        self.file_manager = file_manager
        self.evaluator = evaluator
        self.model = None
        self.history = None
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> models.Sequential:
        """Constrói modelo CNN para dados brutos"""
        if num_classes == 1:  # Classificação binária
            activation = 'sigmoid'
        else:  # Classificação multiclasse
            activation = 'softmax'
        
        self.model = models.Sequential([
            Input(shape=input_shape),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation=activation)
        ])
        
        return self.model
    
    def compile_model(self, loss: str, metrics=['accuracy']) -> None:
        """Compila o modelo"""
        self.model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray,
                   validation_data: Tuple[np.ndarray, np.ndarray], epochs: int = 3) -> None:
        """Treina o modelo"""
        self.history = self.model.fit(train_data, train_labels, epochs=epochs,
                                    validation_data=validation_data)
    
    def evaluate_model(self, test_data: np.ndarray, test_labels: np.ndarray, 
                      output_dir: str, is_binary: bool = False) -> None:
        """Avalia o modelo e salva resultados"""
        
        # Avaliar modelo
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        accuracy_type = "Binary" if is_binary else "Multi-class"
        print(f'\n{accuracy_type} Test accuracy: {test_acc}')
        
        # Obter dados para salvamento
        initial_weights = self.model.get_weights()
        hyperparams = {
            'taxa_de_aprendizado': float(self.model.optimizer.learning_rate.numpy()),
            'epochs': len(self.history.history['loss'])
        }
        final_weights = self.model.get_weights()
        training_history = self.history.history
        training_errors = training_history['loss']
        network_outputs = self.model.predict(test_data)
        
        # Avaliar e plotar
        title = f"Matriz de Confusão {'Binária' if is_binary else ''}"
        accuracy, predicted_labels = self.evaluator.evaluate_and_plot(
            self.model, test_data, test_labels, title
        )
        
        print(f'Acurácia{"" if not is_binary else " Binária"}: {accuracy:.2%}')
        
        # Salvar resultados
        self.file_manager.save_outputs(
            output_dir, hyperparams, initial_weights, training_history,
            training_errors, network_outputs, final_weights,
            predicted_labels, test_labels
        )
    
    def run_multiclass(self) -> None:
        """Executa classificação multiclasse"""
        print("Iniciando CNN MULTI-CLASSE")
        
        # Construir modelo
        self.build_model((28, 28, 1), 10)
        self.compile_model('sparse_categorical_crossentropy')
        
        # Treinar modelo
        self.train_model(
            self.data_manager.train_images, self.data_manager.train_labels,
            (self.data_manager.test_images, self.data_manager.test_labels)
        )
        
        # Avaliar modelo
        self.evaluate_model(self.data_manager.test_images, self.data_manager.test_labels, 'outputs_bruto')
    
    def run_binary(self) -> None:
        """Executa classificação binária"""
        print("Iniciando CNN BINÁRIO")
        
        # Filtrar dados para classificação binária
        train_images_bin, train_labels_bin, test_images_bin, test_labels_bin = \
            self.data_manager.filter_binary_classes()
        
        # Construir modelo
        self.build_model((28, 28, 1), 1)
        self.compile_model('binary_crossentropy')
        
        # Treinar modelo
        self.train_model(train_images_bin, train_labels_bin, (test_images_bin, test_labels_bin))
        
        # Avaliar modelo
        self.evaluate_model(test_images_bin, test_labels_bin, 'outputs_bruto_binario', is_binary=True)