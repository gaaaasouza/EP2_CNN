import numpy as np
from typing import Tuple, List, Dict, Any
from keras import models, layers, Input
from sklearn.model_selection import train_test_split
from data_manager import DataManager
from file_manager import FileManager
from model_evaluator import ModelEvaluator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf # Importar tensorflow para EarlyStopping

class CNNHOGClassifier:
    """Classificador CNN com características HOG, configurável para otimização e uso final."""
    
    def __init__(self, data_manager: DataManager, file_manager: FileManager, evaluator: ModelEvaluator):
        self.data_manager = data_manager
        self.file_manager = file_manager
        self.evaluator = evaluator
        self.model = None
        self.history = None
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int,
                    dense_units_1: int = 128, dense_units_2: int = 64, dropout_rate: float = 0.5) -> models.Sequential: # Adicionado dropout_rate
        """Constrói modelo para características HOG com parâmetros configuráveis."""
        if num_classes == 1:
            activation = 'sigmoid'
        else:
            activation = 'softmax'
        
        self.model = models.Sequential([
            Input(shape=input_shape),
            layers.Dense(dense_units_1, activation='relu'),
            layers.Dense(dense_units_2, activation='relu'),
            layers.Dropout(dropout_rate), # Adicionado Dropout
            layers.Dense(num_classes, activation=activation)
        ])
        
        return self.model
    
    def compile_model(self, loss: str, metrics: List[str] = ['accuracy'], learning_rate: float = 0.001) -> None:
        """Compila o modelo com taxa de aprendizado configurável."""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray,
                   validation_data: Tuple[np.ndarray, np.ndarray], epochs: int = 5,
                   batch_size: int = 64, verbose: int = 1) -> None: # Adicionado batch_size e verbose
        """Treina o modelo com número de épocas e tamanho de batch configuráveis."""
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # Feature do CNN.py
        
        self.history = self.model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size, # Usando batch_size
            validation_data=validation_data,
            callbacks=[early_stopping], # Adicionado EarlyStopping
            verbose=verbose
        )
    
    def evaluate_model(self, test_data: np.ndarray, test_labels: np.ndarray, 
                      output_dir: str, is_binary: bool = False) -> None:
        """Avalia o modelo e salva resultados."""
        
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        accuracy_type = "Binary" if is_binary else "Multi-class"
        print(f'\n{accuracy_type} Test accuracy: {test_acc}')
        
        # Obter hiperparâmetros detalhados para salvamento
        hyperparams = {
            "model_type": "CNN HOG",
            "tipo_class": "binary" if is_binary else "multi",
            "arquitetura_cnn": {
                "camadas_dense": [
                    {"units": self.model.layers[0].units, "activation": "relu"},
                    {"units": self.model.layers[1].units, "activation": "relu"},
                ],
                "dropout_rate": self.model.layers[2].rate, # Taxa do Dropout
            },
            "hiperparametros_treino": {
                "optimizer": "Adam",
                "learning_rate": float(self.model.optimizer.learning_rate.numpy()),
                "loss": self.model.loss,
                "metrics": self.model.metrics_names,
                "batch_size": self.history.params.get('batch_size'), # Pega do history do fit
                "epochs": len(self.history.history['loss']), # Épocas reais treinadas
                "early_stopping_patience": 3 # Hardcoded como no CNN.py
            },
            "test_accuracy": test_acc
        }
        
        # Salvamento de pesos iniciais (capturado antes do treino, idealmente)
        initial_weights_layer = self.model.layers[0].get_weights()[0]
        final_weights_layer = self.model.layers[-1].get_weights()[0]

        training_history = self.history.history
        training_errors = training_history['loss']
        network_outputs = self.model.predict(test_data)
        
        title = f"Matriz de Confusão {'Binária' if is_binary else ''}"
        accuracy, predicted_labels = self.evaluator.evaluate_and_plot(
            self.model, test_data, test_labels, title
        )
        
        print(f'Acurácia{"" if not is_binary else " Binária"}: {accuracy:.2%}')
        
        self.file_manager.save_outputs(
            output_dir, hyperparams, initial_weights_layer, training_history,
            training_errors, network_outputs, final_weights_layer,
            predicted_labels, test_labels
        )
    
    def run_multiclass(self, params: Dict[str, Any] = None) -> None:
        """Executa classificação multiclasse com HOG com parâmetros opcionais."""
        print("Iniciando CNN MULTI-CLASSE com HOG")
        
        dense_units_1 = params.get('dense_units_1', 128) if params else 128
        dense_units_2 = params.get('dense_units_2', 64) if params else 64
        dropout_rate = params.get('dropout_rate', 0.5) if params else 0.5
        epochs = params.get('epochs', 5) if params else 5
        learning_rate = params.get('learning_rate', 0.001) if params else 0.001
        batch_size = params.get('batch_size', 64) if params else 64
        
        train_hog, test_hog = self.data_manager.prepare_hog_data(
            self.data_manager.train_images.squeeze(),
            self.data_manager.test_images.squeeze()
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            train_hog, self.data_manager.train_labels, test_size=0.2, random_state=42,
            stratify=self.data_manager.train_labels
        )
        
        self.build_model((train_hog.shape[1],), 10, dense_units_1=dense_units_1, dense_units_2=dense_units_2, dropout_rate=dropout_rate)
        self.compile_model('sparse_categorical_crossentropy', learning_rate=learning_rate)
        self.train_model(X_train, y_train, (X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
        
        self.evaluate_model(test_hog, self.data_manager.test_labels, 'outputs_hog')
    
    def run_binary(self, params: Dict[str, Any] = None) -> None:
        """Executa classificação binária com HOG com parâmetros opcionais."""
        print("Iniciando CNN BINÁRIO com HOG")
        
        dense_units_1 = params.get('dense_units_1', 128) if params else 128
        dense_units_2 = params.get('dense_units_2', 64) if params else 64
        dropout_rate = params.get('dropout_rate', 0.5) if params else 0.5
        epochs = params.get('epochs', 5) if params else 5
        learning_rate = params.get('learning_rate', 0.001) if params else 0.001
        batch_size = params.get('batch_size', 128) if params else 128
        
        train_images_bin, train_labels_bin, test_images_bin, test_labels_bin = \
            self.data_manager.filter_binary_classes()
        
        train_hog, test_hog = self.data_manager.prepare_hog_data(
            train_images_bin.squeeze(), test_images_bin.squeeze()
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            train_hog, train_labels_bin, test_size=0.2, random_state=42,
            stratify=train_labels_bin
        )
        
        self.build_model((train_hog.shape[1],), 1, dense_units_1=dense_units_1, dense_units_2=dense_units_2, dropout_rate=dropout_rate)
        self.compile_model('binary_crossentropy', learning_rate=learning_rate)
        self.train_model(X_train, y_train, (X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
        
        self.evaluate_model(test_hog, test_labels_bin, 'outputs_hog_binario', is_binary=True)

    # NOVO MÉTODO: Para ser usado pelo HyperparameterOptimizer
    def train_and_evaluate_for_optimization(self, params: Dict[str, Any], is_binary: bool = False) -> float:
        """
        Treina e avalia o modelo com parâmetros específicos para a otimização de hiperparâmetros.
        Retorna a melhor acurácia de validação.
        """
        dense_units_1 = params.get('dense_units_1', 128)
        dense_units_2 = params.get('dense_units_2', 64)
        dropout_rate = params.get('dropout_rate', 0.5)
        epochs = params.get('epochs', 5)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 64 if not is_binary else 128)

        num_classes = 1 if is_binary else 10
        loss = 'binary_crossentropy' if is_binary else 'sparse_categorical_crossentropy'
        
        # Preparar dados HOG (depende se é binário ou multiclasse)
        if is_binary:
            train_images_bin, train_labels_bin, _, _ = self.data_manager.filter_binary_classes()
            train_hog, _ = self.data_manager.prepare_hog_data(train_images_bin.squeeze(), np.array([])) # Teste vazio
            train_labels = train_labels_bin
        else:
            train_hog, _ = self.data_manager.prepare_hog_data(self.data_manager.train_images.squeeze(), np.array([])) # Teste vazio
            train_labels = self.data_manager.train_labels

        input_shape = (train_hog.shape[1],)

        # Dividir dados para validação (interno ao loop de otimização)
        X_train, X_val, y_train, y_val = train_test_split(
            train_hog, train_labels,
            test_size=0.2, random_state=42, stratify=train_labels
        )
        
        self.build_model(input_shape, num_classes, dense_units_1=dense_units_1, dense_units_2=dense_units_2, dropout_rate=dropout_rate)
        self.compile_model(loss, learning_rate=learning_rate)
        self.train_model(X_train, y_train, (X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0) # verbose=0
        
        return max(self.history.history['val_accuracy'])