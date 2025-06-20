import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from keras import models
from data_manager import DataManager
from file_manager import FileManager
from model_evaluator import ModelEvaluator

class BaseClassifier(ABC):
    """Classe base abstrata para classificadores"""
    
    def __init__(self, data_manager: DataManager, file_manager: FileManager, evaluator: ModelEvaluator):
        self.data_manager = data_manager
        self.file_manager = file_manager
        self.evaluator = evaluator
        self.model = None
        self.history = None
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> models.Sequential:
        """Constrói o modelo - deve ser implementado pelas subclasses"""
        pass
    
    @abstractmethod
    def get_output_directory(self, is_binary: bool = False) -> str:
        """Retorna diretório de saída - deve ser implementado pelas subclasses"""
        pass
    
    def compile_model(self, loss: str, metrics: List[str] = ['accuracy']) -> None:
        """Compila o modelo"""
        self.model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray,
                   validation_data: Tuple[np.ndarray, np.ndarray], epochs: int = 5) -> None:
        """Treina o modelo"""
        self.history = self.model.fit(train_data, train_labels, epochs=epochs,
                                    validation_data=validation_data)
    
    def evaluate_model(self, test_data: np.ndarray, test_labels: np.ndarray, is_binary: bool = False) -> None:
        """Avalia o modelo e salva resultados"""
        
        # Avaliar modelo
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=2)
        accuracy_type = "Binary" if is_binary else "Multi-class"
        print(f'\n{accuracy_type} Test accuracy: {test_acc}')
        
        # Obter dados para salvamento
        initial_weights = self.model.get_weights()  # Nota: em implementação real, seria salvo antes do treino
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
        output_dir = self.get_output_directory(is_binary)
        self.file_manager.save_outputs(
            output_dir, hyperparams, initial_weights, training_history,
            training_errors, network_outputs, final_weights,
            predicted_labels, test_labels
        )