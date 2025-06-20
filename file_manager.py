import os
import json
import numpy as np
from typing import Dict, Any, List

class FileManager:
    """Classe responsável pelo salvamento de arquivos"""
    
    @staticmethod
    def save_outputs(directory: str, hyperparams: Dict[str, Any], initial_weights: List[np.ndarray],
                    training_history: Dict[str, List], training_errors: List[float],
                    network_outputs: np.ndarray, final_weights: List[np.ndarray],
                    predicted_classes: np.ndarray, real_classes: np.ndarray) -> None:
        """Salva todos os outputs do modelo"""
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Salvar hiperparâmetros
        with open(os.path.join(directory, 'hiperparametros.json'), 'w') as f:
            json.dump(hyperparams, f)
        
        # Salvar pesos iniciais
        FileManager._save_weights(directory, 'pesos_iniciais.txt', initial_weights)
        
        # Salvar histórico de treinamento
        with open(os.path.join(directory, 'historico_treinamento.json'), 'w') as f:
            json.dump(training_history, f)
        
        # Salvar erros de treinamento
        np.savetxt(os.path.join(directory, 'erros_treinamento.csv'), training_errors, delimiter=',')
        
        # Salvar saídas da rede
        np.savetxt(os.path.join(directory, 'saidas_rede_neural.csv'), network_outputs, delimiter=',')
        
        # Salvar pesos finais
        FileManager._save_weights(directory, 'pesos_finais.txt', final_weights)
        
        # Salvar comparação de classes
        with open(os.path.join(directory, 'classes_previstas.txt'), 'w') as f:
            for pred, real in zip(predicted_classes, real_classes):
                f.write(f"Previu: {pred}\tReal: {real}\n")
    
    @staticmethod
    def _save_weights(directory: str, filename: str, weights: List[np.ndarray]) -> None:
        """Salva pesos em arquivo texto"""
        with open(os.path.join(directory, filename), 'w') as f:
            for i, weight_matrix in enumerate(weights):
                f.write(f"\nPesos da Camada {i+1}:\n")
                flattened_weights = weight_matrix.flatten()
                for weight in flattened_weights:
                    f.write(f"{weight}\n")