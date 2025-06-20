from data_manager import DataManager
from file_manager import FileManager
from model_evaluator import ModelEvaluator
from cnn_raw_classifier import CNNRawClassifier
from cnn_hog_classifier import CNNHOGClassifier

class ExperimentRunner:
    """Classe principal para executar experimentos"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.file_manager = FileManager()
        self.evaluator = ModelEvaluator()
    
    def run_all_experiments(self) -> None:
        """Executa todos os experimentos"""
        # Preparar dados
        self.data_manager.load_mnist_data()
        self.data_manager.normalize_data()
        self.data_manager.add_channel_dimension()
        print("Dados preparados!")
        
        # Executar experimentos CNN bruta
        cnn_raw = CNNRawClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_raw.run_multiclass()
        cnn_raw.run_binary()
        
        # Executar experimentos CNN HOG
        cnn_hog = CNNHOGClassifier(self.data_manager, self.file_manager, self.evaluator)
        cnn_hog.run_multiclass()
        cnn_hog.run_binary()