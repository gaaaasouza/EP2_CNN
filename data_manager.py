import numpy as np
import tensorflow as tf
from keras import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from typing import Tuple, List

class DataManager:
    """Classe responsável pelo gerenciamento e preparação dos dados"""
    
    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.scaler = StandardScaler()
    
    def load_mnist_data(self) -> None:
        """Carrega o dataset MNIST"""
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.mnist.load_data()
        print("Conjunto MNIST carregado!")
    
    def normalize_data(self) -> None:
        """Normaliza os dados de imagem"""
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
    
    def add_channel_dimension(self) -> None:
        """Adiciona dimensão de canal para CNNs"""
        self.train_images = self.train_images[..., tf.newaxis]
        self.test_images = self.test_images[..., tf.newaxis]
    
    def filter_binary_classes(self, classes: List[int] = [0, 1]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filtra dados para classificação binária"""
        train_filter = np.isin(self.train_labels, classes)
        test_filter = np.isin(self.test_labels, classes)
        
        return (self.train_images[train_filter], self.train_labels[train_filter],
                self.test_images[test_filter], self.test_labels[test_filter])
    
    def extract_hog_features(self, images: np.ndarray) -> np.ndarray:
        """Extrai características HOG das imagens"""
        print("Extraindo HOG...")
        hog_features = []
        for image in images:
            # Remove dimensão de canal se existir
            if len(image.shape) == 3:
                image = image.squeeze()
            features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            hog_features.append(features)
        return np.array(hog_features)
    
    def prepare_hog_data(self, train_images: np.ndarray, test_images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados HOG normalizados"""
        train_hog = self.extract_hog_features(train_images)
        test_hog = self.extract_hog_features(test_images)
        
        train_hog_normalized = self.scaler.fit_transform(train_hog)
        test_hog_normalized = self.scaler.transform(test_hog)
        
        return train_hog_normalized, test_hog_normalized