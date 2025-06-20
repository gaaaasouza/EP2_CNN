import os

# Configuração do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from menu_interface import MenuInterface

if __name__ == "__main__":
    menu = MenuInterface()
    menu.run()