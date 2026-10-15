import numpy as np
import math as mt
import os


class finstat:
    def __init__(self):
        current_directory = os.path.dirname(__file__)
        file_path = os.path.join(current_directory, 'data.csv')
        print(current_directory)