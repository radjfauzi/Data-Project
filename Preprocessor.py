import os
import csv
import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from IPython.display import clear_output

class Preprocessor:

    # Case folding lowercase
    def convert_lower_case(self, data):
        return np.char.lower(data)

    # Case folding 
    def remove_punctuation(self, data):
        symbols = "!\"#$%&()*+-./:;<=>?@\n[\]^_,`{|}“”~'"
        for i in range(len(symbols)):
            data = np.char.replace(data, symbols[i], ' ')
            data = np.char.replace(data, "  ", " ")
        data = np.char.replace(data, ',', '') # questionable
        return data
        
    # Case folding 
    def remove_apostrophe(self, data):
        data = np.char.replace(data, "'", "")
        data = np.char.replace(data, "‘", "") # Different ASCII
        return data

    # Cleaning some data 
    def replace_anomalies(self, data):
        csv_name = 'normalisasi.csv'
        nm = pd.read_csv(csv_name)
        for i in range(len(nm)):
            data = np.char.replace(data, nm['before'][i], nm['after'][i])
        return data

    # Stemming
    def stem(self, data):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer.stem(str(data))

    def remove_stopword(self, data):
        factory = StopWordRemoverFactory()
        stopword_remover = factory.create_stop_word_remover()
        return stopword_remover.remove(str(data))

        # preprocess method
    def preprocess(self, data):
        data = self.convert_lower_case(data)
        data = self.remove_punctuation(data)
        data = self.remove_apostrophe(data)
        data = self.replace_anomalies(data)
        data = self.remove_stopword(data)
        data = self.stem(data) # optional, check if needed

        return data