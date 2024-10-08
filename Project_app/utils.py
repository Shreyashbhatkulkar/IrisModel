import pickle
import json 
import numpy as np
import config

class IrisDataset():
    def __init__(self,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
        self.SepalLengthCm = SepalLengthCm
        self.SepalWidthCm = SepalWidthCm
        self.PetalLengthCm = PetalLengthCm
        self.PetalWidthCm = PetalWidthCm
        
    def load_model(self):
        with open(config.MODEL_FILE_PATH,'rb') as f:
            self.model = pickle.load(f)
            
        with open(config.JSON_FILE_PATH,'r') as f:
            self.json_data = json.load(f)

    def predict_species(self):
        self.load_model()
        
        test_array = np.zeros(len(self.json_data['columns']))
        test_array[0] = self.SepalLengthCm
        test_array[1] = self.SepalWidthCm
        test_array[2] = self.PetalLengthCm
        test_array[3] = self.PetalWidthCm
        
        print('TEST ARRAY',test_array) #4columns
        predict_species = self.model.predict([test_array])
        return predict_species

if __name__ == '__main__':
    
    SepalLengthCm = 4.4
    SepalWidthCm = 2.9
    PetalLengthCm = 1.4
    PetalWidthCm = 0.2
    
    iris = IrisDataset(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    iris.predict_species()
