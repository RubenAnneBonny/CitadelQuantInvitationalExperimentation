import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split

class Data_manager:
    def __init__(self,filename,testsize):
        self.filename=filename
        self.df=pd.read_csv(self.filename)

        self.X_train, self.X_test = train_test_split(
            self.df, test_size=testsize, random_state=42
        )

    def get_testdata(self):
        return self.X_test
    
    def get_traindata(self):
        return self.X_train
