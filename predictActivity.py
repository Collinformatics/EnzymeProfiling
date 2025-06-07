import esm
from functions import PredictActivity
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import sys
import torch
import time
from xgboost import XGBRegressor, XGBRFRegressor, DMatrix



# ===================================== User Inputs ======================================
inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
enzymeName = f'SARS-CoV-2 M{'ᵖʳᵒ'}'

inFixedResidue = ['Q']
inFixedPosition = [4]
inExcludeResidues = False
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inMinimumSubstrateCount = 10
inPrintNumber = 10



# ===================================== Misc Inputs ======================================
substrates = {
    'VVLQSAFE': 357412,
    'VALQAASA': 311191,
    'VILQAGNS': 309441,
    'LILQSAFK': 291445,
    'IVLQSAYH': 267741,
    'LVLQAAFT': 243158,
    'AVLQGAHN': 218612,
    'IALQSTGG': 204658,
}

substratesPred = ['AILQSGFE', 'VVLQASFA', 'IALQSGFE']



# ===================================== Set Options ======================================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.5f}'.format)

# Colors: Console
greyDark = '\033[38;2;144;144;144m'
purple = '\033[38;2;189;22;255m'
magenta = '\033[38;2;255;0;128m'
pink = '\033[38;2;255;0;242m'
cyan = '\033[38;2;22;255;212m'
blue = '\033[38;5;51m'
green = '\033[38;2;5;232;49m'
greenLight = '\033[38;2;204;255;188m'
greenLightB = '\033[38;2;204;255;188m'
greenDark = '\033[38;2;30;121;13m'
yellow = '\033[38;2;255;217;24m'
orange = '\033[38;2;247;151;31m'
red = '\033[91m'
resetColor = '\033[0m'

# Set device
print('============================== Set Training Device '
      '==============================')
if torch.cuda.is_available():
    device = 'cuda:0'
    print(f'Train with Device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {torch.cuda.get_device_name(device)}{resetColor}\n\n')
else:
    import platform
    device = 'cpu'
    print(f'Train with Device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {platform.processor()}{resetColor}\n\n')

# Define: Dataset tag
datasetTag = (f'{enzymeName} - {inFixedResidue[0]}@R{inFixedPosition[0]} - '
              f'FinalSort - MinCounts {inMinimumSubstrateCount}')



# =================================== Define Functions ===================================
class CNN:
    def __init__(self, substrates):
        self.substrates = substrates



class GradBoostingRegressor:
    def __init__(self, dfTrain, dfTest):
        print('========================== Gradient Boosting Regressor '
              '==========================')
        print(f'Module: {purple}SK Learn{resetColor}')

        # Record the Embeddings for the predicted substrates
        self.predSubEmbeddings = dfTest

        # Process dataframe
        x = dfTrain.drop(columns='activity').values
        y = np.log1p(dfTrain['activity'].values)

        # Train the model
        print(f'Training the model')
        start = time.time()
        self.model = GradientBoostingRegressor()
        self.model.fit(x, y)
        end = time.time()
        runtime = (end - start) * 1000
        print(f'      Training time: {red}{round(runtime, 3):,} ms{resetColor}\n')

        # Predict with the model
        print(f'Predicting Activity')
        start = time.time()
        # activityPred = self.model.predict(dfTest)
        # activityPred = np.expm1(activityPred)  # Reverse log1p transform
        end = time.time()
        runtime = (end - start) * 1000
        print(f'     Runtime: {red}{round(runtime, 3):,} ms{resetColor}\n')



class GradBoostingRegressorXGB:
    def __init__(self, dfTrain, dfTest):
        print('========================== Gradient Boosting Regressor '
              '==========================')
        print(f'Module: {purple}XGBoost{resetColor}')

        # Record the Embeddings for the predicted substrates
        self.predSubEmbeddings = dfTest

        # Process dataframe
        x = dfTrain.drop(columns='activity').values
        y = np.log1p(dfTrain['activity'].values)

        # Train the model
        print(f'Training the model')
        start = time.time()
        self.model = XGBRegressor(device=device)
        self.model.fit(x, y)
        end = time.time()
        runtime = (end - start) * 1000
        print(f'     Training time: {red}{round(runtime, 3):,} ms{resetColor}\n')


        # Predict with the model
        print(f'Predicting Activity')
        start = time.time()
        # activityPred = self.model.predict(dfTest)
        # activityPred = np.expm1(activityPred)  # Reverse log1p transform
        end = time.time()
        runtime = (end - start) * 1000
        print(f'     Runtime: {red}{round(runtime, 3):,} ms{resetColor}\n')



# =================================== Initialize Class ===================================
PredictActivity(enzymeName=enzymeName, datasetTag=datasetTag, labelsXAxis=inAAPositions,
                subsTrain=substrates, subsTest=substratesPred, printNumber=inPrintNumber)
