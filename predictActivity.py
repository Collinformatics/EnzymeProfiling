import cudf
import esm
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import sys
import torch
import time
from xgboost import XGBRegressor, XGBRFRegressor, DMatrix



inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
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



# =================================== Define Functions ===================================
class CNN:
    def __init__(self, substrates):
        self.substrates = substrates



class RandomForestRegressor:
    def __init__(self, dfTrain, dfTest, getSHAP=False):
        print('=========================== Random Forrest Regressor '
              '============================')
        print(f'Module: {purple}XGBoost{resetColor}')

        # Record the embeddings for the predicted substrates
        self.dTest = DMatrix(dfTest)

        # Process dataframe
        x = dfTrain.drop(columns='activity').values
        y = np.log1p(dfTrain['activity'].values)


        # Train the model
        start = time.time()
        self.model = XGBRFRegressor(device=device, tree_method="hist")
        self.model.fit(x, y)
        end = time.time()
        runtime = (end - start) * 1000
        print(f'Training time: {red}{round(runtime, 3):,} ms{resetColor}\n')


        if getSHAP:
            # Get: SHAP values
            booster = self.model.get_booster()
            booster.set_param({'device': device})
            shapValues = booster.predict(self.dTest, pred_contribs=True)
            shapInteractionValues = booster.predict(self.dTest, pred_interactions=True)
            print(f'Shap Values:\n{shapValues}\n\n'
                  f'Interaction Values:\n{shapInteractionValues}\n\n')


        # Predict with the model
        print(f'Predicting Activity')
        start = time.time()
        activityPred = self.model.predict(dfTest)
        # activityPred = inplace_predict(self.model, self.dTest)
        activityPred = np.expm1(activityPred)  # Reverse log1p transform
        end = time.time()
        runtime = (end - start) * 1000
        print(f'Predicted Activity:\n'
              f'{activityPred}')
        print(f'Testing time: {red}{round(runtime, 3):,} ms{resetColor}\n')

        sys.exit()


class GradBoostingRegressor:
    def __init__(self, dfTrain, dfTest):
        print('========================== Gradient Boosting Regressor '
              '==========================')
        print(f'Module: {purple}SK Learn{resetColor}')

        # Record the embeddings for the predicted substrates
        self.predSubEmbeddings = dfTest

        # Process dataframe
        x = dfTrain.drop(columns='activity').values
        y = np.log1p(dfTrain['activity'].values)

        # Train the model
        start = time.time()
        self.model = GradientBoostingRegressor()
        self.model.fit(x, y)
        end = time.time()
        runtime = (end - start) * 1000
        print(f'Training time: {red}{round(runtime, 3):,} ms{resetColor}\n')

        self.predActivity()

    def predActivity(self):
        print('Here\n\n')



class GradBoostingRegressorXGB:
    def __init__(self, dfTrain, dfTest):
        print('========================== Gradient Boosting Regressor '
              '==========================')
        print(f'Module: {purple}XGBoost{resetColor}')

        # Record the embeddings for the predicted substrates
        self.predSubEmbeddings = dfTest

        # Process dataframe
        x = dfTrain.drop(columns='activity').values
        y = np.log1p(dfTrain['activity'].values)

        # Train the model
        start = time.time()
        self.model = XGBRegressor(device=device)
        self.model.fit(x, y)
        end = time.time()
        runtime = (end - start) * 1000
        print(f'Training time: {red}{round(runtime, 3):,} ms{resetColor}\n')

        self.predActivity()

    def predActivity(self):
        print('Here\n\n')




def ESM(substrates, subLabel, datasetType):
    print('=========================== Convert To Numerical: ESM '
          '===========================')
    print(f'Dataset: {purple}{datasetType}{resetColor}\n'
          f'Total unique substrates: {red}{len(substrates):,}{resetColor}')

    # Step 1: Convert substrates to ESM model format and generate embeddings
    totalSubActivity = 0
    subs = []
    values = []
    if type(substrates) is dict:
        for index, (substrate, value) in enumerate(substrates.items()):
            totalSubActivity += value
            subs.append((f'Sub{index}', substrate))
            values.append(value)
    else:
        for index, substrate in enumerate(substrates):
            subs.append((f'Sub{index}', substrate))
    sampleSize = len(substrates)
    print(f'Collected substrates:{red} {sampleSize:,}{resetColor}')
    if totalSubActivity != 0:
        if isinstance(totalSubActivity, float):
              print(f'Total Values:{red} {round(totalSubActivity, 1):,}'
                    f'{resetColor}')
        else:
            print(f'Total Values:{red} {totalSubActivity:,}{resetColor}')
    print()



    # Step 2: Load the ESM model and batch converter
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    numLayersESM = 36
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # numLayersESM = 33
    # esm2_t36_3B_UR50D has 36 layers
    # esm2_t33_650M_UR50D has 33 layers
    # esm2_t12_35M_UR50D has 12 layers
    model = model.to(device)

    # Get batch tensor
    batchConverter = alphabet.get_batch_converter()


    # Step 3: Convert substrates to ESM model format and generate embeddings
    try:
        batchLabels, batchSubs, batchTokens = batchConverter(subs)
        batchTokensCPU = batchTokens
        batchTokens = batchTokens.to(device)

    except Exception as exc:
        print(f'{orange}ERROR: The ESM has failed to evaluate your substrates\n\n'
              f'Exception:\n{exc}\n\n'
              f'Suggestion:'
              f'     Try replacing: {cyan}esm.pretrained.esm2_t36_3B_UR50D()'
              f'{orange}\n'
              f'     With: {cyan}esm.pretrained.esm2_t33_650M_UR50D()'
              f'{resetColor}\n')
        sys.exit(1)
    print(f'Batch Tokens:{greenLight} {batchTokens.shape}{resetColor}\n'
          f'{greenLight}{batchTokens}{resetColor}\n')

    # Record tokens
    slicedTokens = pd.DataFrame(batchTokensCPU[:, 1:-1],
                                index=batchSubs,
                                columns=subLabel)
    if totalSubActivity != 0:
        slicedTokens['Values'] = values
    print(f'\nSliced Tokens:\n'
          f'{greenLight}{slicedTokens}{resetColor}\n')

    with torch.no_grad():
        results = model(batchTokens, repr_layers=[numLayersESM], return_contacts=False)

    # Step 4: Extract per-sequence embeddings
    tokenReps = results["representations"][numLayersESM]  # (N, seq_len, hidden_dim)
    sequenceEmbeddings = tokenReps[:, 0, :]  # [CLS] token embedding: (N, hidden_dim)
    print(f'Extracted embeddings shape: '
          f'{pink}{sequenceEmbeddings.shape}{resetColor}\n\n')

    # Convert to numpy and store substrate activity proxy
    embeddings = sequenceEmbeddings.cpu().numpy()
    if type(substrates) is dict:
        values = np.array(values).reshape(-1, 1)
        data = np.hstack([embeddings, values])
        columns = [f'feat_{i}' for i in range(embeddings.shape[1])] + ['activity']
    else:
        data = np.hstack([embeddings])
        columns = [f'feat_{i}' for i in range(embeddings.shape[1])]
    subEmbeddings = pd.DataFrame(data, index=batchSubs, columns=columns)

    return slicedTokens, batchSubs, sampleSize, subEmbeddings



# ===================================== Run The Code =====================================
subsTrain = ESM(substrates=substrates, subLabel=inAAPositions, datasetType='Template')
subsPred = ESM(substrates=substratesPred, subLabel=inAAPositions,
               datasetType='Predictions')



# ================================== Initialize Classes ==================================
# cnn = CNN(substrates=substrates)
RandomForestRegressor(dfTrain=subsTrain[3], dfTest=subsPred[3])
GradBoostingRegressor(dfTrain=subsTrain[3], dfTest=subsPred[3])
GradBoostingRegressorXGB(dfTrain=subsTrain[3], dfTest=subsPred[3])
