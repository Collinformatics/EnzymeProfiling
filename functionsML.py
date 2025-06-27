# PURPOSE: This script contains the functions that you will need to process your NGS data



import esm
from itertools import combinations, product
import joblib
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import os
import pandas as pd
import platform
import random
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import time
import torch
from xgboost import XGBRegressor



# ================================== Setup Residue List ==================================
defaultResidues = (('Alanine', 'Ala', 'A'), ('Arginine', 'Arg', 'R'),
                   ('Asparagine', 'Asn', 'N'), ('Aspartic Acid', 'Asp', 'D'),
                   ('Cysteine', 'Cys', 'C'),  ('Glutamic Acid', 'Glu', 'E'),
                   ('Glutamine', 'Gln', 'Q'),('Glycine', 'Gly', 'G'),
                   ('Histidine', 'His ', 'H'),('Isoleucine', 'Ile', 'I'),
                   ('Leucine', 'Leu', 'L'), ('Lysine', 'Lys', 'K'),
                   ('Methionine', 'Met', 'M'), ('Phenylalanine', 'Phe', 'F'),
                   ('Proline', 'Pro', 'P'), ('Serine', 'Ser', 'S'),
                   ('Threonine', 'Thr', 'T'),('Tryptophan', 'Typ', 'W'),
                   ('Tyrosine', 'Tyr', 'Y'), ('Valine', 'Val', 'V'))



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



# =================================== Define Functions ===================================
def pressKey(event):
    if event.key == 'escape':
        plt.close()
    elif event.key == 'e':
        sys.exit()
    elif event.key == 'r':
        python = sys.executable # Doesnt seem to work on windows?
        os.execl(python, python, *sys.argv)



# ================================= Define ML Algorithms =================================
class RandomForestRegressor:
    def __init__(self, dfTrain, dfPred, subsPredChosen, minES, pathModel, modelTag,
                 modelTagHigh, testSize, device, printNumber):
        print('============================ Random Forest Regressor '
              '============================')
        print(f'Module: {purple}Scikit-Learn{resetColor}\n'
              f'Model: {purple}{modelTag}{resetColor}\n')

        self.device = device
        self.paramGrid = {
            'n_estimators': [100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5],
            'min_samples_leaf': [2, 4]
        }
        # self.paramGrid = {
        #     'learning_rate': [0.01],
        #     'n_estimators': [100],
        #     'subsample': [0.5]
        # }
        self.predictions = {}
        subsPred = list(dfPred.index)

        # Parameters: High value dataset
        self.cutoff = 0.8  # 0.8 = Top 20
        self.modelTag = 'All Substrates'
        self.modelTagHigh = f'Top {int(round((100 * (1 - self.cutoff)), 0))} Substrates'
        pathModelH = pathModel.replace('Test Size',
                                       f'High Values - Test Size {self.testSize}')


        # # Get Model: Random Forest Regressor
        if os.path.exists(pathModel) and os.path.exists(pathModelH):
            self.model = self.loadModel(pathModel=pathModel, tag=self.modelTag)
            self.modelH = self.loadModel(pathModel=pathModelH, tag=self.modelTagHigh)
        else:
            # Process dataframe
            x = dfTrain.drop(columns='activity').values
            y = np.log1p(dfTrain['activity'].values)

            # Get High-value subset
            threshold = dfTrain['activity'].quantile(self.cutoff)
            dfHigh = dfTrain[dfTrain['activity'] > threshold]
            xHigh = dfHigh.drop(columns='activity').values
            yHigh = np.log1p(dfHigh['activity'].values)


            def trainModel(model, x, y, tag, path, saveModel=True):
                xTrain, xTest, yTrain, yTest = train_test_split(
                    x, y, test_size=testSize, random_state=19)

                modelCV = GridSearchCV(estimator=model, param_grid=self.paramGrid, cv=3,
                                       scoring='neg_mean_squared_error', n_jobs=-1)

                # Train the model
                print(f'Training Model: {pink}{tag}{resetColor}\n'
                      f'Splitting Training Set: {blue}{round((1 - testSize) * 100, 0)}'
                      f'{pink}:{blue}{round((testSize) * 100, 0)}{resetColor}\n'
                      f'     Train: {blue}{xTrain.shape}{resetColor}\n'
                      f'     Test: {blue}{xTest.shape}{resetColor}\n')
                start = time.time()
                # model.fit(xTrain, yTrain)
                modelCV.fit(xTrain, yTrain)
                end = time.time()
                runtime = (end - start) / 60
                print(f'Training Time: {red}'
                      f'{round(runtime, self.roundVal):,} min{resetColor}\n')
                print(f'Best Hyperparameters: '
                      f'{greenLight}{modelCV.best_params_}{resetColor}')

                # Evaluate the model
                print(f'Evaluate Model Accuracy:')
                # yPred = model.predict(xTest)
                yPred = modelCV.predict(xTest)
                accuracy = pd.DataFrame({
                    'yPred_log': yPred,
                    'yTest_log': yTest,
                })
                yPred = np.expm1(yPred)  # Reverse log1p transform
                yTest = np.expm1(yTest)
                accuracy.loc[:, 'yPred'] = yPred
                accuracy.loc[:, 'yTest'] = yTest
                MAE = mean_absolute_error(yPred, yTest)
                MSE = mean_squared_error(yPred, yTest)
                R2 = r2_score(yPred, yTest)
                print(f'{greenLight}{accuracy}{resetColor}\n\n')
                print(f'Prediction Accuracy: {purple}{tag}{resetColor}\n'
                      f'     MAE: {red}{round(MAE, self.roundVal)}{resetColor}\n'
                      f'     MSE: {red}{round(MSE, self.roundVal)}{resetColor}\n'
                      f'     R2: {red}{round(R2, self.roundVal)}{resetColor}\n')
                if saveModel and not os.path.exists(path):
                    print(f'Saving Trained ESM Model:\n'
                          f'     {greenDark}{path}{resetColor}\n')
                    joblib.dump(modelCV.best_estimator_, path)
                print()

                return modelCV.best_estimator_

            # Train Models
            # self.model = RandomForestRegressor(n_estimators=self.NTrees,
            #                                    random_state=self.randomState)
            self.model = trainModel(model=RandomForestRegressor(), x=x, y=y,
                                    tag=self.modelTag, path=pathModel, saveModel=True)
            self.modelH = trainModel(model=RandomForestRegressor(), x=xHigh, y=yHigh,
                                     tag=self.modelTagHigh, path=pathModelH, saveModel=True)


        def makePredictions(model, tag):
            # Predict substrate activity
            print(f'Predicting Substrate Activity: {purple}{tag}{resetColor}\n'
                  f'     Total Substrates: {red}{len(dfPred.index):,}{resetColor}')
            start = time.time()
            activityPred = model.predict(dfPred.values)
            activityPred = np.expm1(activityPred)  # Reverse log1p transform
            end = time.time()
            runtime = (end - start) * 1000
            print(f'     Runtime: {red}{round(runtime, self.roundVal):,} ms{resetColor}\n')

            # Rank predictions
            activityPredRandom = {
                substrate: float(score)
                for substrate, score in zip(subsPred, activityPred)
            }
            activityPredRandom = dict(sorted(
                activityPredRandom.items(), key=lambda x: x[1], reverse=True))
            self.predictions['Random'] = activityPredRandom
            print(f'Predicted Activity: {purple}Random Substrates - Min ES: {minES}'
                  f'{resetColor}')
            for iteration, (substrate, value) in enumerate(activityPredRandom.items()):
                if iteration >= printNumber:
                    break
                print(f'     {substrate}: {red}{round(value, self.roundVal):,}{resetColor}')
            print('\n')

            # Get: Chosen substrate predictions
            if subsPredChosen != {}:
                print(f'Predicted Activity: {purple}Chosen Substrates{resetColor}')
                for key, substrates in subsPredChosen.items():
                    activityPredChosen = {}
                    for substrate in substrates:
                        activityPredChosen[substrate] = (
                            activityPredRandom)[substrate]
                    activityPredChosen = dict(sorted(activityPredChosen.items(),
                                                     key=lambda x: x[1], reverse=True))
                    self.predictions[key] = activityPredChosen
                    print(f'Substrate Set: {purple}{key}{resetColor}')
                    for iteration, (substrate, activity) in (
                            enumerate(activityPredChosen.items())):
                        activity = float(activity)
                        print(f'     {pink}{substrate}{resetColor}: '
                              f'{red}{round(activity, self.roundVal):,}{resetColor}')
                    print('\n')

        makePredictions(model=self.modelH, tag=self.modelTagHigh)
        makePredictions(model=self.modelM, tag=self.modelTagMid)
        makePredictions(model=self.modelL, tag=self.modelTagLow)
        print(f'Find a way to record multiple predictions.')


    def loadModel(self, pathModel, tag):
        print(f'Loading Trained ESM Model: {purple}{tag}{resetColor}\n'
              f'     {greenDark}{pathModel}{resetColor}\n')

        return joblib.load(pathModel)



"""
Params Grid: Random Forest Regressor - XGB
    max_leaves:
        You can use it instead of max_depth if you care more about model
        complexity in terms of decision regions than tree height.

    max_depth:
        Limits the depth of individual trees.
        Helps prevent overfitting; lower values create simpler trees.
        Easier to interpret than max_leaves.

    learning_rate (aka eta):
        Controls how much each tree contributes to the overall model.
        Lower values improve generalization but require more boosting rounds.
        Often paired with early stopping.

    n_estimators:
        Number of boosting rounds (trees).
        More trees improve performance but increase computation time and risk
        of overfitting.
        Use early stopping to avoid unnecessary rounds.

    subsample:
        Fraction of training data randomly sampled for each tree.
        Helps prevent overfitting; values between 0.5 and 0.9 are common.
        Lower values add randomness and improve generalization.

    colsample_bytree:
        Fraction of features randomly sampled for each tree.
        Like subsample, adds diversity and reduces overfitting.
        Especially helpful for high-dimensional data.

    colsample_bylevel:
        Fraction of features randomly sampled for each tree level (depth).
        More granular version of colsample_bytree; rarely used alone but can
        improve regularization.
        improve regularization.

    colsample_bynode:
        Fraction of features used per tree split.
        Very fine-grained feature sampling—can work well in large feature spaces.

    gamma (aka min_split_loss):
        Minimum loss reduction required to make a split.
        Higher values make trees more conservative (fewer splits).
        Great for reducing overfitting.

    reg_alpha:
        L1 regularization term on weights (like Lasso).
        Encourages sparsity in leaf weights—can help with feature selection.

    reg_lambda:
        L2 regularization term on weights (like Ridge).
        Helps with multicollinearity and prevents weights from growing too large.

    min_child_weight:
        Minimum sum of instance weights (hessian) needed in a child.
        Larger values prevent splitting nodes with few samples—controls
        overfitting.

    tree_method:
        Algorithm used to train trees.
        'auto', 'exact', 'approx', 'hist', or 'gpu_hist'.
        'hist' and 'gpu_hist' are faster and scalable to large datasets.

    predictor:
        Device prediction strategy.
        'auto', 'cpu_predictor', or 'gpu_predictor'.
        'gpu_predictor' is faster for inference on compatible hardware.

    grow_policy:
        Controls how trees grow.
        'depthwise' grows tree level by level, 'lossguide' grows by highest loss
        reduction.
        lossguide' can lead to deeper trees with fewer leaves—good with large
        datasets.

    importance_type:
        Method for computing feature importances: 'weight', 'gain', 'cover', etc.
        Doesn't affect training but useful for model interpretation.
"""
class RandomForestRegressorXGB:
    def __init__(self, parentObj, pathModel, modelTag, trainModel=True,
                 evalMetric='rmse'):
        """
        :param evalMetric: options include 'rmse' and 'mae'
        """

        print('================================== Train Model '
              '==================================')
        print(f'ML Algorithm: {purple}Random Forest Regressor{resetColor}\n'
              f'Module: {purple}XGBoost{resetColor}\n'
              f'Model: {purple}{modelTag}{resetColor}\n')
        dfTrain = parentObj.embeddingsSubsTrain
        dfPred = parentObj.embeddingsSubsPred
        modelAccuracyPaths = parentObj.pathModelAccuracy

        self.device = parentObj.device
        self.tagExperiment = parentObj.tagExperiment
        self.maxValue = parentObj.maxTrainingScore
        self.selectSubsTopPercent = parentObj.subsPercentSelectTop
        self.selectSubsBottomPercent = parentObj.subsPercentSelectBottom
        self.subsPred = list(dfPred.index)
        self.predictions = {}
        self.layersESM = parentObj.layersESM
        self.layerESMTag = parentObj.layersESMTag
        print(f'Tag: {purple}{self.layerESMTag}{resetColor}')
        self.modelAccuracy = parentObj.modelAccuracy
        self.predictionAccuracy = {}
        self.modelBestPredictions = {}
        self.sortedColumns = []
        self.datasetTagHigh = parentObj.subsetTagHigh
        self.datasetTagMid = parentObj.subsetTagMid
        self.datasetTagLow =  parentObj.subsetTagLow
        self.bestParams = {self.datasetTagHigh: {},
                           self.datasetTagMid: {},
                           self.datasetTagLow: {}}

        # Parameters: Figures
        self.pathFigures = parentObj.pathFigures
        self.pathFiguresTraining = parentObj.pathFiguresTraining
        self.figSize = parentObj.figSize
        self.residueLabelType = 2 # 0 = full AA name, 1 = 3-letter code, 2 = 1 letter
        self.labelSizeTitle = 18
        self.labelSizeAxis = 16
        self.labelSizeTicks = 13
        self.lineThickness = 1.5
        self.tickLength = 4
        self.figureResolution = 300

        # Parameters: Misc
        self.roundVal = 3
        self.printMaxColumns = 8
        self.printMaxRows = 10

        # Params: Grid search
        self.paramGrid = {
            'colsample_bytree': np.arange(0.6, 1.0, 0.2),
            'learning_rate': [0.2, 0.15, 0.1, 0.05, 0.01],
            'max_leaves': range(10, 310, 50),
            'min_child_weight': [1, 3, 5, 10],
            'n_estimators': [50, 100, 150, 200, 500, 1000, 2000, 3000],
            'subsample': np.arange(0.6, 1.2, 0.2)
        }
        # 'max_leaves': range(2, 10, 1) # N terminal nodes
        # 'max_depth': range(2, 6, 1)


        # ================================================================================
        # ================================================================================
        # ================================================================================
        # Parameters: Models
        self.trainOnlyTopSubs = True
        self.models = {}
        if self.trainOnlyTopSubs:
            self.models[self.datasetTagHigh] = None
        else:
            self.models[self.datasetTagHigh] = None
            self.models[self.datasetTagMid] = None
            self.models[self.datasetTagLow] = None
        # ================================================================================
        # ================================================================================
        # ================================================================================

        # Parameters: Saving the model
        pathModelH = pathModel.replace(
            'Embeddings', f'{self.datasetTagHigh} - Embeddings')
        pathModelH = pathModelH.replace(' %', '')
        # pathModelH = pathModelH.replace('Activity Scores: ', '')
        pathModelM = pathModel.replace(
            'Embeddings', f'{self.datasetTagMid} - Embeddings')
        pathModelM = pathModelM.replace(' %', '')
        pathModel = pathModel.replace(
            'Embeddings', f'{self.datasetTagLow} - Embeddings')
        pathModel = pathModel.replace(' %', '')
        modelPaths = {
            self.datasetTagHigh: pathModelH,
            self.datasetTagMid: pathModelM,
            self.datasetTagLow: pathModel
        }

        # Process dataframe
        x = dfTrain.drop(columns='activity').values
        y = np.log1p(dfTrain['activity'].values)

        # Parameters: Slitting the dataset
        self.activityQuantile1 = (100 - self.selectSubsTopPercent) / 100
        self.activityQuantile2 = self.selectSubsBottomPercent / 100
        print(f'Selection Quantiles:\n'
              f'     1: {red}{self.activityQuantile1}{resetColor}\n'
              f'     2: {red}{self.activityQuantile2}{resetColor}\n')

        # Get Subset: High activity
        threshold1 = dfTrain['activity'].quantile(self.activityQuantile1)
        dfHigh = dfTrain[dfTrain['activity'] > threshold1]
        xHigh = dfHigh.drop(columns='activity').values
        yHigh = np.log1p(dfHigh['activity'].values)

        # Get Subset: Mid activity
        threshold2 = dfTrain['activity'].quantile(self.activityQuantile2)
        dfMid = dfTrain[(dfTrain['activity'] <= threshold1) &
                        (dfTrain['activity'] > threshold2)]
        xMid = dfMid.drop(columns='activity').values
        yMid = np.log1p(dfMid['activity'].values)

        # Get Subset: Low activity subset
        dfLow = dfTrain[dfTrain['activity'] <= threshold2]
        xLow = dfLow.drop(columns='activity').values
        yLow = np.log1p(dfLow['activity'].values)


        pd.set_option('display.max_columns', self.printMaxColumns)
        pd.set_option('display.max_rows', self.printMaxRows)
        print(f'Selecting Top {red}{self.selectSubsTopPercent} %'
              f'{resetColor} of Substrates')
        print(f'Sorted ESM Embeddings: {pink}{self.datasetTagHigh}{resetColor}\n'
              f'{dfHigh.sort_values(by='activity', ascending=False)}\n\n')
        print(f'Sorted ESM Embeddings: {pink}{self.datasetTagMid}{resetColor}\n'
              f'{dfMid.sort_values(by='activity', ascending=False)}\n\n')
        print(f'Sorted ESM Embeddings: {pink}{self.datasetTagLow}{resetColor}\n'
              f'{dfLow.sort_values(by='activity', ascending=False)}\n\n')
        pd.set_option('display.max_columns', None)

        # Split datasets

        testSize = parentObj.testingSetSize
        if self.trainOnlyTopSubs:
            xTrainingH, xTestingH, yTrainingH, yTestingH = train_test_split(
                x, y, test_size=testSize, random_state=19)
        else:
            xTrainingH, xTestingH, yTrainingH, yTestingH = train_test_split(
                xHigh, yHigh, test_size=testSize, random_state=19)
        xTrainingM, xTestingM, yTrainingM, yTestingM = train_test_split(
            xMid, yMid, test_size=testSize, random_state=19)
        xTrainingL, xTestingL, yTrainingL, yTestingL = train_test_split(
            xLow, yLow, test_size=testSize, random_state=19)


        # Put data on the training device
        if 'cuda' in self.device:
            import cupy
            cupy.cuda.Device(self.device.split(':')[1]).use()
            xTrainingH, yTrainingH = cupy.array(xTrainingH), cupy.array(yTrainingH)
            xTestingH, yTestingH = cupy.array(xTestingH), cupy.array(yTestingH)
            xTrainingM, yTrainingM = cupy.array(xTrainingM), cupy.array(yTrainingM)
            xTestingM, yTestingM = cupy.array(xTestingM), cupy.array(yTestingM)
            xTrainingL, yTrainingL = cupy.array(xTrainingL), cupy.array(yTrainingL)
            xTestingL, yTestingL = cupy.array(xTestingL), cupy.array(yTestingL)


        print(f'Training Data: {pink}{self.datasetTagHigh}{resetColor}\n'
              f'Splitting Training Set: '
              f'N = {red}{xTrainingH.shape[0] + xTestingH.shape[0]}{resetColor}, '
              f'{yellow}{round((1 - testSize) * 100, 0)}{pink}:{yellow}'
              f'{round(testSize * 100, 0)}{resetColor}\n'
              f'     Train: {yellow}{xTrainingH.shape}{resetColor}\n'
              f'     Test: {yellow}{xTestingH.shape}{resetColor}\n')
        if self.trainOnlyTopSubs:
            print(f'Training Data: {pink}{self.datasetTagMid}{resetColor}\n'
                  f'Splitting Training Set: '
                  f'N = {red}{xTrainingM.shape[0] + xTestingM.shape[0]}{resetColor}, '
                  f'{yellow}{round((1 - testSize) * 100, 0)}{pink}:{yellow}'
                  f'{round(testSize * 100, 0)}{resetColor}\n'
                  f'     Train: {yellow}{xTrainingM.shape}{resetColor}\n'
                  f'     Test: {yellow}{xTestingM.shape}{resetColor}\n')
            print(f'Training Data: {pink}{self.datasetTagLow}{resetColor}\n'
                  f'Splitting Training Set: '
                  f'N = {red}{xTrainingL.shape[0] + xTestingL.shape[0]}{resetColor}, '
                  f'{yellow}{round((1 - testSize) * 100, 0)}{pink}:{yellow}'
                  f'{round(testSize * 100, 0)}{resetColor}\n'
                  f'     Train: {yellow}{xTrainingL.shape}{resetColor}\n'
                  f'     Test: {yellow}{xTestingL.shape}{resetColor}\n')
            print(f'Unique Substrates: {red}{dfTrain.shape[0]:,}{resetColor}\n')


        def getLayerNumber(col):
            # Extract the list of numbers inside the brackets
            layers = col.partition('[')[2].rpartition(']')[0]
            if layers:
                layerList = layers.split(",")
                return int(layerList[0].strip())  # First layer number
            return float('inf')  # If not found, push to end


        # # Train Or Load Model: Random Forest Regressor
        if (trainModel or
                not os.path.exists(pathModel) and
                not os.path.exists(pathModelH)):
            # Generate parameter combinations
            paramCombos = list(product(*self.paramGrid.values()))
            paramNames = list(self.paramGrid.keys())

            def trainModel(model, xTrain, xTest, yTrain, yTest, tagData,
                           lastModel=False):
                if printData and lastModel:
                    print('========================================'
                          '=========================================\n')
                    print('                       ===================================\n')
                    print('=============================== Training Progress '
                          '===============================')
                    print(f'Training Progress: {red}{combination}{resetColor} / '
                          f'{red}{totalParamCombos}{resetColor} '
                          f'({red}{percentComplete} %{resetColor})\n'
                          f'Hyperparameters: {greenLight}{params}{resetColor}\n'
                          f'ESM Layers: {yellow}{self.layersESM}{resetColor}\n')

                # Train the model
                start = time.time()
                if tagData == self.datasetTagHigh:
                    model.fit(xTrain, yTrain, eval_set=[(xTest, yTest)],
                              verbose=False, sample_weight=yTrain)
                else:
                    model.fit(xTrain, yTrain,eval_set=[(xTest, yTest)],
                              verbose=False)
                end = time.time()

                # Evaluate the model
                yPred = model.predict(xTest)
                if 'cuda' in self.device:
                    yTest = yTest.get()
                yPredNorm = np.expm1(yPred) # Reverse log1p transform
                yTestNorm = np.expm1(yTest)
                self.predictionAccuracy[tagData] = pd.DataFrame({
                    'yTest Norm': yTestNorm,
                    'yPred Norm': yPredNorm
                })
                yPred = yPredNorm * self.maxValue
                yTest = yTestNorm * self.maxValue
                self.predictionAccuracy[tagData].loc[:, 'yTest'] = yTest
                self.predictionAccuracy[tagData].loc[:, 'yPred'] = yPred
                MAE = mean_absolute_error(yPred, yTest)
                MSE = mean_squared_error(yPred, yTest)
                R2 = r2_score(yPred, yTest)
                accuracy = pd.DataFrame()
                accuracy.loc['MAE', self.layerESMTag] = MAE
                accuracy.loc['MSE', self.layerESMTag] = MSE
                accuracy.loc['R²', self.layerESMTag] = R2
                # if R2 == 0.0:
                #     R2 = -float('inf')

                # Inspect results
                saveModel = False
                betterMSE, betterR2 = True, True
                if self.layerESMTag in self.modelAccuracy[tagData].columns:
                    betterMSE = (accuracy.loc[indexEvalMetric, self.layerESMTag] <
                                 self.modelAccuracy[tagData
                                 ].loc[indexEvalMetric, self.layerESMTag])
                    betterR2 = (
                            R2 > self.modelAccuracy[tagData].loc['R²', self.layerESMTag])
                if (self.layerESMTag not in self.modelAccuracy[tagData].columns or
                        betterR2):
                    saveModel = True
                    newColumn = (self.layerESMTag not in
                                 self.modelAccuracy[tagData].columns)

                    # Record: Model performance
                    self.bestParams[tagData] = {self.layerESMTag: params.copy()}
                    self.modelAccuracy[tagData].loc['MAE', self.layerESMTag] = MAE
                    self.modelAccuracy[tagData].loc['MSE', self.layerESMTag] = MSE
                    self.modelAccuracy[tagData].loc['R²', self.layerESMTag] = R2
                    self.modelAccuracy[tagData] = (
                        self.modelAccuracy[tagData].sort_index(axis=1))

                    # Record: Best predictions
                    self.modelBestPredictions[tagData] = self.predictionAccuracy[tagData]

                    # Sort the columns
                    if self.sortedColumns == [] or newColumn:
                        self.sortedColumns = (
                            sorted(self.modelAccuracy[tagData].columns,
                                   key=getLayerNumber))
                    self.modelAccuracy[tagData] = (
                        self.modelAccuracy)[tagData][self.sortedColumns]

                    # Save the data
                    self.modelAccuracy[tagData].to_csv(modelAccuracyPaths[tagData])
                    joblib.dump(model, modelPaths[tagData])

                if printData and lastModel:
                    print(f'Max Activity Score: {red}Max {self.maxValue:,}{resetColor}')
                    for dataset, predictions in self.predictionAccuracy.items():
                        print(f'Prediction Values: {pink}{dataset}{resetColor}\n'
                              f'{greenDark}{predictions}{resetColor}\n\n')

                    runtime = round((end - start), self.roundVal)
                    runtimeTotal = round((end - startTraining) / 60, self.roundVal)
                    rate = round(combination / runtimeTotal, self.roundVal)
                    if rate == 0:
                        timeRemaining = float('inf')
                    else:
                        timeRemaining = round((totalParamCombos - combination) / rate, self.roundVal)
                    for dataset, values in self.modelAccuracy.items():
                        print(f'Prediction Accuracy For Subset: {pink}{dataset}'
                              f'{resetColor}\n'
                              f'Hyperparameters: {greenLight}{self.bestParams[dataset]}')
                        print(f'{blue}{values}{resetColor}\n')
                    print(f'Time Training This Model: '
                          f'{red}{runtime:,} s{resetColor}\n'
                          f'Time Training All Models: '
                          f'{red}{runtimeTotal:,} min{resetColor}\n'
                          f'Training Rate: {red}{rate:,} combinations{resetColor} / '
                          f'{red}min{resetColor}\n'
                          f'Training Progress: {red}{combination}{resetColor} / '
                          f'{red}{totalParamCombos}{resetColor} '
                          f'({red}{percentComplete} %{resetColor})\n'
                          f'Remaining Runtime: '
                          f'{red}{timeRemaining:,} min{resetColor}')
                    # self.plotTestingPredictions()

                    # plt.hist(yTest, bins=50)
                    # plt.title("Test Substrate Score Distribution")
                    # plt.xlabel("Score")
                    # plt.ylabel("Count")
                    # plt.show()

                return model, saveModel



            # Evaluation metric
            if evalMetric == 'rmse':
                indexEvalMetric = 'MSE'
            elif evalMetric == 'mae':
                indexEvalMetric = 'MAE'
            else:
                print(f'{orange}ERROR: There is no use for the evaluation metric '
                      f'{cyan}{evalMetric}\n\n')
                sys.exit(1)
            print(f'Evaluation Metric Used For Training: '
                  f'{pink}{indexEvalMetric}{resetColor}\n\n')


            # Train Models
            nJobs = -1
            startTraining = time.time()
            totalParamCombos = len(paramCombos)
            for combination, paramCombo in enumerate(paramCombos):
                # if combination < 125:
                #     print(f'Skipping combination: {combination}')
                #     continue
                params = dict(zip(paramNames, paramCombo))
                percentComplete = round((combination / totalParamCombos) * 100,
                                        self.roundVal)
                printData = (combination % 25 == 0)

                # for tag in self.models.keys():
                # Train Model
                tag = self.datasetTagHigh
                model, keepModel = trainModel(
                    model=XGBRegressor(
                        device=self.device, n_jobs=nJobs, eval_metric=evalMetric,
                        tree_method="hist", random_state=42, max_bin=64, **params),
                    xTrain=xTrainingH, yTrain=yTrainingH,
                    xTest=xTestingH, yTest=yTestingH,
                    tagData=tag, lastModel=True) # <----------- lastModel=True -----------
                if keepModel:
                    self.modelH = model

                # # Train Model
                # tag = self.datasetTagMid
                # model, keepModel = trainModel(
                #     model=XGBRegressor(
                #         device=self.device, n_jobs=nJobs, eval_metric=evalMetric,
                #         tree_method="hist", random_state=42, max_bin=64, **params),
                #     xTrain=xTrainingM, yTrain=yTrainingM,
                #     xTest=xTestingM, yTest=yTestingM,
                #     tagData=tag)
                # if keepModel:
                #         self.modelM = model
                #
                # # Train Model
                # tag = self.datasetTagLow
                # model, keepModel = trainModel(
                #     model=XGBRegressor(
                #         device=self.device, n_jobs=nJobs, eval_metric=evalMetric,
                #         tree_method="hist", random_state=42, max_bin=64, **params),
                #     xTrain=xTrainingL, yTrain=yTrainingL,
                #     xTest=xTestingL, yTest=yTestingL,
                #     tagData=tag, lastModel=True)
                # if keepModel:
                #     self.modelL = model


            # # End Training
            print('========================================'
                  '=========================================\n')
            print('                       ===================================\n')
            print('=============================== Training Results '
                  '================================')
            end = time.time()
            runtimeTotal = round((end - startTraining) / 60, self.roundVal)
            rate = round(totalParamCombos / runtimeTotal, self.roundVal)
            print(f'Training Completed\n')
            for tag, params in self.bestParams.items():
                print(f'Subset: {pink}{tag}{resetColor}\n'
                      f'Best Hyperparameters: {greenLight}{params}{resetColor}\n'
                      f'Accuracy:\n{blue}{self.modelAccuracy[tag]}{resetColor}\n')
            print(f'Total Training Time: '
                  f'{red}{runtimeTotal:,} min{resetColor}')
            print('========================================'
                  '=========================================\n')
            print('                       ===================================\n')
            print('========================================'
                  '=========================================\n\n')

            # Plot the results from the best model
            self.plotTestingPredictions(finalSet=True)
        else:
            self.modelH = self.loadModel(pathModel=pathModelH, tag=datasetTagHigh)
            self.modelM = self.loadModel(pathModel=pathModel, tag=datasetTagMid)
            self.modelL = self.loadModel(pathModel=pathModel, tag=datasetTagLow)



    def plotTestingPredictions(self, finalSet=False):
        print(f'====================== Scatter Plot - Prediction Accuracy '
              f'======================')
        if finalSet:
            data = self.modelBestPredictions
        else:
            data = self.predictionAccuracy

        for tag, predictions in data.items():
            print(f'Subset: {pink}{tag}{resetColor}\n{predictions}\n\n')
            x = predictions.loc[:, 'yTest']
            y = predictions.loc[:, 'yPred']

            # Evaluate data
            maxValue = max(max(x), max(y))
            magnitude = math.floor(math.log10(maxValue))
            unit = 10 ** (magnitude - 1)
            maxValue = math.ceil(maxValue / unit) * unit
            maxValue += unit
            axisLimits = [0, maxValue]
            R2 = self.modelAccuracy[tag].loc['R²', self.layerESMTag]


            # Plot the data
            fig, ax = plt.subplots(figsize=self.figSize)
            plt.scatter(x, y, alpha=0.7, color='#BF5700', edgecolors='#F8971F', s=50)
            plt.plot(axisLimits, axisLimits, color='#101010', lw=self.lineThickness)
            plt.xlabel('Experimental Activity', fontsize=self.labelSizeAxis)
            plt.ylabel('Predicted Activity', fontsize=self.labelSizeAxis)
            plt.title(f'{tag}\nR² = {round(R2, 2)}\nRandon Forest Regressor\n'
                      f'{self.layerESMTag}', fontsize=self.labelSizeTitle,
                      fontweight='bold')
            plt.subplots_adjust(top=0.852, bottom=0.075, left=0.162, right=0.935)

            # Set axes
            ax.set_xlim(0, maxValue)
            ax.set_ylim(0, maxValue)

            # Set tick parameters
            ax.tick_params(axis='both', which='major', length=self.tickLength,
                           labelsize=self.labelSizeTicks, width=self.lineThickness)

            # Set the thickness of the figure border
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(self.lineThickness)

            plt.grid(True)
            fig.canvas.mpl_connect('key_press_event', pressKey)
            plt.show()


            # Define: Save location
            figLabel = self.tagExperiment[tag] + '.png'
            saveLocation = os.path.join(self.pathFiguresTraining, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                print(f'{yellow}WARNING{resetColor}: '
                      f'{yellow}The figure already exists at the path\n'
                      f'     {saveLocation}\n\n'
                      f'We will not overwrite the figure{resetColor}\n\n')
            else:
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)




    def makePredictions(self, model, tag):
        # Expand prediction values
        predictedValueScaler = self.maxValue

        # Predict substrate activity
        print(f'Predicting Substrate Activity: {purple}{tag}{resetColor}\n'
              f'     Total Substrates: {red}{len(dfPred.index):,}{resetColor}')
        start = time.time()
        activityPred = model.predict(dfPred.values)
        activityPred = np.expm1(activityPred) # Reverse log1p transform
        end = time.time()
        runtime = (end - start) * 1000
        print(f'     Runtime: {red}{round(runtime, self.roundVal):,} ms{resetColor}\n')

        # Rank predictions
        activityPredRandom = {
            substrate: float(score)
            for substrate, score in zip(subsPred, activityPred)
        }
        activityPredRandom = dict(sorted(
            activityPredRandom.items(), key=lambda x: x[1], reverse=True))
        self.predictions['Random'] = activityPredRandom
        print(f'Predicted Activity: {purple}Random Substrates - Min ES: {minES}'
              f'{resetColor}')
        for iteration, (substrate, value) in enumerate(activityPredRandom.items()):
            if iteration >= printNumber:
                break
            print(f'     {substrate}: {red}{round(value, self.roundVal):,}{resetColor}')
        print('\n')

        # Get: Chosen substrate predictions
        if subsPredChosen != {}:
            print(f'Predicted Activity: {purple}Chosen Substrates{resetColor}')
            for key, substrates in subsPredChosen.items():
                activityPredChosen = {}
                for substrate in substrates:
                    activityPredChosen[substrate] = (
                        activityPredRandom)[substrate]
                activityPredChosen = dict(sorted(activityPredChosen.items(),
                                                 key=lambda x: x[1], reverse=True))
                self.predictions[key] = activityPredChosen
                print(f'Substrate Set: {purple}{key}{resetColor}')
                for iteration, (substrate, activity) in (
                        enumerate(activityPredChosen.items())):
                    activity = float(activity)
                    print(f'     {pink}{substrate}{resetColor}: '
                          f'{red}{round(activity, self.roundVal):,}{resetColor}')
                print('\n')


    def predictSubstrateAffinity(self, pathModel, pathModelHigh):
        print(f'Find a way to record multiple predictions.')

        self.model = self.loadModel(pathModel=pathModel, tag=self.modelTag)
        self.modelH = self.loadModel(pathModel=pathModelHigh, tag=self.modelTagHigh)
        makePredictions(model=self.model, tag=self.modelTag)
        makePredictions(model=self.modelH, tag=self.modelTagHigh)



    @staticmethod
    def loadModel(pathModel, tag):
        print(f'Loading Trained ESM Model: {purple}{tag}{resetColor}\n'
              f'     {greenDark}{pathModel}{resetColor}\n')
        model = joblib.load(pathModel)
        print(f'Model Params: {greenLight}{model.get_params()}{resetColor}\n\n')

        return model



class PredictActivity:
    def __init__(self, enzymeName, datasetTag, folderPath, subsTrain, subsPred,
                 subsPredChosen, subsPercentSelectTop, subsPercentSelectBottom,
                 maxTrainingScore, useEF, filterPCA, tagChosenSubs, minSubCount,
                 minES, modelType, concatESM, layersESM, testSize, batchSize,
                 labelsXAxis, printNumber, saveFigures, modelSize=2):
        # Parameters: Files
        self.pathFolder = folderPath
        self.pathData = os.path.join(self.pathFolder, 'Data')
        self.pathEmbeddings = os.path.join(self.pathFolder, 'Embeddings')
        self.pathModels = os.path.join(self.pathFolder, 'Models')
        self.pathFigures = os.path.join(self.pathFolder, 'Figures')
        self.pathFiguresTraining = os.path.join(self.pathFolder, 'FiguresModelTraining')
        os.makedirs(self.pathData, exist_ok=True)
        os.makedirs(self.pathEmbeddings, exist_ok=True)
        os.makedirs(self.pathModels, exist_ok=True)
        os.makedirs(self.pathFigures, exist_ok=True)
        os.makedirs(self.pathFiguresTraining, exist_ok=True)

        # Parameters: Dataset
        self.enzymeName = enzymeName
        self.datasetTag = datasetTag
        self.subsInitial = None
        self.subsTrain = subsTrain
        self.subsLen = len(next(iter(self.subsTrain)))
        self.subsPercentSelectTop = subsPercentSelectTop
        self.subsPercentSelectBottom = subsPercentSelectBottom
        self.maxTrainingScore = maxTrainingScore
        self.subsTrainN = len(self.subsTrain.keys())
        self.subsPred = subsPred
        self.subsPredN = len(subsPred)
        self.subsPredChosen = subsPredChosen
        self.useEF = useEF
        self.filterPCA = filterPCA
        self.numPCs = 2
        self.tagChosenSubs = tagChosenSubs
        self.minSubCount = minSubCount
        self.minES = minES
        self.labelsXAxis = labelsXAxis
        self.printNumber = printNumber
        self.selectedSubstrates = []
        self.selectedDatapoints = []
        self.rectangles = []
        self.predictions = {}

        # Parameters: Figures
        self.saveFigures = saveFigures
        self.figSize = (9.5, 8)
        self.labelSizeTitle = 18
        self.labelSizeAxis = 16
        self.labelSizeTicks = 13
        self.lineThickness = 1.5
        self.tickLength = 4
        self.figureResolution = 300

        # Parameters: Misc
        self.roundVal = 3
        self.printMaxColumns = 8
        self.printMaxRows = 10

        # Parameters: Model
        self.modelType = modelType
        self.subsetTagHigh = f'Top {self.subsPercentSelectTop} %'
        self.subsetTagMid = f'Mids'
        self.subsetTagLow = f'Bottom {self.subsPercentSelectBottom} %'
        accuracyDF = pd.DataFrame(0.0, index=['MAE','MSE','R²'], columns=[])
        self.trainOnlyTopSubs = True
        if self.trainOnlyTopSubs:
            self.modelAccuracy = {
                self.subsetTagHigh: accuracyDF.copy()
            }
        else:
            self.modelAccuracy = {
                self.subsetTagHigh: accuracyDF.copy(),
                self.subsetTagMid: accuracyDF.copy(),
                self.subsetTagLow: accuracyDF.copy(),
            }
        self.concatESM = concatESM
        self.layersESM = layersESM
        self.layersESMTag = f'ESM L{self.layersESM}'.replace(', ', ',')
        self.embeddingsPathTrain = []
        self.embeddingsPathPred = []
        self.batchSize = batchSize
        self.testingSetSize = testSize
        self.device = self.getDevice()

        # Parameters: ESM
        if modelSize == 0:  # Choose: ESM PLM model
            self.sizeESM = '15B Params'
        elif modelSize == 1:
            self.sizeESM = '3B Params'
        else:
            self.sizeESM = '650M Params'

        # Parameters: Save Paths Embeddings
        if self.useEF:
            scoreType = 'EF'
        else:
            scoreType = 'Counts'
        self.embeddingsTagTrain = (
            f'Embeddings - {self.layersESMTag} {self.sizeESM} - Batch '
            f'{self.batchSize} - {self.enzymeName} - {self.datasetTag} - {scoreType} - '
            f'MinCounts {self.minSubCount} - N {self.subsTrainN} - '
            f'{self.subsLen} AA')
        self.embeddingsTagPred = (
            f'Embeddings - {self.layersESM} {self.sizeESM} - Batch '
            f'{self.batchSize} - {self.enzymeName} -  Predictions - '
            f'Min ES {self.minES} - {scoreType} - MinCounts {self.minSubCount} - '
            f'N {self.subsPredN} - {self.subsLen} AA')
        if (self.concatESM
                and isinstance(self.layersESM, list)
                and len(self.layersESM) > 1):
            for layer in self.layersESM:
                filePath = f'{self.embeddingsTagTrain.replace(
                    f'{self.layersESMTag}', f'ESM L{layer}')}.csv'
                self.embeddingsPathTrain.append(
                    os.path.join(self.pathEmbeddings, filePath))
        else:
            self.concatESM = False
            if isinstance(self.layersESM, int):
                self.layersESM = [self.layersESM]
            filePath = f'{self.embeddingsTagTrain.replace(
            'ESM', f'ESM L{self.layersESM[0]}')}.csv'
            self.embeddingsPathTrain.append(
                    os.path.join(self.pathEmbeddings, filePath))

        for layer in self.layersESM:
            filePath = f'{self.embeddingsTagPred.replace(
                f'{self.layersESMTag}', f'ESM L{layer}')}.csv'
            self.embeddingsPathPred.append(
                    os.path.join(self.pathEmbeddings, filePath))
        if self.tagChosenSubs != '':
            self.embeddingsTagPred = self.embeddingsTagPred.replace(
                f'MinCounts {self.minSubCount}',
                f'MinCounts {self.minSubCount} - Added {self.tagChosenSubs}')

        # Parameters: Save Paths Model Accuracy
        self.tagExperiment = (f'Model Accuracy - {modelType} - {self.layersESMTag} '
                              f'{self.sizeESM} - {enzymeName} - {datasetTag} - '
                              f'N {self.subsTrainN} - MinCounts {minSubCount} - '
                              f'{self.subsLen} AA')
        self.tagExperiment = self.tagExperiment.replace(':', '')
        if self.useEF:
            self.tagExperiment = self.tagExperiment.replace(
                'MinCounts', 'EF - MinCounts')
        else:
            self.tagExperiment = self.tagExperiment.replace(
                'MinCounts', 'Counts - MinCounts')
        self.tagExperiment = {
            self.subsetTagHigh: self.tagExperiment.replace(
                'MinCounts', f'{self.subsetTagHigh} - MinCounts'),
            self.subsetTagMid: self.tagExperiment.replace(
                'MinCounts', f'{self.subsetTagMid} - MinCounts'),
            self.subsetTagLow: self.tagExperiment.replace(
                'MinCounts', f'{self.subsetTagLow} - MinCounts')
        }
        self.pathModelAccuracy = {
            self.subsetTagHigh: os.path.join(
                self.pathModels, self.tagExperiment[self.subsetTagHigh]),
            self.subsetTagMid: os.path.join(
                self.pathModels, self.tagExperiment[self.subsetTagMid]),
            self.subsetTagLow: os.path.join(
                self.pathModels, self.tagExperiment[self.subsetTagLow])
        }
        self.loadModelAccuracies()



    def loadModelAccuracies(self):
        for tag, path in self.pathModelAccuracy.items():
            if os.path.exists(path):
                print('=========================== Loading Model Accuracies '
                      '============================')
                print(f'Loading File:\n'
                      f'     {greenDark}{path}{resetColor}\n')
                values = pd.read_csv(path, index_col=0)
                self.modelAccuracy[tag] = values
                print(f'Loaded Values: {pink}{tag}{resetColor}\n'
                      f'{blue}{self.modelAccuracy[tag]}{resetColor}\n\n')



    def trainModel(self):
        # Define: Model paths
        modelTag = (f'Random Forest - Test Size {self.testingSetSize} - '
                    f'{self.embeddingsTagTrain.replace(self.layersESMTag, 'ESM')}')
        # modelTag = (f'Random Forest - Test Size {self.testingSetSize} - '
        #             f'N Trees {self.NTrees} - {self.embeddingsTagTrain}')
        modelTagScikit = modelTag.replace('Test Size',
                                          f'Scikit - Test Size')
        modelTagXGBoost = modelTag.replace('Test Size',
                                           f'XGBoost - Test Size')
        pathModelScikit = os.path.join(self.pathModels, f'{modelTagScikit}.ubj')
        pathModelXGBoost = os.path.join(self.pathModels, f'{modelTagXGBoost}.ubj')
        # ubj: Binary JSON file

        # Generate: Embeddings
        self.embeddingsSubsTrain = None
        self.embeddingsSubsPred = pd.DataFrame()
        if (not os.path.exists(pathModelScikit) or
                not os.path.exists(pathModelXGBoost)):
            self.embeddingsSubsTrain = self.ESM(
                substrates=self.subsTrain,  filePaths=self.embeddingsPathTrain,
                trainingSet=True)
        # self.embeddingsSubsPred = self.ESM(
        #     substrates=self.subsPred, filePaths=self.embeddingsPathPred)

        # End function
        if self.embeddingsSubsTrain is None:
            print(f'{orange}ESM output{resetColor} is None\n'
                  f'ESM layer {red}{self.layersESM}{resetColor} cannot be extracted\n\n')
            sys.exit()

        # Select a model to train
        if self.modelType == 'Random Forest Regressor: Scikit-Learn':
            # Model: Scikit-Learn Random Forest Regressor
            randomForestRegressor = RandomForestRegressor(
                dfTrain=self.embeddingsSubsTrain, dfPred=self.embeddingsSubsPred,
                subsPredChosen=self.subsPredChosen, minES=self.minES,
                pathModel=pathModelScikit, modelTag=modelTagScikit, layerESM=layerESM,
                testSize=self.testingSetSize, device=self.device,
                printNumber=self.printNumber)
            self.modelAccuracy = randomForestRegressor.modelAccuracy
            self.predictions[self.modelType] = randomForestRegressor.predictions
        elif self.modelType == 'Random Forest Regressor: XGBoost':
            # Model: XGBoost Random Forest
            randomForestRegressorXGB = RandomForestRegressorXGB(parentObj=self,
                pathModel=pathModelXGBoost, modelTag=modelTagXGBoost)
            # Record predictions
            self.predictions[self.modelType] = randomForestRegressorXGB.predictions
            for dataset, values, in randomForestRegressorXGB.modelAccuracy.items():
                self.modelAccuracy[dataset].loc[:, values.columns] = values
        else:
            print(f'{orange}ERROR: There is no use for the model type '
                  f'{cyan}{self.modelType}{resetColor}\n\n')
            sys.exit(1)

        self.predictionAccuracies()



    def getDevice(self):
        # Set device
        print('============================== Set Training Device '
              '==============================')
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f'Train with Device:{magenta} {device}{resetColor}\n'
                  f'Device Name:{magenta} {torch.cuda.get_device_name(device)}'
                  f'{resetColor}\n\n')
        else:
            device = 'cpu'
            print(f'Train with Device:{magenta} {device}{resetColor}\n'
                  f'Device Name:{magenta} {platform.processor()}{resetColor}\n\n')

        return device



    def ESM(self, substrates, filePaths, trainingSet=False):
        print('======================= Evolutionary Scale Modeling (ESM) '
              '=======================')
        pd.set_option('display.max_columns', self.printMaxColumns)
        pd.set_option('display.max_rows', self.printMaxRows)
        missingLayersESM = []

        # Inspect: Data type
        predictions = True
        if trainingSet:
            predictions = False
        print(f'ESM Layers: {yellow}{self.layersESM}{resetColor}\n'
              f'Total unique substrates: {red}{len(substrates):,}{resetColor}')
        print(f'Concatenating ESM Layers: {purple}{self.concatESM}{resetColor}\n')


        # Define Functions
        def generateEmbeddingsESM(seqs, layersESM, savePaths):
            print(f'Generating ESM Embeddings:')
            for layerESM in layersESM:
                print(f'     Layer: {yellow}{layerESM}{resetColor}')
            print('')

            # # Generate Embeddings
            # # Step 1: Convert substrates to ESM model format and generate Embeddings
            totalSubActivity = 0
            subs = []
            values = []
            if trainingSet:
                # Randomize substrates
                items = list(seqs.items())
                random.shuffle(items)
                seqs = dict(items)

                for index, (substrate, value) in enumerate(seqs.items()):
                    totalSubActivity += value
                    subs.append((f'Sub{index}', substrate))
                    values.append(value)
            else:
                for index, substrate in enumerate(seqs):
                    subs.append((f'Sub{index}', substrate))
            if totalSubActivity != 0:
                if isinstance(totalSubActivity, float):
                    print(f'Total Values:{red} {round(totalSubActivity, 1):,}'
                          f'{resetColor}')
                else:
                    print(f'Total Values:{red} {totalSubActivity:,}{resetColor}')
            print()


            # # Step 2: Load the ESM model and batch converter
            if self.sizeESM == '15B Params':
                model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
                layersESMMax = 48
            elif self.sizeESM == '3B Params':
                model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
                layersESMMax = 36
            else:
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                layersESMMax = 33
                # esm2_t36_3B_UR50D has 36 layers
                # esm2_t33_650M_UR50D has 33 layers
                # esm2_t12_35M_UR50D has 12 layers
            model = model.to(self.device)

            # End function
            if any(layer > layersESMMax for layer in layersESM):
                return None


            # Get: Batch tensor
            batchConverter = alphabet.get_batch_converter()

            # # Step 3: Convert substrates to ESM model format and generate Embeddings
            try:
                batchLabels, batchSubs, batchTokens = batchConverter(subs)
                batchTokensCPU = batchTokens
                batchTokens = batchTokens.to(self.device)

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
                                        columns=self.labelsXAxis)
            if totalSubActivity != 0:
                slicedTokens['Values'] = values
            print(f'\nSliced Tokens:\n'
                  f'{greenLight}{slicedTokens}{resetColor}\n')


            # Generate embeddings
            batchTotal = len(batchTokens)

            for index, layerESM in enumerate(layersESM):
                print(f'Generating New ESM Embedding:')
                startInit = time.time()
                allValues = []
                allEmbeddings = []
                pathEmbeddings = savePaths[index]

                with torch.no_grad():
                    for i in range(0, len(batchTokens), self.batchSize):
                        start = time.time()
                        batch = batchTokens[i:i + self.batchSize].to(self.device)
                        result = model(batch, repr_layers=[layerESM],
                                       return_contacts=False)
                        tokenReps = result["representations"][layerESM]
                        seqEmbed = tokenReps[:, 0, :].cpu().numpy()
                        allEmbeddings.append(seqEmbed)
                        end = time.time()
                        runtime = end - start
                        runtimeTotal = (end - startInit) / 60
                        percentCompletion = round((i / batchTotal)* 100, 1)
                        if i % 10 == 0:
                            rate = round(i / runtimeTotal, self.roundVal)
                            if rate == 0:
                                timeRemaining = float('inf')
                            else:
                                timeRemaining = round((batchTotal - i) / rate,
                                                      self.roundVal)
                            print(f'ESM Progress ({yellow}Layer {layerESM}{resetColor}): '
                                  f'{red}{i:,}{resetColor} / {red}{batchTotal:,}'
                                  f'{resetColor} '
                                  f'({red}{percentCompletion} %{resetColor})\n'
                                  f'     Batch Shape: {greenLight}{batch.shape}'
                                  f'{resetColor}\n'
                                  f'     Runtime: {red}'
                                  f'{round(runtime, self.roundVal):,} s{resetColor}\n'
                                  f'     Total Time: {red}'
                                  f'{round(runtimeTotal, self.roundVal):,} min'
                                  f'{resetColor}\n'
                                  f'     Remaining Runtime: {red}{timeRemaining:,} min'
                                  f'{resetColor}\n')
                        if trainingSet:
                            allValues.extend(values[i:i + self.batchSize])

                        # Clear data to help free memory
                        del tokenReps, batch
                        torch.cuda.empty_cache()
                end = time.time()
                runtime = end - start
                runtimeTotal = (end - startInit) / 60
                percentCompletion = round((batchTotal / batchTotal) * 100, 1)
                print(f'ESM Progress ({yellow}Layer {layerESM}{resetColor}): '
                      f'{red}{batchTotal:,}{resetColor} / {red}{batchTotal:,}'
                      f'{resetColor} ({red}{percentCompletion} %{resetColor})\n'
                      f'     Runtime: {red}{round(runtime, self.roundVal):,} s'
                      f'{resetColor}\n'
                      f'     Total Time: {red}{round(runtimeTotal, self.roundVal):,} min'
                      f'{resetColor}\n'
                      f'     Remaining Runtime: {red}{0:,} min'
                      f'{resetColor}\n')


                # # Step 4: Extract per-sequence embeddings
                # (N, seq_len, hidden_dim)
                tokenReps = result["representations"][layerESM]
                # [CLS] token embedding: (N, hidden_dim)
                sequenceEmbeddings = tokenReps[:, 0, :]

                # Convert to numpy and store substrate activity proxy
                embeddings = np.vstack(allEmbeddings)
                if predictions:
                    data = np.hstack([embeddings])
                    columns = [f'L{layerESM}_feat_{i}'
                               for i in range(embeddings.shape[1])]
                else:
                    values = np.array(allValues).reshape(-1, 1)
                    data = np.hstack([embeddings, values])
                    columns = [f'L{layerESM}_feat_{i}'
                               for i in range(embeddings.shape[1])] + ['activity']

                # Process Embeddings
                subEmbeddings = pd.DataFrame(data, index=batchSubs, columns=columns)
                print(f'Substrate Embeddings:\n{subEmbeddings}\n\n')
                print(f'Substrate Embeddings shape: '
                      f'{pink}{sequenceEmbeddings.shape}{resetColor}')
                print(f'Embeddings saved at:\n'
                      f'     {greenDark}{pathEmbeddings}{resetColor}\n\n')
                subEmbeddings.to_csv(pathEmbeddings)

                # plt.hist(subEmbeddings.loc[:, 'activity'], bins=100)
                # plt.title("Activity Distribution")
                # plt.show()

            del model, alphabet


        def loadESM():
            missingFiles = []
            loadedEmbeddings = []

            # Load: ESM Embeddings
            genEmbeddings = False
            for index, pathEmbeddings in enumerate(filePaths):
                if not os.path.exists(pathEmbeddings):
                    genEmbeddings = True
                    layer = self.layersESM[index]
                    missingFiles.append(pathEmbeddings)
                    missingLayersESM.append(layer)

            if genEmbeddings:
                generateEmbeddingsESM(seqs=substrates, layersESM=missingLayersESM,
                                      savePaths=missingFiles)
                loadedEmbeddings = loadESM()
            else:
                for index, pathEmbeddings in enumerate(filePaths):
                    loadedEmbeddings.append(pd.read_csv(pathEmbeddings, index_col=0))
                for index, layerEmbeddings in enumerate(loadedEmbeddings):
                    print(f'Loaded Embeddings: '
                          f'{yellow}Layer {self.layersESM[index]}{resetColor}\n'
                          f'{layerEmbeddings}\n\n')

            return loadedEmbeddings

        # Get: ESM data
        loadedEmbeddings = loadESM()
        if self.concatESM:
            if trainingSet:
                subEmbeddings = pd.concat(
                    [df.drop(columns=['activity']) for df in loadedEmbeddings],
                    axis=1 # concatenate columns
                )
                subEmbeddings['activity'] = loadedEmbeddings[0]['activity']
            else:
                subEmbeddings = pd.concat(
                    [df for df in loadedEmbeddings],
                    axis=1 # concatenate columns
                )
        else:
            subEmbeddings = loadedEmbeddings[0]
        if self.filterPCA:
            # Filter Data: PCA
            subEmbeddings = self.plotPCA(substrates=substrates, embeddings=subEmbeddings)
        else:
            print(f'Substrate Embeddings:\n'
                  f'{greenLight}{subEmbeddings}{resetColor}\n\n')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        
        return subEmbeddings



    def predictionAccuracies(self):
        print('============================= Prediction Accuracies '
              '=============================')
        print(f'Machine Learning Model: {purple}{self.modelType}{resetColor}\n'
              f'Training Dataset: {purple}{self.enzymeName}{resetColor} - '
              f'{purple}{self.datasetTag}{resetColor}\n\n'
              f'Most Accurate Predicions:')
        for tag, values in self.modelAccuracy.items():
            pathSave = self.pathModelAccuracy[tag]
            print(f'Subset: {pink}{tag}\n'
                  f'{blue}{values}{resetColor}\n')
            values.to_csv(pathSave)



    def plotPCA(self, substrates, embeddings, combinedMotifs=False):
        print('====================================== PCA '
              '======================================')
        N = len(substrates.keys())
        indices = embeddings.index

        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n'
              f'Total unique substrates: {red}{N:,}{resetColor}\n')

        # Initialize lists for the clustered substrates
        rectangles = []


        # # Cluster the datapoints
        # Step 1: Apply PCA on the standardized data
        pca = PCA(n_components=self.numPCs) # Adjust the number of components as needed
        scaler = StandardScaler()
        data = scaler.fit_transform(embeddings.drop(columns='activity'))
        dataPCA = pca.fit_transform(data)
        # loadings = pca.components_.T
        if self.numPCs == 2:
            # Define component labels
            pcaHeaders = []
            for componentNumber in range(1, self.numPCs + 1):
                pcaHeaders.append(f'PC{componentNumber}')
        else:
            # Generate a matching number of column headers
            pcaHeaders = [f'PC{i + 1}' for i in range(dataPCA.shape[1])]
        headerCombinations = list(combinations(pcaHeaders, 2))

        # Step 2: Create a DataFrame for PCA results
        dataPCA = pd.DataFrame(dataPCA, columns=pcaHeaders, index=indices)
        print(f'PCA data:{red} # of components = {self.numPCs}\n'
              f'{greenLight}{dataPCA}{resetColor}\n\n')

        # Step 3: Print explained variance ratio
        varRatio = pca.explained_variance_ratio_ * 100
        print(f'Explained Variance Ratio: '
              f'{red}{" ".join([f"{x:.3f}" for x in varRatio])}{resetColor} %\n\n')


        # Define: Figure title
        title = (f'\n{self.enzymeName}\n'
                 f'{self.datasetTag}\n'
                 f'{self.layersESMTag.replace(',', ', ')}\n'
                 f'{N:,} Unique Substrates')

        if combinedMotifs and len(self.motifIndexExtracted) > 1:
            title = title.replace('Motifs', 'Combined Motifs')


        # Plot the data
        for components in headerCombinations:
            fig, ax = plt.subplots(figsize=self.figSize)

            def selectDatapoints(eClick, eRelease):
                # # Function to update selection with a rectangle

                nonlocal ax, rectangles

                # Define x, y coordinates
                x1, y1 = eClick.xdata, eClick.ydata  # Start of the rectangle
                x2, y2 = eRelease.xdata, eRelease.ydata  # End of the rectangle

                # Collect selected datapoints
                selection = []
                selectedSubs = []
                for index, (x, y) in enumerate(zip(dataPCA.loc[:, 'PC1'],
                                                   dataPCA.loc[:, 'PC2'])):
                    if (min(x1, x2) <= x <= max(x1, x2) and
                            min(y1, y2) <= y <= max(y1, y2)):
                        selection.append((x, y))
                        selectedSubs.append(dataPCA.index[index])
                if selection:
                    self.selectedDatapoints.append(selection)
                    self.selectedSubstrates.append(selectedSubs)

                # Draw the boxes
                if self.selectedDatapoints:
                    for index, box in enumerate(self.selectedDatapoints):
                        # Calculate the bounding box for the selected points
                        padding = 0.05
                        xMinBox = min(x for x, y in box) - padding
                        xMaxBox = max(x for x, y in box) + padding
                        yMinBox = min(y for x, y in box) - padding
                        yMaxBox = max(y for x, y in box) + padding

                        # Draw a single rectangle around the bounding box
                        boundingRect = plt.Rectangle((xMinBox, yMinBox),
                                                     width=xMaxBox - xMinBox,
                                                     height=yMaxBox - yMinBox,
                                                     linewidth=2,
                                                     edgecolor='black',
                                                     facecolor='none')
                        ax.add_patch(boundingRect)
                        self.rectangles.append(boundingRect)

                        # Add text only if there are multiple boxes
                        if len(self.selectedDatapoints) > 1:
                            # Calculate the center of the rectangle for text positioning
                            centerX = (xMinBox + xMaxBox) / 2
                            centerY = (yMinBox + yMaxBox) / 2

                            # Number the boxes
                            text = ax.text(centerX, centerY, f'{index + 1}',
                                           horizontalalignment='center',
                                           verticalalignment='center',
                                           fontsize=25,
                                           color='#F79620',
                                           fontweight='bold')
                            text.set_path_effects(
                                [path_effects.Stroke(linewidth=2, foreground='black'),
                                 path_effects.Normal()])
                plt.draw()
            plt.scatter(dataPCA[components[0]], dataPCA[components[1]],
                        c='#CC5500', edgecolor='black')
            plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
            plt.xlabel(f'Principal Component {components[0][-1]} '
                       f'({np.round(varRatio[0], self.roundVal)} %)',
                       fontsize=self.labelSizeAxis)
            plt.ylabel(f'Principal Component {components[1][-1]} '
                       f'({np.round(varRatio[1], self.roundVal)} %)',
                       fontsize=self.labelSizeAxis)
            plt.subplots_adjust(top=0.852, bottom=0.075, left=0.13, right=0.938)


            # Set tick parameters
            ax.tick_params(axis='both', which='major', length=self.tickLength,
                           labelsize=self.labelSizeTicks, width=self.lineThickness)

            # Set the thickness of the figure border
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(self.lineThickness)


            # Create a RectangleSelector
            selector = RectangleSelector(ax,
                                         selectDatapoints,
                                         useblit=True,
                                         minspanx=5,
                                         minspany=5,
                                         spancoords='pixels',
                                         interactive=True)

            # Change rubber band color
            selector.set_props(facecolor='none', edgecolor='green', linewidth=3)

            fig.canvas.mpl_connect('key_press_event', pressKey)
            # fig.tight_layout()
            plt.show()


        # # Save the Figure
        if self.saveFigures:
            # Define: Save location
            figLabel = \
                (f'{self.enzymeName} - PCA - ESM {self.sizeESM} {self.layersESMTag} - '
                 f'{self.datasetTag} - {N} - MinCounts {self.minSubCount} - '
                 f'{self.subsLen} AA.png')
            # if self.useEF:
            #     figLabel = figLabel.replace('MinCounts', 'EF - MinCounts')
            # else:
            #     figLabel = figLabel.replace('MinCounts', 'Count - MinCounts')
            saveLocation = os.path.join(self.pathFiguresTraining, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                figExists = True
                figLabel = figLabel.replace('PCA', f'PCA 1')
                saveLocation = os.path.join(self.pathFiguresTraining, figLabel)
                if os.path.exists(saveLocation):
                    while figExists:
                        for index in range(2, 501):
                            figLabel = figLabel.replace(f'PCA {index - 1}',
                                                        f'PCA {index}')
                            saveLocation = os.path.join(self.pathFiguresTraining,
                                                        figLabel)
                            if not os.path.exists(saveLocation):
                                figExists = False
                                print(f'Exists: {figExists}')
                                break
            print(f'Saving figure at path:\n'
                  f'     {greenDark}{saveLocation}{resetColor}\n\n')
            fig.savefig(saveLocation, dpi=self.figureResolution)

        # Create a list of collected substrate dictionaries
        if self.selectedSubstrates:
            filteredEmbeddings = pd.DataFrame()
            for index, substrateSet in enumerate(self.selectedSubstrates):
                filteredData = embeddings.loc[substrateSet, :]
                print(f'Sorted Filtered Substrate Embeddings: '
                      f'{red}PCA Cluster {index + 1}{resetColor}\n'
                      f'{filteredData.sort_values(by='activity', ascending=False)}\n\n')
                filteredEmbeddings = pd.concat([filteredEmbeddings, filteredData],
                                               ignore_index=False)
            print(f'Sorted Filtered Data: {purple}{self.layersESMTag}{resetColor}\n'
                  f'{greenLight}{filteredEmbeddings.sort_values(
                      by='activity', ascending=False)}{resetColor}\n\n')
            time.sleep(2)
            return filteredEmbeddings
        else:
            return embeddings
