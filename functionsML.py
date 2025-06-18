# PURPOSE: This script contains the functions that you will need to process your NGS data



import esm
from itertools import  product
import joblib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
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
                 modelTagHigh, testSize, NTrees, device, printNumber):
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
                print(f'Training Time: {red}{round(runtime, 3):,} min{resetColor}\n')
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
                      f'     MAE: {red}{round(MAE, 3)}{resetColor}\n'
                      f'     MSE: {red}{round(MSE, 3)}{resetColor}\n'
                      f'     R2: {red}{round(R2, 3)}{resetColor}\n')
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
            print(f'     Runtime: {red}{round(runtime, 3):,} ms{resetColor}\n')

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
                print(f'     {substrate}: {red}{round(value, 3):,}{resetColor}')
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
                              f'{red}{round(activity, 3):,}{resetColor}')
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
Params Grid: Random ForestRegressor - XGB
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
    def __init__(self, dfTrain, dfPred, maxValue, tagExperiment, selectSubsTopPercent,
                 selectSubsBottomPercent, pathModel, modelTag, modelAccuracy,
                 modelAccuracyPaths, pathFigures, datasetTagHigh, datasetTagMid,
                 datasetTagLow, layerESM, testSize, device, trainModel=True,
                 evalMetric='rmse'):
        """
        :param evalMetric: options include 'rmse' and 'mae'
        """

        print('================================== Train Model '
              '==================================')
        print(f'ML Algorithm: {purple}Random Forest Regressor{resetColor}\n'
              f'Module: {purple}XGBoost{resetColor}\n'
              f'Model: {purple}{modelTag}{resetColor}\n')

        self.device = device
        self.tagExperiment = tagExperiment
        self.maxValue = maxValue
        self.selectSubsTopPercent = selectSubsTopPercent
        self.selectSubsBottomPercent = selectSubsBottomPercent
        self.subsPred = list(dfPred.index)
        self.predictions = {}
        self.layerESM = layerESM
        self.layerESMTag = f'ESM Layer {self.layerESM}'
        self.modelAccuracy = modelAccuracy
        self.predictionAccuracy = {}
        self.modelBestPredictions = {}
        self.bestParams = {datasetTagHigh: {}, datasetTagMid: {}, datasetTagLow: {}}
        self.sortedColumns = []

        # Parameters: Figures
        self.pathFigures = pathFigures
        self.figSize = (9.5, 8)
        self.figSizeMini = (self.figSize[0], 6)
        self.residueLabelType = 2 # 0 = full AA name, 1 = 3-letter code, 2 = 1 letter
        self.labelSizeTitle = 18
        self.labelSizeAxis = 16
        self.labelSizeTicks = 13
        self.lineThickness = 1.5
        self.tickLength = 4
        self.figureResolution = 300

        # Params: Grid search
        # self.paramGrid = {
        #     'colsample_bytree': np.arange(0.6, 1.0, 0.2),
        #     'learning_rate': [0.05, 0.01],
        #     'max_leaves': range(100, 250, 50),
        #     'min_child_weight': range(1, 3, 1),
        #     'n_estimators': range(3500, 5500, 500),
        #     'subsample': np.arange(0.5, 1.0, 0.1)
        # }
        self.paramGrid = {
            'colsample_bytree': np.arange(0.6, 1.0, 0.2),
            'learning_rate': [0.05, 0.1],
            'max_leaves': range(100, 450, 50),
            'min_child_weight': [1],
            'n_estimators': [4000], # 3500
            'subsample': np.arange(0.6, 1.2, 0.2)
        }
        # 'max_leaves': range(2, 10, 1) # N terminal nodes
        # 'max_depth': range(2, 6, 1)


        # Process dataframe
        x = dfTrain.drop(columns='activity').values
        y = np.log1p(dfTrain['activity'].values)

        # Parameters: Slitting the dataset
        self.activityQuantile1 = (100 - self.selectSubsTopPercent) / 100
        self.activityQuantile2 = self.selectSubsBottomPercent / 100
        print(f'Selection Quantiles:\n'
              f'     1: {red}{self.activityQuantile1}{resetColor}\n'
              f'     2: {red}{self.activityQuantile2}{resetColor}\n')

        # Parameters: Saving the model
        pathModelH = pathModel.replace('Embeddings', f'{datasetTagHigh} - Embeddings')
        pathModelH = pathModelH.replace(' %', '')
        # pathModelH = pathModelH.replace('Activity Scores: ', '')
        pathModelM = pathModel.replace('Embeddings', f'{datasetTagMid} - Embeddings')
        pathModelM = pathModelM.replace(' %', '')
        pathModel = pathModel.replace('Embeddings', f'{datasetTagLow} - Embeddings')
        pathModel = pathModel.replace(' %', '')
        modelPaths = {
            datasetTagHigh: pathModelH,
            datasetTagMid: pathModelM,
            datasetTagLow: pathModel
        }

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


        pd.set_option('display.max_columns', 10)
        pd.set_option('display.max_rows', 10)
        print(f'Selecting Top {red}{self.selectSubsTopPercent} %'
              f'{resetColor} of Substrates')
        print(f'Sorted ESM Embeddings: {pink}{datasetTagHigh}{resetColor}\n'
              f'{dfHigh.sort_values(by='activity', ascending=False)}\n\n')
        print(f'Sorted ESM Embeddings: {pink}Mids{resetColor}\n'
              f'{dfMid.sort_values(by='activity', ascending=False)}\n\n')
        print(f'Sorted ESM Embeddings: {pink}{datasetTagLow}{resetColor}\n'
              f'{dfLow.sort_values(by='activity', ascending=False)}\n\n')
        pd.set_option('display.max_columns', None)

        # Split datasets
        # xTraining, xTesting, yTraining, yTesting = train_test_split(
        #     x, y, test_size=testSize, random_state=19)
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


        print(f'Training Data: {pink}{datasetTagHigh}{resetColor}\n'
              f'Splitting Training Set: '
              f'{yellow}{round((1 - testSize) * 100, 0)}{pink}:{yellow}'
              f'{round(testSize * 100, 0)}{resetColor}\n'
              f'     Train: {yellow}{xTrainingH.shape}{resetColor}\n'
              f'     Test: {yellow}{xTestingH.shape}{resetColor}\n')
        print(f'Training Data: {pink}{datasetTagMid}{resetColor}\n'
              f'Splitting Training Set: '
              f'{yellow}{round((1 - testSize) * 100, 0)}{pink}:{yellow}'
              f'{round(testSize * 100, 0)}{resetColor}\n'
              f'     Train: {yellow}{xTrainingM.shape}{resetColor}\n'
              f'     Test: {yellow}{xTestingM.shape}{resetColor}\n')
        print(f'Training Data: {pink}{datasetTagLow}{resetColor}\n'
              f'Splitting Training Set: '
              f'{yellow}{round((1 - testSize) * 100, 0)}{pink}:{yellow}'
              f'{round(testSize * 100, 0)}{resetColor}\n'
              f'     Train: {yellow}{xTrainingL.shape}{resetColor}\n'
              f'     Test: {yellow}{xTestingL.shape}{resetColor}\n')
        print(f'Unique Substrates: {red}{dfTrain.shape[0]:,}{resetColor}\n')


        def getLayerNumber(col):
            return int(col.replace("ESM Layer ", ""))

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
                          f'ESM Layer: {yellow}{self.layerESM}{resetColor}\n')

                # Train the model
                start = time.time()
                if tagData == datasetTagHigh:
                    model.fit(xTrain, yTrain, eval_set=[(xTest, yTest)],
                              early_stopping_rounds=50, verbose=False,
                              sample_weight=yTrain)
                else:
                    model.fit(xTrain, yTrain,eval_set=[(xTest, yTest)],
                              early_stopping_rounds=50, verbose=False)
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

                # Inspect results
                saveModel = False
                if (self.layerESMTag not in self.modelAccuracy[tagData].columns or
                        (accuracy.loc[indexEvalMetric, self.layerESMTag] <
                         self.modelAccuracy[tagData].loc[
                             indexEvalMetric, self.layerESMTag])):
                    saveModel = True
                    newColumn = self.layerESMTag not in self.modelAccuracy[tagData].columns

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
                    self.modelAccuracy[tagData] = self.modelAccuracy[tagData][self.sortedColumns]

                    # Save the data
                    self.modelAccuracy[tagData].to_csv(modelAccuracyPaths[tagData])
                    joblib.dump(model, modelPaths[tagData])

                if printData and lastModel:
                    print(f'Max Activity Score: {red}Max {self.maxValue:,}{resetColor}')
                    for dataset, predictions in self.predictionAccuracy.items():
                        print(f'Prediction Values: {pink}{dataset}{resetColor}\n'
                              f'{greenDark}{predictions}{resetColor}\n\n')

                    runtime = round((end - start), 3)
                    runtimeTotal = round((end - startTraining) / 60, 3)
                    rate = round(combination / runtimeTotal, 3)
                    if rate == 0:
                        timeRemaining = float('inf')
                    else:
                        timeRemaining = round((totalParamCombos - combination) / rate, 3)
                    for dataset, values in self.modelAccuracy.items():
                        print(f'Prediction Accuracy For Subset: {pink}{dataset}'
                              f'{resetColor}\n'
                              f'Hyperparameters: {greenLight}{self.bestParams[dataset]}')
                        print(f'{blue}{values}{resetColor}\n')
                    print(f'Time Training This Model: '
                          f'{red}{runtime:,} s{resetColor}\n'
                          f'Time Training All Models: '
                          f'{red}{runtimeTotal:,} min{resetColor}\n'
                          f'Training Rate: '
                          f'{red}{rate:,} combinations / min{resetColor}\n'
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
                params = dict(zip(paramNames, paramCombo))
                percentComplete = round((combination / totalParamCombos) * 100, 3)
                printData = (combination % 25 == 0)

                # Train Model
                tag = datasetTagHigh
                model, keepModel = trainModel(
                    model=XGBRegressor(
                        device=self.device, n_jobs=nJobs, eval_metric=evalMetric,
                        tree_method="hist", random_state=42, max_bin=64, **params),
                    xTrain=xTrainingH, yTrain=yTrainingH,
                    xTest=xTestingH, yTest=yTestingH,
                    tagData=tag)
                if keepModel:
                    self.modelH = model

                # Train Model
                tag = datasetTagMid
                model, keepModel = trainModel(
                    model=XGBRegressor(
                        device=self.device, n_jobs=nJobs, eval_metric=evalMetric,
                        tree_method="hist", random_state=42, max_bin=64, **params),
                    xTrain=xTrainingM, yTrain=yTrainingM,
                    xTest=xTestingM, yTest=yTestingM,
                    tagData=tag)
                if keepModel:
                        self.modelM = model

                # Train Model
                tag = datasetTagLow
                model, keepModel = trainModel(
                    model=XGBRegressor(
                        device=self.device, n_jobs=nJobs, eval_metric=evalMetric,
                        tree_method="hist", random_state=42, max_bin=64, **params),
                    xTrain=xTrainingL, yTrain=yTrainingL,
                    xTest=xTestingL, yTest=yTestingL,
                    tagData=tag, lastModel=True)
                if keepModel:
                    self.modelL = model


            # # End Training
            print('========================================'
                  '=========================================\n')
            print('                       ===================================\n')
            print('=============================== Training Results '
                  '================================')
            end = time.time()
            runtimeTotal = round((end - startTraining) / 60, 3)
            rate = round(totalParamCombos / runtimeTotal, 3)
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


            # Plot the data
            fig, ax = plt.subplots(figsize=self.figSize)
            plt.scatter(x, y, alpha=0.7, color='#BF5700', edgecolors='#F8971F', s=50)
            plt.plot(axisLimits, axisLimits, color='#101010', lw=self.lineThickness)
            plt.xlabel('Experimental Activity', fontsize=self.labelSizeAxis)
            plt.ylabel('Predicted Activity', fontsize=self.labelSizeAxis)
            plt.title(f'{tag}\nRandon Forest Regressor Accuracy\n'
                      f'{self.layerESMTag}',
                      fontsize=self.labelSizeTitle, fontweight='bold')
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
            figLabel = figLabel.replace('Params', f'Params Layer {self.layerESM}')
            saveLocation = os.path.join(self.pathFigures, figLabel)

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
        print(f'     Runtime: {red}{round(runtime, 3):,} ms{resetColor}\n')

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
            print(f'     {substrate}: {red}{round(value, 3):,}{resetColor}')
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
                          f'{red}{round(activity, 3):,}{resetColor}')
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
    def __init__(self, enzymeName, datasetTag, folderPath, subsTrain,
                 subsPercentSelectTop, subsPercentSelectBottom,
                 maxTrainingScore, subsPred, subsPredChosen, useEF, tagChosenSubs,
                 minSubCount, minES, modelType, concatESM, layersESM, testSize, batchSize,
                 labelsXAxis, printNumber, modelSize=2):
        # Parameters: Files
        self.pathFolder = folderPath
        self.pathData = os.path.join(self.pathFolder, 'Data')
        self.pathEmbeddings = os.path.join(self.pathFolder, 'Embeddings')
        self.pathModels = os.path.join(self.pathFolder, 'Models')
        self.pathFigures = os.path.join(self.pathFolder, 'Figures')
        os.makedirs(self.pathData, exist_ok=True)
        os.makedirs(self.pathEmbeddings, exist_ok=True)
        os.makedirs(self.pathModels, exist_ok=True)

        # Parameters: Dataset
        self.enzymeName = enzymeName
        self.datasetTag = datasetTag
        self.subsInitial = None
        self.subsTrain = subsTrain
        self.subsPercentSelectTop = subsPercentSelectTop
        self.subsPercentSelectBottom = subsPercentSelectBottom
        self.maxTrainingScore = maxTrainingScore
        self.subsTrainN = len(self.subsTrain.keys())
        self.subsPred = subsPred
        self.subsPredN = len(subsPred)
        self.subsPredChosen = subsPredChosen
        self.useEF = useEF
        self.tagChosenSubs = tagChosenSubs
        self.minSubCount = minSubCount
        self.minES = minES
        self.labelsXAxis = labelsXAxis
        self.printNumber = printNumber
        self.predictions = {}

        # Parameters: Model
        self.modelType = modelType
        self.subsetTagHigh = f'Top {self.subsPercentSelectTop} %'
        self.subsetTagMid = f'Mids'
        self.subsetTagLow = f'Bottom {self.subsPercentSelectBottom} %'
        accuracyDF = pd.DataFrame(0.0, index=['MAE','MSE','R²'], columns=[])
        self.modelAccuracy = {
            self.subsetTagHigh: accuracyDF.copy(),
            self.subsetTagMid: accuracyDF.copy(),
            self.subsetTagLow: accuracyDF.copy(),
        }
        self.concatESM = concatESM
        self.layersESM = layersESM
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
        self.embeddingsTrainTag = (
            f'Embeddings - ESM {self.sizeESM} - '
            f'Batch {self.batchSize} - {self.enzymeName} - '
            f'{self.datasetTag} - MinCounts {self.minSubCount} - '
            f'N {self.subsTrainN} - {len(self.labelsXAxis)} AA')
        self.embeddingsPredTag = (
            f'Embeddings - ESM {self.sizeESM} - '
            f'Batch {self.batchSize} - {self.enzymeName} - '
            f'Predictions - Min ES {self.minES} - MinCounts {self.minSubCount} - '
            f'N {self.subsPredN} - {len(self.labelsXAxis)} AA')
        if (self.concatESM
                and isinstance(self.layersESM, list)
                and len(self.layersESM) > 1):
            for layer in self.layersESM:
                filePath = f'{self.embeddingsTrainTag.replace(
                    'ESM', f'ESM L{layer}')}.csv'
                self.embeddingsPathTrain.append(
                    os.path.join(self.pathEmbeddings, filePath))
        else:
            self.concatESM = False
            if isinstance(self.layersESM, int):
                self.layersESM = [self.layersESM]
            filePath = f'{self.embeddingsTrainTag.replace(
            'ESM', f'ESM L{self.layersESM[0]}')}.csv'
            self.embeddingsPathTrain.append(
                    os.path.join(self.pathEmbeddings, filePath))
        for layer in self.layersESM:
            filePath = f'{self.embeddingsPredTag.replace(
                'ESM', f'ESM L{layer}')}.csv'
            self.embeddingsPathPred.append(
                    os.path.join(self.pathEmbeddings, filePath))
        if self.tagChosenSubs != '':
            self.embeddingsPredTag = self.embeddingsPredTag.replace(
                f'MinCounts {self.minSubCount}',
                f'MinCounts {self.minSubCount} - Added {self.tagChosenSubs}')
        if self.useEF:
            self.embeddingsTrainTag = self.embeddingsTrainTag.replace(
                'MinCounts', 'Scores EF - MinCounts')
            self.embeddingsPredTag = self.embeddingsPredTag.replace(
                'MinCounts', 'Scores EF - MinCounts')
        else:
            self.embeddingsTrainTag = self.embeddingsTrainTag.replace(
                'MinCounts', 'Scores Counts - MinCounts')
            self.embeddingsPredTag = self.embeddingsPredTag.replace(
                'MinCounts', 'Scores Counts - MinCounts')


        # Parameters: Save Paths Model Accuracy
        self.tagExperiment = (f'Model Accuracy - {modelType} - ESM {self.sizeESM} '
                              f' Layer {self.layersESM} - {enzymeName} - {datasetTag} - '
                              f'MinCounts {minSubCount}')
        self.tagExperiment = self.tagExperiment.replace(':', '')
        if self.useEF:
            self.tagExperiment = self.tagExperiment.replace(
                'MinCounts', 'Scores EF - MinCounts')
        else:
            self.tagExperiment = self.tagExperiment.replace(
                'MinCounts', 'Scores Counts - MinCounts')
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
                    f'{self.embeddingsTrainTag}')
        # modelTag = (f'Random Forest - Test Size {self.testingSetSize} - '
        #             f'N Trees {self.NTrees} - {self.embeddingsTrainTag}')
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
                  f'ESM layer {red}{layerESM}{resetColor} cannot be extracted\n\n')
            sys.exit()

        # Select a model to train
        if self.modelType == 'Random Forest Regressor: Scikit-Learn':
            # Model: Scikit-Learn Random Forest Regressor
            RandomForestRegressor = RandomForestRegressor(
                dfTrain=self.embeddingsSubsTrain, dfPred=self.embeddingsSubsPred,
                subsPredChosen=self.subsPredChosen, minES=self.minES,
                pathModel=pathModelScikit, modelTag=modelTagScikit, layerESM=layerESM,
                testSize=self.testingSetSize, device=self.device,
                printNumber=self.printNumber)
            self.modelAccuracy = randomForestRegressorXGB.modelAccuracy
            self.predictions[self.modelType] = RandomForestRegressor.predictions
        elif self.modelType == 'Random Forest Regressor: XGBoost':
            # Model: XGBoost Random Forest
            randomForestRegressorXGB = RandomForestRegressorXGB(
                dfTrain=self.embeddingsSubsTrain, dfPred=self.embeddingsSubsPred,
                selectSubsTopPercent=self.subsPercentSelectTop,
                selectSubsBottomPercent=self.subsPercentSelectBottom,
                tagExperiment=self.tagExperiment, maxValue=self.maxTrainingScore,
                pathModel=pathModelXGBoost, modelTag=modelTagXGBoost,
                modelAccuracy=self.modelAccuracy,
                modelAccuracyPaths=self.pathModelAccuracy,
                pathFigures=self.pathFigures, datasetTagHigh=self.subsetTagHigh,
                datasetTagMid=self.subsetTagMid, datasetTagLow=self.subsetTagLow,
                layerESM=layerESM, testSize=self.testingSetSize, device=self.device)

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
        missingLayersESM = []

        # Inspect: Data type
        predictions = True
        if trainingSet:
            predictions = False
        print(f'ESM Layers: {yellow}{self.layersESM}{resetColor}\n'
              f'Total unique substrates: {red}{len(substrates):,}{resetColor}')
        print(f'Concatenating ESM Layers: {purple}{self.concatESM}{resetColor}\n')
        sequenceEmbeddings = []

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


            batchTotal = len(batchTokens)
            allEmbeddings = []
            allValues = []
            startInit = time.time()


            # Generate embeddings
            for index, layerESM in enumerate(layersESM):
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
                            rate = round(i / runtimeTotal, 3)
                            if rate == 0:
                                timeRemaining = float('inf')
                            else:
                                timeRemaining = round((batchTotal - i) / rate, 3)
                            print(f'ESM Progress (Layer {yellow}{layerESM}{resetColor}): '
                                  f'{red}{i:,}{resetColor} / {red}{batchTotal:,}'
                                  f'{resetColor} '
                                  f'({red}{percentCompletion} %{resetColor})\n'
                                  f'     Batch Shape: {greenLight}{batch.shape}'
                                  f'{resetColor}\n'
                                  f'     Runtime: {red}{round(runtime, 3):,} s'
                                  f'{resetColor}\n'
                                  f'     Total Time: {red}{round(runtimeTotal, 3):,} min'
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
                print(f'ESM Progress (Layer {yellow}{layerESM}{resetColor}): '
                      f'{red}{batchTotal:,}{resetColor} / {red}{batchTotal:,}'
                      f'{resetColor} ({red}{percentCompletion} %{resetColor})\n'
                      f'     Runtime: {red}{round(runtime, 3):,} s'
                      f'{resetColor}\n'
                      f'     Total Time: {red}{round(runtimeTotal, 3):,} min'
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
                    columns = [f'feat_L{layerESM}_{i}'
                               for i in range(embeddings.shape[1])]
                else:
                    values = np.array(allValues).reshape(-1, 1)
                    data = np.hstack([embeddings, values])
                    columns = [f'feat_L{layerESM}_{i}'
                               for i in range(embeddings.shape[1])] + ['activity']

                # Process Embeddings
                subEmbeddings = pd.DataFrame(data, index=batchSubs, columns=columns)
                pd.set_option('display.max_columns', 10)
                pd.set_option('display.max_rows', 10)
                print(f'Substrate Embeddings:\n{subEmbeddings}\n\n')
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)
                print(f'Substrate Embeddings shape: '
                      f'{pink}{sequenceEmbeddings.shape}{resetColor}\n\n')
                print(f'Embeddings saved at:\n'
                      f'     {greenDark}{pathEmbeddings}{resetColor}\n\n')
                subEmbeddings.to_csv(pathEmbeddings)

                # plt.hist(subEmbeddings.loc[:, 'activity'], bins=100)
                # plt.title("Activity Distribution")
                # plt.show()

            return subEmbeddings


        def loadESM():
            missingFiles = []

            # Load: ESM Embeddings
            genEmbedding = False
            for index, pathEmbeddings in enumerate(filePaths):
                if not os.path.exists(pathEmbeddings):
                    genEmbedding = True
                    layer = self.layersESM[index]
                    missingFiles.append(pathEmbeddings)
                    missingLayersESM.append(layer)

            if genEmbedding:
                generateEmbeddingsESM(seqs=substrates, layersESM=missingLayersESM,
                                      savePaths=missingFiles)
            else:
                for index, pathEmbeddings in enumerate(filePaths):
                    print(f'Loading: ESM Embeddings\n'
                          f'     {greenDark}{pathEmbeddings}{resetColor}\n')
                # subEmbeddings = pd.read_csv(pathEmbeddings, index_col=0)
                # print(f'Substrate Embeddings shape: '
                #       f'{pink}{subEmbeddings.shape}{resetColor}\n\n')

            sys.exit()

        # Get: ESM data
        subEmbeddings = loadESM()



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

