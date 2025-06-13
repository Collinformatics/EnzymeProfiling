from functions import filePaths, NGS, PredictActivity
import pandas as pd
import sys



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inPathFolder = f'{inEnzymeName}'
inSaveFigures = True
inSetFigureTimer = False

# Input 2: Experimental Parameters
# inMotifPositions = ['-2', '-1', '0', '1', '2', '3']
inMotifPositions = ['P4', 'P3', 'P2', 'P1', 'P1\'', 'P2\'', 'P3\'', 'P4\'']  #
inIndexNTerminus = 0  # Define the index if the first AA in the binned substrate

# Input 3: Computational Parameters
inPlotOnlyWords = True
inFixedResidue = ['Q']
inFixedPosition = [4]
inExcludeResidues = False
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inMinimumSubstrateCount = 5000

# Input 4: Machine Learning
inModelTypes = ['Scikit-Learn: Random Forest Regressor',
                'XGBoost: Random Forest Regressor']
inModelType = inModelTypes[1]
inLayersESM = [5,10,15,20]
inTestSize = 0.2
inBatchSize = 4096 # Batch size for ESM
inMinES = 0
inSubsPred = {
    'Dennis': ['CLLQARFS', 'VLLQGFVH', 'AKLQGDFH', 'VHLQCSIH', 'TLLQACVG', 'IRLQCGIM']}
inGeneratedSubsFilter = { # Restrictions for generated substrates
    'R3': ['L'],
    'R4': ['Q']
}

# Input 4: Figures
inPlotEntropy = True
inPlotEnrichmentMap = True
inPlotEnrichmentMapScaled = False
inPlotLogo = True
inPlotWeblogo = True
inPlotMotifEnrichment = True
inPlotMotifEnrichmentNBars = True
inPlotWordCloud = True
if inPlotOnlyWords:
    inPlotEntropy = False
    inPlotEnrichmentMap = False
    inPlotEnrichmentMapScaled = False
    inPlotLogo = False
    inPlotWeblogo = False
    inPlotMotifEnrichment = False
    inPlotMotifEnrichmentNBars = False
    inPlotWordCloud = True
inPlotWordCloud = False # <--------------------

inPlotBarGraphs = False
inPlotPCA = False  # PCA plot of the combined set of motifs
inShowSampleSize = True  # Include the sample size in your figures

# Input 5: Processing The data
inPrintNumber = 10

# Input 6: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False

# Input 7: Plot Sequence Motif
inNormLetters = False  # Normalize fixed letter heights
inPlotWeblogoMotif = False
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inPlotNegativeWeblogoMotif = False
inBigLettersOnTop = False

# Input 8: Motif Enrichment
inPlotNBars = 50

# Input 9: Word Cloud
inLimitWords = True
inTotalWords = inPlotNBars

# Input 10: PCA
inNumberOfPCs = 2
inTotalSubsPCA = int(5 * 10 ** 4)
inIncludeSubCountsESM = True
inPlotEntropyPCAPopulations = False



# ==================================== Set Parameters ====================================
# Print options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)

# Colors:
white = '\033[38;2;255;255;255m'
greyDark = '\033[38;2;144;144;144m'
purple = '\033[38;2;189;22;255m'
magenta = '\033[38;2;255;0;128m'
pink = '\033[38;2;255;0;242m'
cyan = '\033[38;2;22;255;212m'
green = '\033[38;2;5;232;49m'
greenLight = '\033[38;2;204;255;188m'
greenDark = '\033[38;2;30;121;13m'
yellow = '\033[38;2;255;217;24m'
orange = '\033[38;2;247;151;31m'
red = '\033[91m'
resetColor = '\033[0m'

# Load: Dataset labels
enzymeName, filesInitial, filesFinal, labelAAPos = filePaths(enzyme=inEnzymeName)
motifLen = len(inMotifPositions)
motifFramePos = [inIndexNTerminus, inIndexNTerminus + motifLen]



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=enzymeName, substrateLength=len(labelAAPos),
          filterSubs=True, fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          minEntropy=None, figEMSquares=inShowEnrichmentAsSquares, xAxisLabels=labelAAPos,
          xAxisLabelsMotif=inMotifPositions, printNumber=inPrintNumber,
          showNValues=inShowSampleSize, bigAAonTop=inBigLettersOnTop, findMotif=False,
          folderPath=inPathFolder, filesInit=filesInitial, filesFinal=filesFinal,
          plotPosS=inPlotEntropy, plotFigEM=inPlotEnrichmentMap,
          plotFigEMScaled=inPlotEnrichmentMapScaled, plotFigLogo=inPlotLogo,
          plotFigWebLogo=inPlotWeblogo, plotFigMotifEnrich=inPlotMotifEnrichment,
          plotFigMotifEnrichSelect=inPlotMotifEnrichmentNBars,
          plotFigWords=inPlotWordCloud, wordLimit=inLimitWords, wordsTotal=inTotalWords,
          plotFigBars=inPlotBarGraphs, NSubBars=inPlotNBars, plotFigPCA=inPlotPCA,
          numPCs=inNumberOfPCs, NSubsPCA=inTotalSubsPCA, plotSuffixTree=False,
          saveFigures=inSaveFigures, setFigureTimer=inSetFigureTimer)



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



# ====================================== Load data =======================================
# Set param
combinedMotifs = True

# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort')

# Calculate: RF
probInitialAvg = ngs.calculateProbabilities(counts=countsInitial, N=countsInitialTotal,
                                            fileType='Initial Sort', calcAvg=True)

# Load: Substrates
substratesInitial, totalSubsInitial = ngs.loadUnfilteredSubs(loadInitial=True)

# Get dataset tag
ngs.getDatasetTag(combinedMotifs=combinedMotifs)

# Load: Filtered reading frame
motifs, motifsCountsTotal, substratesFiltered = ngs.loadMotifSeqs(
    motifLabel=inMotifPositions, motifIndex=motifFramePos)



# ===================================== Run The Code =====================================
# Display current sample size
ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=motifsCountsTotal)


# # Evaluate: Motif Sequences
# Count fixed substrates
motifCountsFinal, motifsCountsTotal = ngs.countResidues(substrates=motifs,
                                                        datasetType='Final Sort')

# Calculate: Prob
probMotif = ngs.calculateProbabilities(counts=motifCountsFinal, N=motifsCountsTotal,
                                       fileType='Final Sort')

# Calculate: Positional entropy
ngs.calculateEntropy(probability=probMotif, combinedMotifs=combinedMotifs)

# Calculate: AA Enrichment
ngs.calculateEnrichment(probInitial=probInitialAvg, probFinal=probMotif,
                        combinedMotifs=combinedMotifs)

# Evaluate: Sequences
subsTrain = ngs.processSubstrates(
    subsInit=substratesInitial, subsFinal=substratesFiltered, motifs=motifs,
    subLabel=inMotifPositions, combinedMotifs=combinedMotifs)


# # Predicting Substrate Activity
# Generate: Prediction substrates
substratesPred, tagPredSubs = ngs.generateSubstrates(
    df=probMotif, eMap=ngs.eMap, minES=inMinES, dataType='AA Probabilities',
    subsReq=inSubsPred, filter=inGeneratedSubsFilter)


bestLayer = None
# Predict activity
predictions = PredictActivity(
    enzymeName=enzymeName, folderPath=inPathFolder, datasetTag=ngs.datasetTag,
    subsTrain=subsTrain, subsPred=substratesPred, subsPredChosen=inSubsPred,
    tagChosenSubs=tagPredSubs, minSubCount=inMinimumSubstrateCount,
    layersESM=inLayersESM, minES=inMinES, testSize=inTestSize, batchSize=inBatchSize,
    labelsXAxis=inMotifPositions, printNumber=inPrintNumber)

predictions.trainModel(modelType=inModelType)
print(f'Prediction Accuracies:\n'
      f'{red}{predictions.modelAccuracy}{resetColor}\n\n')
sys.exit()

# Evaluate: Predictions
ngs.processSubstrates(
    subsInit=None, subsFinal=None, motifs=predictions.predictions,
    subLabel=inMotifPositions, predActivity=True)

