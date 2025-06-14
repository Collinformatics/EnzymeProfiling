from functions import getFileNames, NGS, PredictActivity
import pandas as pd
import sys


# Evolutionary Scale Modeling (ESM)
# | **Layer** | **Properties**                                                           |
# | --------- | ------------------------------------------------------------------------ |
# | **6–8**   | Early layers start forming **secondary structure** and local motifs.     |
# |           |                                                                          |
# | **12–16** | Middle layers capture **functional signals** (useful for contact         |
# |           | prediction and mutational effect predictions). Often sweet spot.         |
# |           |                                                                          |
# | **24–30** | Higher layers have stronger **semantic representation**, helpful for     |
# |           | **homology or global fold-related features**.                            |
# |           |                                                                          |
# | **33–36** | Final layers often focus on **language model objectives**                |
#             | (less task-specific); can be noisy for regression/classification.        |
#             | Sometimes still useful.                                                  |
# |           |                                                                          |

# | **Layer** | **Characteristic**                  | **Useful For**                     |
# | --------- | ----------------------------------- | ---------------------------------- |
# | **0**     | Raw AA encoding, no context         | Baseline control,                  |
# |           |                                     | uninformative alone                |
# |           |                                     |                                    |
# | **1–5**   | Captures local amino acid motifs    | **Disorder**, **short motifs**,    |
# |           |                                     | **early SS features**              |
# |           |                                     |                                    |
# | **6–12**  | Builds regional context             | **Solvent accessibility**,         |
# |           |                                     | **loop dynamics**                  |
# |           |                                     |                                    |
# | **13–20** | Intermediate structure and function | **Activity prediction**,           |
#             |                                     | **region-specific scoring**        |
# |           |                                     |                                    |
# | **21–30** | Transition into semantic/           | **Binding site predictions**,      |
# |           | structural representations          | **fold matching**                  |
# |           |                                     |                                    |
# | **31–36** | High-level semantic and global      | **Function annotation**,           |
# |           | fold recognition                    | **homolog detection**,             |
# |           |                                     | **global fold classification**     |
# |           |                                     |                                    |


# Averaging:
#       Mean of layers 26–34: empirically shown to be strong for predictive tasks.
#       embedding = torch.stack([layer26, ..., layer34]).mean(0)
#
#           meanEmbedding = torch.stack(hidden_states[26:35]).mean(dim=0)



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
inModelSize = 1
inUseFilteredReadingFrame = False
inPlotOnlyWords = True
inFixedResidue = ['Q']
inFixedPosition = [4]
inExcludeResidues = True
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inMinimumSubstrateCount = 10
inUseEnrichmentFactor = True

# Input 4: Machine Learning
inModelTypes = ['Random Forest Regressor: Scikit-Learn',
                'Random Forest Regressor: XGBoost']
inModelType = inModelTypes[1]
inLayersESM = [16, 14, 12, 8, 5] # [36, 30, 25, 20, 15, 10, 5]
inTestSize = 0.2
inESMBatchSizes = [4096, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
inESMBatchSize = inESMBatchSizes[5]
inMinES = 0 # Minimum ES for randomized substrates
inSubsPred = {
    'Dennis': ['CLLQARFS', 'VLLQGFVH', 'AKLQGDFH', 'VHLQCSIH', 'TLLQACVG', 'IRLQCGIM']}
inGeneratedSubsFilter = { # Restrictions for generated substrates
    'R3': ['L'],
    'R4': ['Q']
}

# Input 5: Figures
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

# Input 6: Processing The data
inPrintNumber = 10

# Input 7: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False

# Input 8: Plot Sequence Motif
inNormLetters = False  # Normalize fixed letter heights
inPlotWeblogoMotif = False
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inPlotNegativeWeblogoMotif = False
inBigLettersOnTop = False

# Input 9: Motif Enrichment
inPlotNBars = 50

# Input 10: Word Cloud
inLimitWords = True
inTotalWords = inPlotNBars




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
enzymeName, filesInitial, filesFinal, labelAAPos = getFileNames(enzyme=inEnzymeName)
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
          numPCs=None, NSubsPCA=None, plotSuffixTree=False, saveFigures=inSaveFigures,
          setFigureTimer=inSetFigureTimer)



# ====================================== Load data =======================================
# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort')

# Calculate: RF
probInitialAvg = ngs.calculateProbabilities(counts=countsInitial, N=countsInitialTotal,
                                            fileType='Initial Sort', calcAvg=True)

# Load: Substrates
substratesInitial, totalSubsInitial = ngs.loadUnfilteredSubs(loadInitial=True)


# Get dataset tag
ngs.getDatasetTag(combinedMotifs=inUseFilteredReadingFrame)
print(f'Tag: {purple}{ngs.datasetTag}{resetColor}')

paths = ngs.getFilePath(datasetTag=ngs.datasetTag, motifPath=inUseFilteredReadingFrame)
if inUseFilteredReadingFrame:
    pathSubstrates, pathSubstrateCounts, _ = paths

    # Load: Filtered reading frame
    motifs, motifsCountsTotal, substratesFiltered = ngs.loadMotifSeqs(
        motifLabel=inMotifPositions, motifIndex=motifFramePos)
else:
    pathSubstrates, pathSubstrateCounts = paths

    # Load: Substrates
    motifs, motifsCountsTotal = ngs.loadSubstratesFiltered()
    substratesFiltered = motifs



# ===================================== Run The Code =====================================
# Display current sample size
ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=motifsCountsTotal,
                     NFinalUnique=len(motifs.keys()))


# # Evaluate: Motif Sequences
# Count fixed substrates
motifCountsFinal, motifsCountsTotal = ngs.countResidues(substrates=motifs,
                                                        datasetType='Final Sort')

# Calculate: Prob
probMotif = ngs.calculateProbabilities(counts=motifCountsFinal, N=motifsCountsTotal,
                                       fileType='Final Sort')

# Calculate: Positional entropy
ngs.calculateEntropy(probability=probMotif, combinedMotifs=inUseFilteredReadingFrame)

# Calculate: AA Enrichment
ngs.calculateEnrichment(probInitial=probInitialAvg, probFinal=probMotif,
                        combinedMotifs=inUseFilteredReadingFrame)

if inUseEnrichmentFactor:
    # Evaluate: Sequences
    subsTrain = ngs.processSubstrates(
        subsInit=substratesInitial, subsFinal=substratesFiltered, motifs=motifs,
        subLabel=inMotifPositions, plotEF=inUseEnrichmentFactor,
        combinedMotifs=inUseFilteredReadingFrame)
else:
    subsTrain = motifs

# # Predicting Substrate Activity
# Generate: Prediction substrates
substratesPred, tagPredSubs = ngs.generateSubstrates(
    df=probMotif, eMap=ngs.eMap, minES=inMinES, dataType='AA Probabilities',
    subsReq=inSubsPred, filter=inGeneratedSubsFilter)

# Set option
pd.set_option('display.max_rows', 10)

# Predict activity
predictions = PredictActivity(
    enzymeName=enzymeName, folderPath=inPathFolder, datasetTag=ngs.datasetTag,
    subsTrain=subsTrain, subsPred=substratesPred, subsPredChosen=inSubsPred,
    useEF=inUseEnrichmentFactor, tagChosenSubs=tagPredSubs,
    minSubCount=inMinimumSubstrateCount, layersESM=inLayersESM, minES=inMinES,
    modelType=inModelType, testSize=inTestSize, batchSize=inESMBatchSize,
    labelsXAxis=inMotifPositions, printNumber=inPrintNumber, modelSize=inModelSize)
predictions.trainModel()

# # Evaluate: Predictions
# ngs.processSubstrates(
#     subsInit=None, subsFinal=None, motifs=predictions.predictions,
#     subLabel=inMotifPositions, predActivity=True)

