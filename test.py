from functions import filePaths, NGS, PredictActivity
import pandas as pd
import sys
from xgboost import XGBRegressor, XGBRFRegressor, DMatrix



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
inMinimumSubstrateCount = 10

# Input 4: Figures
# inPlotPCA = False # PCA plot of an individual fixed frame
# inPlotPCACombined = True
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

# Input 11: Printing The data
inPrintLoadedSubs = True
inPrintSampleSize = True
inPrintCounts = True
inPrintRF = True
inPrintES = True
inPrintEntropy = True
inPrintMotifData = True
inPrintNumber = 10
inCodonSequence = 'NNS'  # Base probabilities of degenerate codons (can be N, S, or K)
inUseCodonProb = False  # If True: use "inCodonSequence" for baseline probabilities



# ==================================== Set Parameters ====================================
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

# Print options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)

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



# ====================================== Load data =======================================
# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort')

# Calculate: RF
probInitialAvg = ngs.calculateProbabilities(counts=countsInitial, N=countsInitialTotal,
                                            fileType='Initial Sort', calcAvg=True)

# Load: Substrates
substratesInitial, totalSubsInitial = ngs.loadUnfilteredSubs(loadInitial=True)

# Set param
combinedMotifs = True

# Get dataset tag
ngs.getDatasetTag(combinedMotifs=combinedMotifs)

# Load: Filtered reading frame
motifs, motifsCountsTotal, substratesFiltered = ngs.loadMotifSeqs(
    motifLabel=inMotifPositions, motifIndex=motifFramePos)

# Display current sample size
ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=motifsCountsTotal)

# Evaluate: Sequences
ngs.processSubstrates(subsInit=substratesInitial, subsFinal=substratesFiltered,
                      motifs=motifs, subLabel=inMotifPositions,
                      combinedMotifs=combinedMotifs)

# Calculate: Prob
probMotif = ngs.calculateProbabilities(counts=motifCountsFinal, N=motifsCountsTotal,
                                       fileType='Final Sort')

# Calculate: Positional entropy
ngs.calculateEntropy(probability=probMotif, combinedMotifs=combinedMotifs)

# Calculate: AA Enrichment
ngs.calculateEnrichment(probInitial=probInitialAvg, probFinal=probMotif,
                        combinedMotifs=combinedMotifs)

ngs.processSubstrates(subsInit=substratesInitial, subsFinal=substratesFiltered,
                      motifs=motifs, subLabel=inMotifPositions,
                      combinedMotifs=combinedMotifs)
sys.exit()



# ===================================== Run The Code =====================================
PredictActivity(enzymeName=enzymeName, datasetTag=ngs.datasetTa,
                labelsXAxis=inMotifPositions,
                subsTrain=substrates, subsTest=substratesPred, printNumber=inPrintNumber)

sys.exit()

# Load: Substrate motifs
motifs, motifsCountsTotal, substratesFiltered = ngs.loadMotifSeqs(
    motifLabel=inMotifPositions, motifIndex=motifFramePos)

# Get dataset tag
ngs.getDatasetTag(combinedMotifs=True)

# Display current sample size
ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=motifsCountsTotal)

# Evaluate dataset
combinedMotifs = False
if len(ngs.motifIndexExtracted) > 1:
    combinedMotifs = True

# # Evaluate: Count Matrices
# Load: Motif counts
countsRelCombined, countsRelCombinedTotal = ngs.loadMotifCounts(
    motifLabel=inMotifPositions, motifIndex=motifFramePos)

# Calculate: RF
probCombinedReleasedMotif = ngs.calculateProbabilitiesCM(
    countsCombinedMotifs=countsRelCombined)

# Calculate: Positional entropy
ngs.calculateEntropy(probability=probCombinedReleasedMotif,
                     combinedMotifs=combinedMotifs,
                     releasedCounts=True)

# Calculate enrichment scores
ngs.calculateEnrichment(probInitial=probInitialAvg, probFinal=probCombinedReleasedMotif,
                        combinedMotifs=combinedMotifs, releasedCounts=True)

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

ngs.processSubstrates(subsInit=substratesInitial, subsFinal=substratesFiltered,
                      motifs=motifs, subLabel=inMotifPositions,
                      combinedMotifs=combinedMotifs)

sys.exit()