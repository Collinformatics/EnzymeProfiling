from functions import filePaths, NGS
import logomaker
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pk
import random
from sklearn.metrics import r2_score
import sys
import time
import threading



# ========================================== User Inputs =========================================
# Input 1: File Location
inEnzymeName = 'your name'
inBasePath = f'path/to/folder/{inEnzymeName}'
inFilePath = os.path.join(inBasePath, 'Extracted Data')
inSavePath = inFilePath
inSavePathFigures = os.path.join(inBasePath, 'Figures')
inFileNamesInitialSort, inFileNamesFinalSort, inAAPositions = (
    filePaths(enzyme=inEnzymeName))
inSaveFigures = True
inFigureTimer = False

# Input 2: Experimental Parameters
inSubstrateLength = len(inAAPositions)
inSaveScaledEnrichment = False # Saving Scaled Enrichment Values
# inFixedFramePositions = ['-2', '-1', '0', '1', '2', '3']
inFixedFramePositions = ['P4', 'P3', 'P2', 'P1', 'P1\'', 'P2\'']
inFixedFrameLength = len(inFixedFramePositions)
inIndexNTerminus = 0 # Define the index if the first AA in the binned substrate
inFramePositons = [inIndexNTerminus, inIndexNTerminus+inFixedFrameLength-1]
inAAPositionsBinned = inAAPositions[inFramePositons[0]:inIndexNTerminus+inFixedFrameLength]

# Input 3: Computational Parameters
inFixResidues = True # True: fix AAs in the substrate, False: Don't fix AAs, plot raw the data
inFixedResidue = ['L']
inFixedPosition = [5]
inMinimumSubstrateCount = 10
inPrintFixedSubs = True
inFigureTitleSize = 18
inFigureLabelSize = 16
inFigureTickSize = 13
inLineThickness = 1.5
inTickLength = 4

# Input 4: Processing The Data
inPlotEnrichmentMap = True
inPlotEnrichmentMotif = False
inPCAIndividual = False # PCA plot of an individual fixed frame
inPlotEnrichmentMapPCA = True
inPlotEnrichmentMotifPCA = inPlotEnrichmentMapPCA
inPlotWeblogoMotif = False
inPlotActivityFACS = False
inPredictSubstrateActivity = False
inPredictSubstrateActivityPCA = False
inPlotWordsPCA = True # Plot word cloud for the selected populations
inPlotBinnedSubstrates = True
inPlotBinnedSubstratePCA = True
inPlotBinnedSubstrateES = False
inPlotBinnedSubstratePrediction = False
inPlotCounts = False
inCountsColorMap = ['white', 'white', 'lightcoral', 'red', 'firebrick', 'darkred']
inStDevColorMap = ['white', 'white', '#FF76FA', '#FF50F9', '#FF00F2', '#CA00DF', '#BD16FF']
inShowSampleSize = False # Include the sample size in your figures

# Input 5: PCA
inNumberOfPCs = 2
inTotalSubsPCA = int(5*10**3)
inEncludeSubstrateCounts = False
inExtractPopulations = False
inPlotPositionalEntropyPCAPopulations = False
inAdjustZeroCounts = True # Prevent counts of 0 in PCA EM & Motif

# Input 6: Printing The Data
inPrintLoadedSubs = True
inPrintSampleSize = True
inPrintCounts = True
inPrintRF = True
inPrintES = True
inPrintEntropy = True
inPrintMotifData = True
inPrintNumber = 10
inCodonSequence = 'NNS' # Base probabilities of degenerate codons (can be N, S, or K)
inUseCodonProb = False # If True: use "inCodonSequence" for baseline probabilities

# Input 7: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False
inTitleEnrichmentMap = inEnzymeName
inYLabelEnrichmentMap = 2 # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter
inPrintSelectedSubstrates = 1 # Set = 1, to print substrates with fixed residue
inFigureSize = (9.5, 8) # (width, height)
inFigureBorders = [0.873, 0.075, 0.117, 0.967] # Top, bottom, left, right
inFigureAsSquares = (4.5, 8)
inFigureBordersAsSquares = [0.873, 0.075, 0.075, 0.943]
inEnrichmentColorMap = ['navy', 'royalblue', 'dodgerblue', 'lightskyblue', 'white', 'white',
                        'lightcoral', 'red', 'firebrick', 'darkred']

# Input 8: Plot Sequence Motif
inNormLetters = False # Normalize fixed letter heights
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inPlotNegativeWeblogoMotif = False
inTitleMotif = inTitleEnrichmentMap
inBigLettersOnTop = False
inFigureSizeMotif = inFigureSize
inFigureBordersMotifYTicks = [0.94, 0.075, 0.07, 0.98] # [top, bottom, left, right]
inFigureBordersMotifMaxYTick = [0.94, 0.075, 0.102, 0.98]
inFigureBordersNegativeWeblogoMotif = [0.94, 0.075, 0.078, 0.98]
inFigureBordersEnrichmentMotif = [0.94, 0.075, 0.138, 0.98]
inLetterColors = ['darkgreen', 'firebrick', 'deepskyblue', 'pink', 'navy', 'black', 'gold']
                  # Aliphatic, Acidic, Basic, Hydroxyl, AmMpro22, Aromatic, Sulfur

# Input 9: Evaluate Known Substrates
inNormalizePredictions = True
inYMaxPred = 1.05
inYMinPred, inYMinPredScaled, inYMinPredAI = 0, 0, -0.25
inYTickMinPred, inYTickMinScaled, inYTickMinAI = inYMinPred, inYMinPredScaled, -0.4
inSubsPredict = ['VVLQSGFR', 'VVLQSPFR', 'VYLQSGFR', 'VVLQAGFR', 'VVMQSGFR',
                 'IVLQSGFR', 'VVLHSGFR', 'VGLQSGFR', 'VVLMSGFR', 'VVVQSGFR',
                 'VVLQIGFR', 'VVGQSGFR', 'KVLQSGFR', 'VVLQNGFR', 'VVLYSGFR']
# inSubsPredict = ['VVLQSGFR', 'VVMQSGFR', 'VVVQSGFR', 'VVGQSGFR', 'VVLHSGFR', 'VVLMSGFR',
#                'VVLYSGFR', 'IVLQSGFR', 'KVLQSGFR', 'VYLQSGFR', 'VGLQSGFR', 'VVLQAGFR',
#                'VVLQNGFR', 'VVLQIGFR', 'VVLQSPFR']
               # 'AVLQSGFR', 'VTFQSAVK', 'ATVQSKMS', 'ATLQAIAS', 'VKLQNNEL', 'VRLQAGNA',
               # 'PMLQSADA', 'TVLQAVGA', 'ATLQAENV', 'TRLQSLEN', 'PKLQSSQA',
               # 'VTFQGKFK', 'PLMQSADA', 'PKLQASQA']
# lenSubsTotal = len(inSubsPredict)
# inSubsManual = ['VVLQSGFR', 'VVMQSGFR', 'VVVQSGFR', 'VVGQSGFR', 'VVLHSGFR', 'VVLMSGFR',
#                 'VVLYSGFR', 'IVLQSGFR', 'KVLQSGFR', 'VYLQSGFR', 'VGLQSGFR', 'VVLQAGFR',
#                 'VVLQNGFR', 'VVLQIGFR', 'VVLQSPFR'] # Double: VVLQSPFR
# inSubsCovid = ['AVLQSGFR', 'VTFQSAVK', 'ATVQSKMS', 'ATLQAIAS', 'VKLQNNEL', 'VRLQAGNA',
#                'PMLQSADA', 'TVLQAVGA', 'ATLQAENV', 'TRLQSLEN', 'PKLQSSQA']
# inSubsSARS = ['VTFQGKFK', 'PLMQSADA', 'PKLQASQA']
# inSubsPredict = inSubsManual
# inDatapointColor = ['#CC5500', '#CC5500', '#CC5500', '#CC5500', '#CC5500', '#CC5500',
#                     '#CC5500', '#CC5500', '#CC5500', '#CC5500', '#CC5500', '#CC5500',
#                     '#CC5500', '#CC5500', '#CC5500',
#                     'black', 'black', 'black', 'black', 'black', 'black',
#                     'black', 'black', 'black', 'black', 'black',
#                     '#F79620', '#F79620', '#F79620']
# inSubsPredict = ['VLLQCV', 'SRLQAS', 'VTLQSY', 'PILQSG', 'GWVQLH', 'GCILHA',
#                  'SRLQSG', 'VLLQCV', 'SVLQGF', 'TSLQAG', 'ALMQSG', 'VLLQAT'
#                  'VRLQSS', 'TILQGA', 'TFLQCR', 'SRLQAS', 'LVLQAH', 'SLLQGM'
#                  'VTLQSY', 'PILQSG', 'GWVQLH', 'VVLQAS']
# inSubsPredict = ['CKLQCL', 'SWLQSG', 'AMLQCH', 'VRLQNK', 'LKLQAC', 'PILQST',
#                  'VDLQAW', 'SILQVM', 'CNLQCL', 'FVLQCL', 'VRLQGW', 'LLLQAA',
#                  'GVLQAV', 'GVLQSH', 'QILQIE', 'VELQGA', 'IVLQCM', 'CRLQSG',
#                  'SKLQGV']
inSubsPredictStartIndex = 0
inKnownTarget = ['nsp4/5', 'nsp5/6', 'nsp6/7', 'nsp7/8', 'nsp8/9', 'nsp9/10',
                 'nsp10/12', 'nsp12/13', 'nsp13/14', 'nsp14/15', 'nsp15/16']
inBarWidth = 0.75
inBarColor = '#CC5500'
inEdgeColor = 'black'
inEdgeColorOrange = '#F79620'
inDatapointColor = []
for _ in inSubsPredict:
    inDatapointColor.append(inBarColor)

# Input 10: Evaluate Binned Substrates
inPlotEnrichedSubstrateFrame = False
inPrintLoadedFrames = True
inPlotBinnedSubNumber = 30
inPlotBinnedSubProb = True
inPlotBinnedSubYMax = 0.07

# Input 11: Predict Binned Substrate Enrichment
inEvaluatePredictions = False
inPrintPredictions = False
inBottomParam = 0.16
inPredictionDatapointColor = '#CC5500'
inMiniumSubstrateScoreLimit = False
inMiniumSubstrateScore = -55
inNormalizeValues = False
inPlotSubsetOfSubstrates = False
inPrintPredictionAccuracy = False
inInspectExperimentalES = True
inExperimentalESUpperLimit = 3.6
inExperimentalESLowerLimit = 3.0
inInspectPredictedES = False
inPredictedESUpperLimit = 3.5
inPredictedESLowerLimit = 2.5
inSetAxisLimits = False
inPlotSubstrateText = False
inTestBinnedSubES = True
inSaveBinnedSubES = False



# ======================================== Set Parameters ========================================
# Dataset label
if len(inFixedPosition) == 1:
    inSubstrateFrame = f'Fixed Frame {inFixedResidue[0]}@R{inFixedPosition[0]}'
else:
    inFixedPosition = sorted(inFixedPosition)
    continuous = True
    for index in range(len(inFixedPosition)-1):
        pos1, pos2 = inFixedPosition[index], inFixedPosition[index + 1]
        if pos1 == pos2 - 1 or pos1 == pos2 + 1:
            continue
        else:
            continuous = False
            break
    if continuous:
        inSubstrateFrame = (f'Fixed Frame {inFixedResidue[0]}@R{inFixedPosition[0]}-'
                            f'R{inFixedPosition[-1]}')
    else:
        inSubstrateFrame = (f'Fixed Frame {inFixedResidue[0]}@R{inFixedPosition[0]}-'
                            f'R{inFixedPosition[1]}, R{inFixedPosition[-1]}')

# Figure parameters
if inShowEnrichmentAsSquares:
    # Set figure dimension when plotting EM plots as squares
    figSizeEM = inFigureAsSquares
    figBordersEM = inFigureBordersAsSquares
else:
    # Set figure dimension when plotting EM plots as rectangles
    figSizeEM = inFigureSize
    figBordersEM = inFigureBorders

# Colors:
white = '\033[38;2;255;255;255m'
silver = '\033[38;2;204;204;204m'
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



# ======================================= Initialize Class =======================================
ngs = NGS(enzymeName=inEnzymeName, substrateLength=inSubstrateLength,
          fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          minCounts=inMinimumSubstrateCount, colorsCounts=inCountsColorMap,
          colorStDev=inStDevColorMap, colorsEM=inEnrichmentColorMap,
          colorsMotif=inLetterColors, xAxisLabels=inAAPositions,
          xAxisLabelsBinned=inAAPositionsBinned, residueLabelType=inYLabelEnrichmentMap,
          titleLabelSize=inFigureTitleSize, axisLabelSize=inFigureLabelSize,
          tickLabelSize=inFigureTickSize, printNumber=inPrintNumber,
          showNValues=inShowSampleSize, saveFigures=inSaveFigures, savePath=inFilePath,
          savePathFigs=inSavePathFigures, setFigureTimer=inFigureTimer)



# ========================================== Load Data ===========================================
def loadSubstrates(filePath, fileNames, fileType, printLoadedData, result):
    subsLoaded, totalSubs = ngs.loadSubstrates(filePath=filePath,
                                               fileNames=fileNames,
                                               fileType=fileType,
                                               printLoadedData=printLoadedData)
    result[fileType] = (subsLoaded, totalSubs)

# Load: Inital subs
if inPlotEnrichedSubstrateFrame or inPlotBinnedSubstratePrediction:
    loadInitialSubs = True
    if inPlotBinnedSubstratePrediction:
        if not inPlotEnrichedSubstrateFrame:
            # Define: Save path
            fileNameFixedSubsInitial = (f'subFrameProb {inEnzymeName} InitialSort '
                                        f'{inFixedResidue[0]}@R{inFixedPosition[0]}-'
                                        f'R{inFixedPosition[-1]}')
            fileNameFixedCountsInitial = (f'subFrameTotalCounts {inEnzymeName} InitialSort '
                                          f'{inFixedResidue[0]}@R{inFixedPosition[0]}-'
                                          f'R{inFixedPosition[-1]}.csv')
            fileNameFixedSubsFinal = (f'subFrameProb {inEnzymeName} FinalSort '
                                      f'{inFixedResidue[0]}@R{inFixedPosition[0]}-'
                                      f'R{inFixedPosition[-1]}')
            saveFileName = (f'binnedSubsES {inEnzymeName} Fixed Frame {inFixedResidue[0]}'
                            f'@R{inFixedPosition[0]}-R{inFixedPosition[-1]}')


            if '/' in inSavePath:
                filePathFixedFrameInitial = f'{inSavePath}/{fileNameFixedSubsInitial}'
                filePathFixedFrameInitialTotalCounts = \
                    f'{inSavePath}/{fileNameFixedCountsInitial}'
                filePathFixedFrameFinal = f'{inSavePath}/{fileNameFixedSubsFinal}'
                filePathBinnedSubsES = f'{inSavePath}/{saveFileName}'
            else:
                filePathFixedFrameInitial = f'{inSavePath}\\{fileNameFixedSubsInitial}'
                filePathFixedFrameInitialTotalCounts = \
                    f'{inSavePath}\\{fileNameFixedCountsInitial}'
                filePathFixedFrameFinal = f'{inSavePath}\\{fileNameFixedSubsFinal}'
                filePathBinnedSubsES = f'{inSavePath}\\{saveFileName}'

            if (os.path.exists(filePathBinnedSubsES)
                    or os.path.exists(filePathFixedFrameInitial)
                    or os.path.exists(filePathFixedFrameFinal)):
                loadInitialSubs = False

    if loadInitialSubs:
        # Initialize result dictionary
        loadedResults = {}

        # Create threads for loading initial and final substrates
        threadInitial = threading.Thread(target=loadSubstrates,
                                         args=(inFilePath, inFileNamesInitialSort,
                                               'Initial Sort', inPrintCounts,
                                               loadedResults))

        # Start the threads
        threadInitial.start()

        # Wait for the threads to complete
        threadInitial.join()

        # Retrieve the loaded substrates
        substratesInitial, totalSubsInitial = loadedResults['Initial Sort']
        time.sleep(0.5)


# Load Data: Initial sort
filePathsInitial = []
for fileName in inFileNamesInitialSort:
   filePathsInitial.append(os.path.join(inFilePath, f'counts_{fileName}'))

# Verify that all files exist
missingFile = False
indexMissingFile = []
for indexFile, path in enumerate(filePathsInitial):
    if os.path.exists(path):
        continue
    else:
        missingFile = True
        indexMissingFile.append(indexFile)
if missingFile:
    print('\033[91mERROR: File not found at path:')
    for indexMissing in indexMissingFile:
        print(f'     {filePathsInitial[indexMissing]}')
    print(f'\nMake sure your path is correctly named, and that you have already extracted and '
          f'counted your NGS data\n')
    sys.exit()
else:
    countsInitial, totalSubsInitial = ngs.loadCounts(filePath=filePathsInitial,
                                                     files=inFileNamesInitialSort,
                                                     printLoadedData=inPrintCounts,
                                                     fileType='Initial Sort', fixedSeq=None)
    # Calculate RF
    initialRF = ngs.calculateRF(counts=countsInitial, N=totalSubsInitial,
                                fileType='Initial Sort', printRF=inPrintRF)


    RFsum = np.sum(initialRF, axis=1)
    initialRFAvg = RFsum / len(initialRF.columns)
    initialRFAvg = pd.DataFrame(initialRFAvg, index=initialRFAvg.index, columns=['Average RF'])




# ======================================= Define Functions =======================================
def pressKey(event):
    if event.key == 'escape':
        plt.close()



def loadSubstratesFixedFrame():
    fixedFrameSubs = []
    for index in range(len(inFixedPosition)):
        print('============================== Load: Fixed Frame Substrates '
              '=============================')

        # Define: File path
        frame = f'{inFixedResidue[0]}@R{inFixedPosition[index]}'
        labelTag = (f'{inEnzymeName}_FixedFrame_{frame}_'
                    f'MinCounts {inMinimumSubstrateCount}')
        filePathFixedFrameSubs = os.path.join(inFilePath,
                                              f'fixedSubs_{labelTag}')
        print(f'Loading substrates at path:\n'
              f'     {greenDark}{filePathFixedFrameSubs}{resetColor}\n')


        # Load Data: Fixed substrates
        with open(filePathFixedFrameSubs, 'rb') as file:
            loadedSubs = pk.load(file)
        fixedFrameSubs.append(loadedSubs)
        print(f'Fixed Frame:{purple} {frame}{resetColor}')
        iteration = 0
        for substrate, count in loadedSubs.items():
            print(f'Substrate:{greenLight} {substrate}{resetColor}\n'
                  f'     Count:{red} {count:,}{resetColor}')
            iteration += 1
            if iteration >= inPrintNumber:
                print('\n')
                break

        # PCA: On a single fixed frame
        if inPCAIndividual:
            # Convert substrate data to numerical
            tokensESM, subsESM, subCountsESM = ngs.ESM(substrates=loadedSubs,
                                                       collectionNumber=inTotalSubsPCA,
                                                       useSubCounts=inEncludeSubstrateCounts,
                                                       subPositions=inAAPositions,
                                                       datasetTag=frame)

            # Cluster substrates
            subPopulations = ngs.PCA(substrates=loadedSubs, data=tokensESM, indices=subsESM,
                                     numberOfPCs=inNumberOfPCs, fixedTag=inSubstrateFrame,
                                     N=subCountsESM, fixedSubs=True, figSize=inFigureSize,
                                     saveTag=inSubstrateFrame)

            # Plot: Substrate clusters
            if (inPlotEnrichmentMapPCA or inPlotEnrichmentMotifPCA
                    or inPredictSubstrateActivityPCA or inPlotWordsPCA):
                clusterCount = len(subPopulations)
                for index, subCluster in enumerate(subPopulations):
                    # Plot data
                    plotSubstratePopulations(clusterSubs=subCluster, clusterIndex=index,
                                             numClusters=clusterCount)

    return fixedFrameSubs



def fixInitialSubs(substrates):
    print('================================ Fix: Initial Substrates '
          '================================')
    print(f'Substrate Frames:{purple} Initial Sort{resetColor}')

    subFrameInitial = {}
    numberOfSubs = 0

    for substrate in substrates:
        for indexFrame in range(len(inFixedPosition)):
            subFrame = substrate[inFramePositons[0]+indexFrame:
                                 inFramePositons[-1]+indexFrame]
            if subFrame in subFrameInitial:
                subFrameInitial[subFrame] += 1
            else:
                subFrameInitial[subFrame] = 1
            numberOfSubs += 1

    # Sort the substrate frames
    subFrameInitial = dict(sorted(subFrameInitial.items(),
                                  key=lambda x: x[1], reverse=True))

    # Print the substrat frames
    iteration = 0
    for substrate, counts in subFrameInitial.items():
        print(f'{green} {substrate}{resetColor}, {pink}{counts}')
        if iteration == inPrintNumber:
            print(f'{resetColor}\n')
            break
        else:
            iteration += 1

    return subFrameInitial, numberOfSubs



def trimSubstrates():
    print('=============================== Process Known Substrates '
          '================================')
    trimedSubs = {}
    for index, substrate in enumerate(inSubsPredict):
        trimedSub = substrate[inSubsPredictStartIndex:
                              inFixedFrameLength+inSubsPredictStartIndex]
        print(f'Substrate:{pink} {substrate}{resetColor}\n'
              f'     Sub:{yellow} {trimedSub}{resetColor}')
        if trimedSub in trimedSubs:
            print(f'     {white}Duplicate Sub{resetColor}')
        trimedSubs[trimedSub] = 0
    print('\n')

    return trimedSubs



def substrateProbability(substrates, N, sortType):
    print('=========================== Calcualte: Substrate Probability '
          '============================')
    print(f'Substrate Frames:{purple} {sortType}')

    subFrameProb = {}
    for substrate, count in substrates.items():
        subFrameProb[substrate] = count / N

    iteration = 0

    for substrate, count in subFrameProb.items():
        print(f'{green}{substrate}{resetColor}, {pink}{count}')
        if iteration == inPrintNumber:
            print(f'{resetColor}\n')
            break
        else:
            iteration += 1

    return subFrameProb



def subFrameEnrichment(substratesInitial, initialN, substratesFinal):
    print('============================ Calcualte: Substrate Enrichment '
          '============================')
    decimals = 3
    enrichedSubs = {}
    minInitialProb = 1 / initialN
    print(f'Min Initial Prob:{pink} {minInitialProb}{resetColor}\n')
    for substrate, probabiliy in substratesFinal.items():
        if substrate in substratesInitial:
            enrichedSubs[substrate] = np.log2(substratesFinal[substrate] /
                                              substratesInitial[substrate])
        else:
            enrichedSubs[substrate] = np.log2(substratesFinal[substrate] / minInitialProb)


    # Rank enriched substrates
    enrichedSubs = dict(sorted(enrichedSubs.items(), key=lambda x: x[1], reverse=True))

    # Print: Enriched subs
    iteration = 0
    print(f'Enriched Substrates:')
    for substrate, ES in enrichedSubs.items():
        if iteration == inPrintNumber:
            print('\n')
            break
        iteration += 1
        print(f'Substrate:{green} {substrate}{resetColor}\n'
              f'     ES: {white}{np.round(ES, decimals)}{resetColor}')
        if substrate in substratesInitial:
            print(f'     Prob Final:{pink} {substratesFinal[substrate]}'
                  f'{resetColor}\n'
                  f'     Prob Initial:{pink} {substratesInitial[substrate]}'
                  f'{resetColor}\n')
        else:
            print(f'     Prob Final:{pink} {substratesFinal[substrate]}'
                  f'{resetColor}\n'
                  f'     Prob Initial:{pink} {minInitialProb}'
                  f'{resetColor}\n')

    return enrichedSubs



def predictActivity(substrates, predictionMatrix, normalizeValues, matrixType):
    print('=================================== Predict Activity '
          '====================================')
    print(f'Matrix Type:{white} {matrixType}\n'
          f'{predictionMatrix}{resetColor}\n')
    maxScore = 0
    minScore = 0
    for substrate in substrates:
        score = 0
        for index, AA in enumerate(substrate):
            position = inFixedFramePositions[index]
            score += predictionMatrix.loc[AA, position]
        substrates[substrate] = score
        if score > maxScore:
            maxScore = score
        if score < minScore:
            minScore = score


    if normalizeValues:
        if minScore < 0:
            # Set minimum predicted score = 0
            for substrate, score in substrates.items():
                newScore = score - minScore
                substrates[substrate] = newScore

                # Update max score
                if newScore > maxScore:
                    maxScore = newScore

        # Normalize Values
        for substrate, score in substrates.items():
            substrates[substrate] = score / maxScore
        maxScore, minScore = 1, 0


    print(f'Prediction Matrix:{white} {matrixType}{resetColor}')
    for substrate, score in substrates.items():
        print(f'     {yellow} {substrate}{resetColor}, '
              f'Score:{pink} {np.round(score, 2)}{resetColor}')
    print('\n')

    return substrates, maxScore, minScore



def generateKinetics(predictions):
    print('============================== Evalute Prediction Accuracy '
          '==============================')
    # Create random kinetics data
    kinetics = {}
    maxScore = 0
    minScore = 0
    for substrate, score in predictions.items():
        maxRandomValue = 0.2
        mod = random.randint(0, 9)
        if mod % 2 == 0:
            newScore = score + random.uniform(0, maxRandomValue)
            kinetics[substrate] = newScore
            if newScore > maxScore:
                maxScore = newScore
        else:
            newScore = score - random.uniform(0, maxRandomValue)
            kinetics[substrate] = newScore
            if newScore < minScore:
                minScore = newScore

    # Print simulated data
    for substrate, score in kinetics.items():
        newScore = score / maxScore
        kinetics[substrate] = newScore
        print(f'Substrate:{green} {substrate}{resetColor}\n'
              f'     Predicted:{purple} {predictions[substrate]}{resetColor}\n'
              f'     Kinetics:{green} {newScore}{resetColor}\n')

    # Evaluate the predictions
    xValues = list(predictions.values())
    yValues = list(kinetics.values())
    avg = []
    stDev = []
    for index in range(len(xValues)):
        avg.append(np.mean([xValues[index], yValues[index]]))
        stDev.append(np.std([xValues[index], yValues[index]]))
    r2 = r2_score(yValues, xValues)

    return [predictions.keys(), xValues, yValues, avg, stDev, r2]



def plotSubstratePrediction(substrates, predictValues, scaledMatrix, plotdataPCA, popPCA):
    # Prep data for the figure
    xValues = []
    yValues = []
    for substrate, score in substrates.items():
        xValues.append(substrate)
        yValues.append(score)

    substrateColors = {}
    for index, substrate in enumerate(substrates.keys()):
        substrateColors[substrate] = inDatapointColor[index]

    # Define: Figure parameters
    if scaledMatrix:
        yMin = inYMinPredScaled
        yTickMin = inYTickMinScaled
        if plotdataPCA:
            title = (f'{inEnzymeName}: PCA Population {popPCA + 1}\n'
                     f'{inSubstrateFrame}\nScaled Enrichment Scores')
        else:
            title = f'{inEnzymeName}\n{inSubstrateFrame}\nScaled Enrichment Scores'
    else:
        yMin = inYMinPred
        yTickMin = inYTickMinPred
        if plotdataPCA:
            title = (f'{inEnzymeName}: PCA Population {popPCA + 1}\n'
                     f'{inSubstrateFrame}\nEnrichment Scores')
        else:
            title = f'{inEnzymeName}\n{inSubstrateFrame}\nEnrichment Scores'


    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = plt.bar(xValues, yValues, color=inDatapointColor, width=inBarWidth)
    plt.ylabel('Normalized Predicted Activity', fontsize=inFigureLabelSize)
    plt.title(title, fontsize=inFigureTitleSize, fontweight='bold')
    if not predictValues:
        plt.title(f'{inEnzymeName}\n'
                  f'Fixed Frames '
                  f'{inFixedResidue[0]}@R{inFixedPosition[0]}-R{inFixedPosition[-1]}\n'
                  f'Prediction: SArKS - ESM',
                  fontsize=inFigureTitleSize, fontweight='bold')
        yMin = inYMinPredAI
        yTickMin = inYTickMinAI
    plt.xticks(rotation=90, ha='center')
    plt.axhline(y=0, color='black', linewidth=inLineThickness)
    plt.ylim(yMin, inYMaxPred)

    # Set edge color
    try:
        inSubsManual
    except NameError:
        for index, bar in enumerate(bars):
            bar.set_edgecolor(inEdgeColor)
    else:
        lenSubs = len(substrates)
        lenSubsManual = len(inSubsManual)
        lenSubsCovid = len(inSubsCovid)
        if lenSubs == lenSubsTotal:
            for index, bar in enumerate(bars):
                if index < lenSubsManual:
                    bar.set_edgecolor('black')
                elif index < lenSubsManual + lenSubsCovid:
                    bar.set_edgecolor('#CC5500')
                else:
                    bar.set_edgecolor('black')
        else:
            if lenSubs == lenSubsCovid:
                for bar in bars:
                    bar.set_edgecolor(inEdgeColorOrange)
            else:
                for bar in bars:
                    bar.set_edgecolor('black')

    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=inTickLength,
                   labelsize=inFigureTickSize, width=inLineThickness)

    # Set yticks
    tickStepSize = 0.2
    yTicks = np.arange(yTickMin, 1 + tickStepSize, tickStepSize)
    yTickLabels = [f'{tick:.0f}' if tick == 0 or int(tick) == 1 else f'{tick:.1f}'
                   for tick in yTicks]
    ax.set_yticks(yTicks)
    ax.set_yticklabels(yTickLabels)

    # Set the thickness of the figure border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(inLineThickness)

    fig.canvas.mpl_connect('key_press_event', pressKey)
    fig.tight_layout()
    plt.show()



def plotMotif(data, initialN, finalN, xLabels, figBorders, bigLettersOnTop, plotReleasedCounts,
              addLines, motifType):
    # Set local parameters
    try:
        if bigLettersOnTop:
            stackOrder = 'big_on_top'
        else:
            stackOrder = 'small_on_top'
        colors = {
            'A': inLetterColors[0],
            'R': inLetterColors[2],
            'N': inLetterColors[4],
            'D': inLetterColors[1],
            'C': inLetterColors[6],
            'E': inLetterColors[1],
            'Q': inLetterColors[4],
            'G': inLetterColors[0],
            'H': inLetterColors[2],
            'I': inLetterColors[0],
            'L': inLetterColors[0],
            'K': inLetterColors[2],
            'M': inLetterColors[6],
            'F': inLetterColors[5],
            'P': inLetterColors[0],
            'S': inLetterColors[3],
            'T': inLetterColors[3],
            'W': inLetterColors[5],
            'Y': inLetterColors[5],
            'V': inLetterColors[0]
        }

        # Redefine column header
        data.columns = range(len(data.columns))

        # Define Bounds: y-axis
        yMax = []
        yMin = []
        for column in data.columns:
            sumPositive = 0
            sumNegative = 0
            for index in data.index:
                value = data.loc[index, column]
                if value > 0:
                    sumPositive += value
                if value < 0:
                    sumNegative += value
            yMax.append(sumPositive)
            yMin.append(sumNegative)
        # print(f'yMax: {yMax}\n'
        #       f'yMin: {yMin}\n')
        yMax = max(yMax)
        yMin = min(yMin)
        index = 0
        # yMin = [-10, -10]
        # yMin = yMin[1]
        # index = 1

        # Plot the sequence motif
        fig, ax = plt.subplots(figsize=inFigureSize)
        motif = logomaker.Logo(data.transpose(), ax=ax, color_scheme=colors,
                               width=0.95, stack_order=stackOrder)
        if 'Scaled' in motifType:
            lefts = [0.131, 0.119]
            plt.subplots_adjust(top=figBorders[0], bottom=figBorders[1], left=lefts[index],
                                right=figBorders[3])
        elif 'Enrichment' in motifType:
            lefts = [0.117, 0.117]
            plt.subplots_adjust(top=figBorders[0], bottom=figBorders[1], left=lefts[index],
                                right=figBorders[3])
        else:
            plt.subplots_adjust(top=figBorders[0], bottom=figBorders[1], left=figBorders[2],
                                right=figBorders[3])

        # Set figure title
        if inShowSampleSize:
            if plotReleasedCounts:
                maxSubCount = max(finalN)
                minSubCount = min(finalN)

                if 'Enrichment' in motifType:
                    motif.ax.set_title(f'{inEnzymeName}\n'
                                       f'N Initial = {initialN:,}\n'
                                       f'N Final = {maxSubCount:,}-{minSubCount:,}',
                                       fontsize=inFigureTitleSize, fontweight='bold')
                else:
                    motif.ax.set_title(f'{inEnzymeName}\n'
                                       f'N = {maxSubCount:,}-{minSubCount:,}',
                                       fontsize=inFigureTitleSize, fontweight='bold')
            else:
                if 'Enrichment' in motifType:
                    motif.ax.set_title(f'{inTitleEnrichmentMap}\n'
                                       f'N Initial = {initialN:,}\n'
                                       f'N Final = {finalN:,}',
                                       fontsize=inFigureTitleSize, fontweight='bold')
                else:
                    motif.ax.set_title(f'{inTitleEnrichmentMap}\n'
                                       f'N = {finalN:,}',
                                       fontsize=inFigureTitleSize, fontweight='bold')
        else:
            motif.ax.set_title(inTitleEnrichmentMap, fontsize=inFigureTitleSize,
                               fontweight='bold')

        # Set borders
        motif.style_spines(visible=False)
        motif.style_spines(spines=['left', 'bottom'], visible=True)
        for spine in motif.ax.spines.values():
            spine.set_linewidth(inLineThickness)

        # Set xticks
        motif.ax.set_xticks([pos for pos in range(len(xLabels))])
        motif.ax.set_xticklabels(xLabels, fontsize=inFigureTickSize,
                                 rotation=0, ha='center')
        for tick in motif.ax.xaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(inLineThickness)  # Set tick width

        # Set yticks
        if 'Enrichment' in motifType:
            yTicks = [yMin, 0, yMax]
            yTickLabels = [f'{tick:.2f}' if tick != 0 else f'{int(tick)}' for tick in yTicks]
            yLimitUpper = yMax
            yLimitLower = yMin
        else:
            yTicks = range(0, 5)
            yTickLabels = [f'{tick:.0f}' if tick != 0 else f'{int(tick)}' for tick in yTicks]
            yLimitUpper = 4.32
            yLimitLower = 0
        motif.ax.set_yticks(yTicks)
        motif.ax.set_yticklabels(yTickLabels, fontsize=inFigureTickSize)
        motif.ax.set_ylim(0, yLimitUpper)
        for tick in motif.ax.yaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(inLineThickness)  # Set tick width

        # Label the axes
        motif.ax.set_xlabel('Position', fontsize=inFigureLabelSize)
        if motifType == 'Weblogo':
            motif.ax.set_ylabel('Bits', fontsize=inFigureLabelSize)
        else:
            motif.ax.set_ylabel(motifType, fontsize=inFigureLabelSize)

        # Set horizontal line
        motif.ax.axhline(y=0, color='black', linestyle='-', linewidth=inLineThickness)
        if 'Enrichment' not in motifType and addLines:
            for tick in yTicks:
                motif.ax.axhline(y=tick, color='black', linestyle='--',
                                 linewidth=inLineThickness)

        # Evaluate dataset for fixed residues
        spacer = np.diff(motif.ax.get_xticks())  # Find the space between each tick
        spacer = spacer[0] / 2

        # Use the spacer to set a grey background to fixed residues
        fixedPosition = []  # Ensure this is defined properly
        for index, position in enumerate(xLabels):
            if position in fixedPosition:
                # Plot grey boxes on each side of the xtick
                motif.ax.axvspan(index - spacer, index + spacer,
                                 facecolor='darkgrey', alpha=0.2)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        plt.show()

    except Exception as error:
        print(f'{orange}ERROR{resetColor}:{orange}plotMotif()\n'
              f'     {error}{resetColor}\n\n')
        sys.exit()



def plotPredictionAccuracy(data, predictionType):
    x, y, accuracy = data[1], data[2], data[5]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=inFigureSize)
    ax.scatter(x, y, color=inDatapointColor)
    plt.xlabel('Predicted Scores', fontsize=inFigureLabelSize)
    plt.ylabel('Kinetics', fontsize=inFigureLabelSize)
    plt.title(f'{inEnzymeName}\n'
              f'{predictionType}\n'
              f'R² = {np.round(accuracy, 3)}',
              fontsize=inFigureTitleSize, fontweight='bold')
    if min(y) >= 0:
        plt.subplots_adjust(top=0.873, bottom=inBottomParam, left=0.088, right=0.979)
    else:
        plt.subplots_adjust(top=0.873, bottom=inBottomParam, left=0.104, right=0.979)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=inTickLength,
                   labelsize=inFigureTickSize, width=inLineThickness)

    # Set the thickness of the figure border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(inLineThickness)

    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()



def plotPredictionStats(data, predictionType):
    substrates, avgScores, stDev, accuracy = data[0], data[3], data[4], data[5]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=inFigureSize)
    bars = ax.bar(substrates, avgScores, yerr=stDev, color='white', width=inBarWidth)
    ax.errorbar(substrates, avgScores, yerr=stDev, fmt='none',
                ecolor='black', elinewidth=inLineThickness, capsize=5, capthick=inLineThickness)
    plt.axhline(y=0, color='black', linewidth=inLineThickness)
    plt.ylabel('Average Activity Score', fontsize=inFigureLabelSize)
    plt.title(f'{inEnzymeName}\n'
              f'{predictionType}\n'
              f'Prediction Precision',
              fontsize=inFigureTitleSize, fontweight='bold')
    plt.xticks(rotation=90, ha='center')
    if min(avgScores) >= 0:
        plt.subplots_adjust(top=0.873, bottom=inBottomParam, left=0.088, right=0.979)
    else:
        plt.subplots_adjust(top=0.873, bottom=inBottomParam, left=0.104, right=0.979)

    # Set edge color
    for index, bar in enumerate(bars):
        bar.set_edgecolor(inDatapointColor[index])
        bar.set_linewidth(inLineThickness)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=inTickLength,
                   labelsize=inFigureTickSize, width=inLineThickness)

    # Set the thickness of the figure border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(inLineThickness)

    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()



def predictSubstrateEnrichment(substratesEnriched, matrix, matrixType):
    print('=============================== Predict Enrichment Scores '
          '===============================')
    print(f'Number of Unique Enriched Subs:'
          f'{white} {len(substratesEnriched):,}{resetColor}\n\n'
          f'Matrix:{pink} {matrixType}{green}\n{matrix}{resetColor}\n\n')

    # Predict enrichment scores
    iteration = 0
    substrates = {}
    for substrate, score in substratesEnriched.items():
        substrateScorePredicted = 0
        for index, AA, in enumerate(substrate):
            valueAA = matrix.loc[AA, inFixedFramePositions[index]]

            # # Increase weight:
            # if 1 >= abs(valueAA):
            #     # print(f'Initial Value: {valueAA}')
            #     valueAA = valueAA + (valueAA * 1 / 4)
            #     # print(f'New Value 1: {valueAA}\n')
            #     # sys.exit()
            # elif 2 >= abs(valueAA) > 1:
            #     # print(f'Initial Value: {valueAA}')
            #     valueAA = valueAA + (valueAA * 1 / 2)
            #     # print(f'New Value 2: {valueAA}\n')
            #     # sys.exit()
            # elif 3 >= abs(valueAA) > 2:
            #     # print(f'Initial Value: {valueAA}')
            #     valueAA = valueAA + (valueAA * 2 / 3)
            #     # print(f'New Value 3: {valueAA}\n')
            #     # sys.exit()
            # elif 4 >= abs(valueAA) > 3:
            #     # print(f'Initial Value: {valueAA}')
            #     valueAA = valueAA + (valueAA * 3 / 2)
            #     # print(f'New Value 4: {valueAA}\n')
            #     # sys.exit()
            # else:
            #     # print(f'Initial Value: {valueAA}')
            #     valueAA = valueAA + (valueAA * 5 / 3)
            #     # print(f'New Value 5: {valueAA}\n')
            #     # sys.exit()

            if valueAA >= 0:
                # Increase weight: Enriched residues
                if 1 <= valueAA:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 1 / 4
                    # print(f'New Value 1: {valueAA}\n')
                    # sys.exit()
                elif 2 <= valueAA > 1:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 1 / 2
                    # print(f'New Value 2: {valueAA}\n')
                    # sys.exit()
                elif 3 <= valueAA > 2:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 2 / 3
                    # print(f'New Value 3: {valueAA}\n')
                    # sys.exit()
                elif 4 <= valueAA > 3:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 3 / 2
                    # print(f'New Value 4: {valueAA}\n')
                    # sys.exit()
                else:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 1 / 3
                    # print(f'New Value 5: {valueAA}\n')
                    # sys.exit()
            else:
                # Increase weight: Deenriched residues
                if valueAA > -1:
                    # print(f'Initial Value: {valueAA}')
                    valueAA = valueAA + valueAA
                    # print(f'New Value A: {valueAA}\n')
                elif -1 >= valueAA > -2:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 1 / 4
                    # print(f'New Value B: {valueAA}\n')
                    # sys.exit()
                elif -2 >= valueAA > -3:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 2 / 4
                    # print(f'New Value C: {valueAA}\n')
                    # sys.exit()
                elif -3 >= valueAA > -4:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 3 / 4
                    # print(f'New Value D: {valueAA}\n')
                    # sys.exit()
                else:
                    # print(f'Initial Value: {valueAA}')
                    valueAA += valueAA * 5 / 3
                    # print(f'New Value E: {valueAA}\n')
                    # sys.exit()

            # Update: Prediction score
            substrateScorePredicted += valueAA
        iteration += 1
        substrates[substrate] = (score, substrateScorePredicted)
    
    # Print: Predictions
    iteration = 0
    iterationPrint = 0
    for substrate, score in substrates.items():
        expES, predES = score
        if iteration % 10 == 0:
            print(f'Substrate:{green} {substrate}{resetColor}\n'
                  f'    Experimental:{pink} {expES}{resetColor}\n'
                  f'    Predicted:{purple} {predES}{resetColor}\n')
            iterationPrint += 1
        else:
            iteration += 1
        if iterationPrint == inPrintNumber:
            print('')
            break

    
    # Extract x and y values for the scatter plot
    xValues, yValues = [], []
    if inPlotSubsetOfSubstrates:
        iteration = 0
        selectedSubstrates = {}
        for substrate, scores in substrates.items():
            if iteration % 10 == 0:
                selectedSubstrates[substrate] = scores
            iteration += 1

        # Unpack the datapoints
        substrates = selectedSubstrates.copy()
        xValues = [scores[0] for scores in substrates.values()]
        yValues = [scores[1] for scores in substrates.values()]
    else:
        for substrate, scores in substrates.items():
            xValues.append(scores[0])
            yValues.append(scores[1])

    # Normalize the values
    if inNormalizeValues:
        if inPrintNormalizedValues:
            print('Adjust all values to range from 0 to 1')

            # Normalize x-values
            x_max = max(xValues)
            x_min = min(xValues)
            xValues_normalized = []
            print(f'Normalize the x-values:\n'
                  f'     Initial Boundaries:\n'
                  f'          X Max: {x_max}\n'
                  f'          X Min: {x_min}\n')

            # Make sure that there are no negative values
            if x_min < 0:
                xValues = xValues + abs(x_min)
                x_max = max(xValues)

            for x in xValues:
                xValues_normalized.append(x / x_max)
            xValues = xValues_normalized
            print(f'     Adjusted Boundaries:\n'
                  f'          X Max: {max(xValues)}\n'
                  f'          X Min: {min(xValues)}\n\n')

            # Normalize y-values
            y_max = max(yValues)
            y_min = min(yValues)
            yValues_normalized = []
            print(f'Normalize the y-values:\n'
                  f'     Initial Boundaries:\n'
                  f'          Y Max: {y_max}\n'
                  f'          Y Min: {y_min}\n')

            # Make sure that there are no negative values
            if y_min < 0:
                yValues = yValues + abs(y_min)
                y_max = max(yValues)
                print(f'     Y Min was less than 0\n'
                      f'          Y Max Adjusted: {y_max}\n'
                      f'          Y Min Adjusted: {min(yValues)}\n')

            # Normalize y-values
            for y in yValues:
                yValues_normalized.append(y / y_max)
            yValues = yValues_normalized
            print(f'     Adjusted Boundaries:\n'
                  f'          Y Max: {max(yValues)}\n'
                  f'          Y Min: {min(yValues)}\n\n')
        else:
            x_max = max(xValues)
            x_min = min(xValues)
            xValues_normalized = []

            # Make sure that there are no negative values
            if x_min < 0:
                xValues = xValues + abs(x_min)
                x_max = max(xValues)

            # Normalize x-values
            for x in xValues:
                xValues_normalized.append(x / x_max)
            xValues = xValues_normalized

            y_max = max(yValues)
            y_min = min(yValues)
            yValues_normalized = []

            # Make sure that there are no negative values
            if y_min < 0:
                yValues = yValues + abs(y_min)
                y_max = max(yValues)

            # Normalize y-values
            for y in yValues:
                yValues_normalized.append(y / y_max)
            yValues = yValues_normalized

    # Inspect pillar
    if inPrintPredictionAccuracy:
        iteration = 0
        iterationPrint = 0
        avgEnrichment = 0
        avgComputational = 0
        print(f'{green}Inspect x-axis:{pink} {inInspectDataUpperValue}{resetColor} '
              f'>= ES: Experimentally Enriched Substrate >='
              f'{pink} {inInspectDataLowerValue}{resetColor}')
        for substrate in substrates.keys():
            valueEnrichment = xValues[iteration]
            if inInspectDataUpperValue >= valueEnrichment >= inInspectDataLowerValue:
                if iterationPrint >= inPrintNumber:
                    avgEnrichment /= iterationPrint
                    avgComputational /= iterationPrint
                    print(f'Average Displayed Enrichment Score:'
                          f'{pink} {avgEnrichment}{resetColor}\n'
                          f'Average Displayed Predicted Score:'
                          f'{pink} {avgComputational}{resetColor}\n\n')
                    break

                # Update datapoints
                valuePredicted = yValues[iteration]

                difference = abs(valueEnrichment - valuePredicted)
                if difference >= inDifferenceThreshold:
                    avgEnrichment += valueEnrichment
                    avgComputational += valuePredicted

                    print(f'{pink}{substrate}{resetColor}\n'
                          f'     Enrichment:{white} {valueEnrichment}{resetColor}\n'
                          f'     Predicted:{white} {valuePredicted}{resetColor}\n'
                          f'     Difference:{green} {difference}{resetColor}\n')
                    iterationPrint += 1
            iteration += 1
    
    decimals = 3
    if inInspectExperimentalES:
        print(f'Inspect x-axis:{purple} {inExperimentalESUpperLimit}{resetColor} '
              f'>= Substrate Enrichment >= '
              f'{purple} {inExperimentalESLowerLimit}{resetColor}')
        iteration = 0
        for substrate, scores in substrates.items():
            iteration += 1
            valueEnrichment, valuePredicted = scores[0], scores[1]
            if inExperimentalESUpperLimit >= valueEnrichment >= inExperimentalESLowerLimit:
                if iteration % 10 == 0:
                    difference = valueEnrichment - valuePredicted
                    print(f'{green}{substrate}{resetColor}\n'
                          f'     Enrichment:{pink} {np.round(valueEnrichment, decimals)}'
                          f'{resetColor}\n'
                          f'     Predicted:{purple} {np.round(valuePredicted, decimals)}'
                          f'{resetColor}\n'
                          f'     Difference:{white} {np.round(difference, decimals)}'
                          f'{resetColor}\n')
    
    if inInspectPredictedES:
        print(f'\nInspect y-axis:{purple} {inPredictedESUpperLimit}{resetColor} '
              f'>= Predicted Substrate Enrichment >= '
              f'{purple} {inPredictedESLowerLimit}{resetColor}')
        iteration = 0
        for substrate, scores in substrates.items():
            iteration += 1
            valueEnrichment, valuePredicted = scores[0], scores[1]
            if inPredictedESUpperLimit >= valuePredicted >= inPredictedESLowerLimit:
                if iteration % 10 == 0:
                    difference = valueEnrichment - valuePredicted
                    print(f'{green}{substrate}{resetColor}\n'
                          f'     Enrichment:{pink} {np.round(valueEnrichment, decimals)}'
                          f'{resetColor}\n'
                          f'     Predicted:{purple} {np.round(valuePredicted, decimals)}'
                          f'{resetColor}\n'
                          f'     Difference:{white} {np.round(difference, decimals)}'
                          f'{resetColor}\n')

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xValues, yValues, color=inPredictionDatapointColor, alpha=0.5)
    ax.set_title(f'Substrate Enrichment Scores\n'
                 f'Matrix Type: {matrixType}',
                 fontsize=inFigureTitleSize, fontweight='bold')
    ax.set_xlabel('Experimental Substrate Scores', fontsize=inFigureLabelSize,
                  fontweight='bold')
    ax.set_ylabel('Predicted Substrate Scores', fontsize=inFigureLabelSize,
                  fontweight='bold')
    ax.grid(True, color='black')
    plt.subplots_adjust(top=0.908, bottom=0.083, left=0.094, right=0.958)

    if inSetAxisLimits:
        plt.xlim(inMinX, inMaxX)
        plt.ylim(inMinY, inMaxY)
    else:
        if inNormalizeValues:
            plt.xlim(-0.05, 1.05)
            # plt.ylim(-0.05, 1.05)
        else:
            if inMiniumSubstrateScoreLimit:
                plt.xlim(inMiniumSubstrateScore - 0.5, 10.5)
            # else:
            # plt.xlim(-7, 10.5)

    if inPlotSubstrateText:
        # Annotate the points with the substrate names (optional)
        for substrate, (x, y) in substrates.items():
            ax.annotate(substrate, (x, y), textcoords="offset points", xytext=(0, 10),
                        ha='center')

    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=inTickLength,
                   labelsize=inFigureTickSize, width=inLineThickness)

    # Set the thickness of the figure border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(inLineThickness)

    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()



def plotFACSData():
    print('================================= Measured FACS Activity '
          '================================')
    # Define the data
    substrates = ['VVLQSGFR', 'VVLQSPFR', 'VYLQSGFR', 'VVLQAGFR', 'VVMQSGFR',
                  'IVLQSGFR', 'VVLHSGFR', 'VGLQSGFR', 'VVLMSGFR', 'VVVQSGFR',
                  'VVLQIGFR', 'VVGQSGFR', 'KVLQSGFR', 'VVLQNGFR', 'VVLYSGFR']
    dataFACS = {
        '% Run 1':
            [100.0, 100, 100.0, 100.0, 94.5, 97.3, 85.1, 89.6, 77.0, 37.0, 5.5,
             0.2, 0.2, 0.4, 0.3],
        '% Run 2':
            [100.0, 100, 100.0, 99.3, 100.0, 93.8, 86.1, 85.8, 68.2, 31.2, 5.2,
             0.3, 0.3, 0.1, 0.1],
        '% Run 3':
            [100.0, 100, 99.4, 100.0, 94.6, 93.2, 90.6, 76.2, 74.5, 42.0, 5.1,
             1.1, 0.4, 0.1, 0.1]
    }

    # Convert the data into a pandas DataFrame
    dataFACS = pd.DataFrame(dataFACS, index=substrates)
    dataFACS = dataFACS / 100
    dataFACS['Avg Activity'] = dataFACS[
        ['% Run 1', '% Run 2', '% Run 3']].mean(axis=1)
    dataFACS['Std Dev'] = dataFACS[
        ['% Run 1', '% Run 2', '% Run 3']].std(axis=1)
    print(f'{dataFACS}\n\n')


    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = plt.bar(dataFACS.index, dataFACS.loc[:, 'Avg Activity'],
                   yerr=dataFACS['Std Dev'], capsize=5,
                   color='lightgrey', width=inBarWidth)
    plt.title(f'\n{inEnzymeName}\n'
              f'FACS Activity', fontsize=inFigureTitleSize, fontweight='bold')
    plt.ylabel('Normalized Activity', fontsize=inFigureLabelSize)
    plt.xticks(rotation=90, ha='center')
    plt.axhline(y=0, color='black', linewidth=inLineThickness)
    plt.ylim(0, 1.1)

    # Set edge color
    for index, bar in enumerate(bars):
        bar.set_edgecolor(inBarColor)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=inTickLength,
                   labelsize=inFigureTickSize, width=inLineThickness)

    # Set yticks
    tickStepSize = 0.2
    yTicks = np.arange(0, 1 + tickStepSize, tickStepSize)
    yTickLabels = [f'{tick:.0f}' if tick == 0 or int(tick) == 1 else f'{tick:.1f}'
                   for tick in yTicks]
    ax.set_yticks(yTicks)
    ax.set_yticklabels(yTickLabels)

    # Set the thickness of the figure border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(inLineThickness)

    fig.canvas.mpl_connect('key_press_event', pressKey)
    fig.tight_layout()
    plt.show()



def plotSubstratePopulations(clusterSubs, clusterIndex, numClusters):
    print('==================================== Plot PC Clusters '
          '===================================')
    print(f'Cluster Number:{white} {clusterIndex + 1}{resetColor}\n'
          f'     Total Clusters:{white} {numClusters}{resetColor}\n\n')


    # Define figure titles
    if numClusters == 1:
        figureTitleEM = (f'{inTitleEnrichmentMap}: PCA Population\n'
                         f'{inSubstrateFrame}')
        figureTitleMotif = (f'{inTitleMotif}: PCA Population\n'
                            f'{inSubstrateFrame}')
        figureTitleWordCloud = (f'{inTitleEnrichmentMap}: PCA Population\n'
                                f'{inSubstrateFrame}')
        datasetTag = f'PCA Pop - {inSubstrateFrame}'
    else:
        figureTitleEM = (f'{inTitleEnrichmentMap}: PCA Population {clusterIndex + 1}\n'
                         f'{inSubstrateFrame}')
        figureTitleMotif = (f'{inTitleMotif}: PCA Population {clusterIndex + 1}\n'
                            f'{inSubstrateFrame}')
        figureTitleWordCloud = (f'{inTitleEnrichmentMap}: '
                                f'PCA Population {clusterIndex + 1}\n'
                                f'{inSubstrateFrame}')
        datasetTag = f'PCA Pop {clusterIndex + 1} - {inSubstrateFrame}'

    # Count fixed substrates
    countFullSubstrate = False
    if countFullSubstrate:
        countsFinal, countsFinalTotal = ngs.countResiduesBinned(
            substrates=clusterSubs, positions=inAAPositions, printCounts=inPrintCounts)
    else:
        countsFinal, countsFinalTotal = ngs.countResiduesBinned(
            substrates=clusterSubs, positions=inFixedFramePositions,
            printCounts=inPrintCounts)
    ngs.updateSampleSize(NSubs=countsFinalTotal, sortType='Final Sort',
                         printCounts=inPrintSampleSize, fixedTag=inSubstrateFrame)

    # Adjust the zero counts at nonfixed positions
    countsFinalAdjusted = countsFinal.copy()
    for indexColumn in countsFinalAdjusted.columns:
        for AA in countsFinalAdjusted.index:
            if countsFinalAdjusted.loc[AA, indexColumn] == 0:
                countsFinalAdjusted.loc[AA, indexColumn] = 1
    print(f'Adjusted Final Counts:{pink} {inEnzymeName}\n'
          f'{red}{countsFinalAdjusted}{resetColor}\n\n')


    # Calculate: RF
    finalRF = ngs.calculateRF(counts=countsFinal, N=countsFinalTotal,
                              fileType='Final Sort', printRF=inPrintRF)
    finalRFAdjusted = ngs.calculateRF(counts=countsFinalAdjusted, N=countsFinalTotal,
                                      fileType='Final Sort', printRF=inPrintRF)

    # Calculate: Positional entropy
    positionalEntropy = ngs.calculateEntropy(RF=finalRF, printEntropy=inPrintEntropy)
    if inPlotPositionalEntropyPCAPopulations:
        # Plot: Positional entropy
        ngs.plotPositionalEntropy(entropy=positionalEntropy,
                                  fixedDataset=inFixResidues, fixedTag=inSubstrateFrame,
                                  titleSize=inFigureTitleSize, avgDelta=False)


    # Calculate: Enrichment scores
    fixedFramePopES = ngs.calculateEnrichment(initialSortRF=initialRFAvg,
                                              finalSortRF=finalRF,
                                              fixedSeq=inSubstrateFrame,
                                              printES=inPrintES)
    fixedFramePopESAdjusted = ngs.calculateEnrichment(initialSortRF=initialRFAvg,
                                                      finalSortRF=finalRFAdjusted,
                                                      fixedSeq=inSubstrateFrame,
                                                      printES=inPrintES)

    # Calculate: Enrichment scores scaled
    fixedFramePCAESScaled = pd.DataFrame(0.0, index=fixedFramePopES.index,
                                        columns=fixedFramePopES.columns)
    fixedFramePCAESScaledAdjusted = pd.DataFrame(0.0, index=fixedFramePopES.index,
                                                columns=fixedFramePopES.columns)

    # Scale enrichment scores with Shannon Entropy
    print(f'Entropy:\n{positionalEntropy}\n\n')
    for positon in fixedFramePopES.columns:
        fixedFramePCAESScaled.loc[:, positon] = (fixedFramePopES.loc[:, positon] *
                                                positionalEntropy.loc[
                                                positon, 'ΔEntropy'])
        fixedFramePCAESScaledAdjusted.loc[:, positon] = (
                fixedFramePopESAdjusted.loc[:, positon] *
                positionalEntropy.loc[positon, 'ΔEntropy'])
    print(f'Motif: Scaled\n{fixedFramePCAESScaled}\n\n')
    yMax = max(fixedFramePCAESScaled[fixedFramePopES > 0].sum())
    yMin = min(fixedFramePCAESScaled[fixedFramePopES < 0].sum())


    # # Plot data
    if inPlotEnrichmentMapPCA:
        # Plot: Enrichment Map
        ngs.plotEnrichmentScores(scores=fixedFramePopESAdjusted,
                                 motifType='Enrichment',
                                 figSize=figSizeEM,
                                 figBorders=figBordersEM,
                                 title=f'{figureTitleEM}\nEnrichment Scores',
                                 showScores=inShowEnrichmentScores,
                                 squares=inShowEnrichmentAsSquares,
                                 saveTag=datasetTag,
                                 fixingFrame=False,
                                 initialFrame=True,
                                 duplicateFigure=False)

        # Plot: Enrichment Map Scaled
        ngs.plotEnrichmentScores(scores=fixedFramePCAESScaledAdjusted,
                                 motifType='Scaled Enrichment',
                                 figSize=figSizeEM,
                                 figBorders=figBordersEM,
                                 title=f'{figureTitleEM}\nScaled Enrichment Scores',
                                 showScores=inShowEnrichmentScores,
                                 squares=inShowEnrichmentAsSquares,
                                 saveTag=datasetTag,
                                 fixingFrame=False,
                                 initialFrame=True,
                                 duplicateFigure=False)

    if inPlotEnrichmentMotifPCA:
        if inAdjustZeroCounts:
            # Plot: Sequence Motif
            ngs.plotMotif(data=fixedFramePCAESScaled.copy(),
                          motifType='Scaled Enrichment',
                          bigLettersOnTop=inBigLettersOnTop, figureSize=inFigureSize, 
                          figBorders=inFigureBordersEnrichmentMotif,
                          title=f'{figureTitleMotif}', titleSize=inFigureTitleSize,
                          yMax=yMax, yMin=yMin, yBoundary=2, lines=inAddHorizontalLines,
                          fixingFrame=False, initialFrame=False, duplicateFigure=False,
                          saveTag=datasetTag)

    # Plot: Work cloud
    if inPlotWordsPCA:
        ngs.plotWordCloud(clusterSubs=clusterSubs,
                          clusterIndex=clusterIndex,
                          title=figureTitleWordCloud,
                          saveTag=datasetTag)


    # Plot: Activity predictions
    if inPredictSubstrateActivityPCA:
        # Extract substrate frames
        global subsPredict
        if subsPredict is None:
            subsPredict = trimSubstrates()

        # Predict: Enrichment
        subsPredict, yMax, yMin = predictActivity(
            substrates=subsPredict, predictionMatrix=fixedFramePopESAdjusted,
            normalizeValues=inNormalizePredictions, matrixType='Enrichment Scores')
        plotSubstratePrediction(substrates=subsPredict, predictValues=True,
                                scaledMatrix=False, plotdataPCA=True, popPCA=clusterIndex)

        # Predict: Scaled Enrichment
        subsPredictScaled, yMax, yMin = predictActivity(
            substrates=subsPredict, predictionMatrix=fixedFramePCAESScaledAdjusted,
            normalizeValues=inNormalizePredictions, matrixType='Scaled Enrichment Scores')
        plotSubstratePrediction(substrates=subsPredictScaled, predictValues=True,
                                scaledMatrix=True, plotdataPCA=True, popPCA=clusterIndex)



# ========================================= Run The Code =========================================
# Set flag
subsPredict = None
if inPredictSubstrateActivityPCA:
    inPlotBinnedSubstratePCA = True

if (inPredictSubstrateActivity or inPlotEnrichmentMap
        or inPlotEnrichmentMotif or inPlotWeblogoMotif
        or inPlotBinnedSubstratePrediction):

    # Load: Counts
    (fixedFrameCounts,
     fixedFrameCountsTotal) = ngs.loadFixedFrameCounts(
        filePath=inFilePath, substrateFrame=inFixedFramePositions,
        substrateFrameAAPos=inAAPositionsBinned, frameIndicies=inFramePositons,
        datasetTag=inSubstrateFrame)

    # Calculate: Probability & Entropy
    fixedFrameProb = ngs.calculateFixedFrameRF(countsTotal=fixedFrameCountsTotal,
                                               datasetTag=inSubstrateFrame)
    positionalEntropy = ngs.calculateEntropy(RF=fixedFrameProb,
                                             printEntropy=inPrintEntropy)

    # Calculate: Enrichment Scores
    if (inPredictSubstrateActivity or inPlotEnrichmentMap
            or inPlotEnrichmentMotif or inPlotBinnedSubstrates
            or inPlotBinnedSubstratePrediction):
        fixedSubSeq = f'Fixed Frame {inFixedResidue[0]}@R{inFixedPosition[0]}'
        fixedFrameES = ngs.calculateEnrichment(initialSortRF=initialRFAvg,
                                               finalSortRF=fixedFrameProb,
                                               fixedSeq=fixedSubSeq,
                                               printES=inPrintES)
        fixedFrameESScaled = pd.DataFrame(0.0, index=fixedFrameES.index,
                                          columns=fixedFrameES.columns)
        for positon in fixedFrameES.columns:
            fixedFrameESScaled.loc[:, positon] = (fixedFrameES.loc[:, positon] *
                                                  positionalEntropy.loc[
                                                      positon, 'ΔEntropy'])

        # Calculate: Test ES matrix
        fixedFrameESTest = ngs.calculateEnrichmentTest(initialSortRF=initialRFAvg,
                                                       finalSortRF=fixedFrameProb,
                                                       printES=inPrintES)


# # Plot Data
if inPlotEnrichmentMap:
    # plotES = pd.DataFrame(-math.inf, index=fixedFrameES.index,
    #                       columns=fixedFrameES.columns)
    # addData = [3, 2, 4, 0, 1]
    # for column in addData:
    #     plotES.iloc[:, column] = fixedFrameES.iloc[:, column]
    #     # Plot: Enrichment Map
    #     ngs.plotEnrichmentScores(scores=plotES,
    #                              motifType='Enrichment',
    #                              figSize=figSizeEM,
    #                              figBorders=figBordersEM,
    #                              title=f'{inTitleEnrichmentMap}\n{inSubstrateFrame}\n'
    #                                    f'Enrichment Scores',
    #                              showScores=inShowEnrichmentScores,
    #                              squares=inShowEnrichmentAsSquares,
    #                              saveTag=inSubstrateFrame,
    #                              fixingFrame=False,
    #                              initialFrame=True,
    #                              duplicateFigure=False)

    # Plot: Enrichment Map
    ngs.plotEnrichmentScores(scores=fixedFrameES.copy(),
                             motifType='Enrichment',
                             figSize=figSizeEM,
                             figBorders=figBordersEM,
                             title=f'{inTitleEnrichmentMap}\n{inSubstrateFrame}\n'
                                   f'Enrichment Scores',
                             showScores=inShowEnrichmentScores,
                             squares=inShowEnrichmentAsSquares,
                             saveTag=inSubstrateFrame,
                             fixingFrame=False,
                             initialFrame=True,
                             duplicateFigure=False)

    # Calculate: Stats
    if len(inFixedPosition) != 1:
        ngs.fixedFrameStats(countsList=fixedFrameCounts,
                            initialProb=initialRFAvg,
                            substrateFrame=inFixedFramePositions,
                            figSize=inFigureSize,
                            datasetTag=inSubstrateFrame)

    # Plot: Enrichment Map - Scaled
    ngs.plotEnrichmentScores(scores=fixedFrameESScaled.copy(),
                             motifType='Scaled Enrichment',
                             figSize=figSizeEM,
                             figBorders=figBordersEM,
                             title=f'{inTitleEnrichmentMap}\n{inSubstrateFrame}\n'
                                   f'Scaled Enrichment Scores',
                             showScores=inShowEnrichmentScores,
                             squares=inShowEnrichmentAsSquares,
                             saveTag=inSubstrateFrame,
                             fixingFrame=False,
                             initialFrame=True,
                             duplicateFigure=False)


if inPlotEnrichmentMotif:
    yMax = max(fixedFrameESScaled[fixedFrameESScaled > 0].sum())
    yMin = min(fixedFrameESScaled[fixedFrameESScaled < 0].sum())

    # Plot: Sequence Motif
    ngs.plotMotif(data=fixedFrameESScaled.copy(), motifType='Scaled Enrichment', 
                  bigLettersOnTop=inBigLettersOnTop, figureSize=inFigureSize,
                  figBorders=inFigureBordersEnrichmentMotif,
                  title=f'{inTitleMotif}\n{inSubstrateFrame}',
                  titleSize=inFigureTitleSize, yMax=yMax, yMin=yMin, yBoundary=2,
                  lines=inAddHorizontalLines, fixingFrame=False, initialFrame=False,
                  duplicateFigure=False, saveTag=inSubstrateFrame)

    # Plot: Sequence Motif
    ngs.plotMotif(data=fixedFrameESScaled, motifType='Scaled Enrichment', 
                  bigLettersOnTop=inBigLettersOnTop, figureSize=inFigureSize,
                  figBorders=inFigureBordersEnrichmentMotif,
                  title=f'{inTitleMotif}\n{inSubstrateFrame}',
                  titleSize=inFigureTitleSize, yMax=yMax, yMin=0, yBoundary=2,
                  lines=inAddHorizontalLines, fixingFrame=False, initialFrame=False,
                  duplicateFigure=False, saveTag=f'Y Min - {inSubstrateFrame}')


if inPlotWeblogoMotif:
    releasedRFscaled, fixedAA, yMax, yMin = ngs.heightsRF(
        counts=fixedFrameCountsTotal.copy(), N=fixedFrameCountsTotal,
        invertRF=False, printRF=inPrintMotifData)

    if inShowWeblogoYTicks:
        yMax, yMin = 5, 0
        # Plot: Sequence Motif
        ngs.plotMotif(data=releasedRFscaled.copy(), motifType='Weblogo',
                      bigLettersOnTop=inBigLettersOnTop, figureSize=inFigureSize,
                      figBorders=inFigureBordersEnrichmentMotif,
                      title=f'{inTitleMotif}\n{inSubstrateFrame}',
                      titleSize=inFigureTitleSize, yMax=yMax, yMin=yMin, yBoundary=0,
                      lines=inAddHorizontalLines, fixingFrame=False, initialFrame=False,
                      duplicateFigure=False, saveTag=inSubstrateFrame)
    else:
        # Plot: Sequence Motif
        ngs.plotMotif(data=releasedRFscaled.copy(), motifType='Weblogo',
                      bigLettersOnTop=inBigLettersOnTop, figureSize=inFigureSize,
                      figBorders=inFigureBordersEnrichmentMotif,
                      title=f'{inTitleMotif}\n{inSubstrateFrame}',
                      titleSize=inFigureTitleSize, yMax=yMax, yMin=yMin, yBoundary=2,
                      lines=inAddHorizontalLines, fixingFrame=False,
                      initialFrame=False, duplicateFigure=False, saveTag=inSubstrateFrame)


# Plot: Measured FACS activity
if inPlotActivityFACS:
    plotFACSData()


# Predict activity towards substrates
if inPredictSubstrateActivity:
    # Extract substrate frames
    if subsPredict is None:
        subsPredict = trimSubstrates()

    # Predict: Enrichment
    subsPredict, yMax, yMin = predictActivity(substrates=subsPredict.copy(),
                                              predictionMatrix=fixedFrameES,
                                              normalizeValues=inNormalizePredictions,
                                              matrixType='Enrichment Scores')
    plotSubstratePrediction(substrates=subsPredict, predictValues=True, scaledMatrix=False,
                            plotdataPCA=False, popPCA=None)

    # Predict: Scaled Enrichment
    subsPredictScaled, yMax, yMin = predictActivity(substrates=subsPredict.copy(),
                                                    predictionMatrix=fixedFrameESScaled,
                                                    normalizeValues=inNormalizePredictions,
                                                    matrixType='Scaled Enrichment Scores')
    plotSubstratePrediction(substrates=subsPredictScaled, predictValues=True, scaledMatrix=True,
                            plotdataPCA=False, popPCA=None)

    # Predict: AI
    subsPredictAI = {
        'VVLQSGFR': 6.168589689,
        'VVMQSGFR': 5.012006811,
        'VVVQSGFR': 0.01899718477,
        'VVGQSGFR': -0.4574221781,
        'VVLHSGFR': 2.41369248,
        'VVLMSGFR': 1.682469002,
        'VVLYSGFR': -1.546786043,
        'IVLQSGFR': 5.711949158,
        'KVLQSGFR': 4.342788918,
        'VYLQSGFR': 5.619216201,
        'VGLQSGFR': 3.36114118,
        'VVLQAGFR': 6.478588254,
        'VVLQNGFR': 3.759282483,
        'VVLQIGFR': 3.277549563,
        'VVLQSPFR': 3.933666618,
        'AVLQSGFR': 5.55764466,
        'VTFQSAVK': 2.7657616,
        'ATVQSKMS': 0.8655595173,
        'ATLQAIAS': 5.974816893,
        'VKLQNNEL': 2.965594448,
        'VRLQAGNA': 6.732328298,
        'PMLQSADA': 6.745837998,
        'TVLQAVGA': 5.644710202,
        'ATLQAENV': 6.915391686,
        'TRLQSLEN': 5.03887008,
        'PKLQSSQA': 5.167261885,
        'VTFQGKFK': 4.626522297,
        'PLMQSADA': 4.541013113,
        'PKLQASQA': 5.259179405
        }
    maxValue = max(subsPredictAI.values())
    for substrate, score in subsPredictAI.items():
        subsPredictAI[substrate] = score / maxValue
    subLimit = len(inSubsPredict)
    if subLimit != len(subsPredictAI):
        iteration = 0
        subSubSet = {}
        for substrate, score in subsPredictAI.items():
            subSubSet[substrate] = score
            iteration += 1
            if iteration == subLimit:
                break
        subsPredictAI = subSubSet
    # plotSubstratePrediction(substrates=subsPredictAI, predictValues=False, scaledMatrix=True,
    #                         plotdataPCA=False, popPCA=None)


    # Evaluate predictions
    if inEvaluatePredictions:
        # simKinetics = generateKinetics(predictions=subsPredict)
        # plotPredictionAccuracy(data=simKinetics, predictionType='Enrichment')
        # plotPredictionStats(data=simKinetics, predictionType='Enrichment')

        # simKineticsScaled = generateKinetics(predictions=subsPredictScaled)
        # plotPredictionAccuracy(data=simKineticsScaled, predictionType='Scaled Enrichment')
        # plotPredictionStats(data=simKineticsScaled, predictionType='Scaled Enrichment')

        simKineticsAI = generateKinetics(predictions=subsPredictAI)
        plotPredictionAccuracy(data=simKineticsAI, predictionType='SArKS - ESM')
        plotPredictionStats(data=simKineticsAI, predictionType='SArKS - ESM')


# Load: Fixed frames
substratesFixedFrame = loadSubstratesFixedFrame()

# Evaluate substrates
if (inPlotBinnedSubstrates or inPlotBinnedSubstrateES
        or inPlotBinnedSubstratePrediction or inPlotBinnedSubstratePCA
        or inPlotEnrichmentMapPCA or inPlotEnrichmentMotifPCA
        or inPredictSubstrateActivityPCA or inPlotWordsPCA):

    # Bin substrate frames
    binnedSubs, frameTotalCountsFinal = ngs.binSubstrates(substrates=substratesFixedFrame,
                                                          substrateFrame=inFixedFramePositions,
                                                          frameIndicies=inFramePositons,
                                                          datasetTag=inSubstrateFrame)

    if inPlotBinnedSubstrates:
        ngs.plotBinnedSubstrates(substrates=binnedSubs,
                                 countsTotal=frameTotalCountsFinal,
                                 datasetTag=inSubstrateFrame,
                                 dataType='Counts',
                                 title=f'{inEnzymeName}\n{inSubstrateFrame}\n'
                                       f'Top {inPlotBinnedSubNumber} Substrates',
                                 numDatapoints=inPlotBinnedSubNumber,
                                 barColor=inBarColor,
                                 barWidth=inBarWidth)

        ngs.plotBinnedSubstrates(substrates=binnedSubs,
                                 countsTotal=frameTotalCountsFinal,
                                 datasetTag=inSubstrateFrame,
                                 dataType='Probability',
                                 title=f'{inEnzymeName}\n{inSubstrateFrame}\n'
                                       f'Top {inPlotBinnedSubNumber} Substrates',
                                 numDatapoints=inPlotBinnedSubNumber,
                                 barColor=inBarColor,
                                 barWidth=inBarWidth)

        # Plot: Work cloud
        if inPlotWordsPCA:
            ngs.plotWordCloud(clusterSubs=binnedSubs,
                              clusterIndex=None,
                              title=f'{inTitleEnrichmentMap}\n'
                                    f'{inSubstrateFrame}',
                              saveTag=f'Binned Substrates - {inSubstrateFrame}')


    if (inPlotBinnedSubstratePCA or inPlotEnrichmentMapPCA or inPlotEnrichmentMotifPCA
            or inPredictSubstrateActivityPCA or inPlotWordsPCA):
        subPopulations = []

        # Convert substrate data to numerical
        tokensESM, subsESM, subCountsESM = ngs.ESM(substrates=binnedSubs,
                                                   collectionNumber=inTotalSubsPCA,
                                                   useSubCounts=inEncludeSubstrateCounts,
                                                   subPositions=inFixedFramePositions,
                                                   datasetTag=inSubstrateFrame)

        # Cluster substrates
        subPopulations = ngs.PCA(substrates=binnedSubs, data=tokensESM, indices=subsESM,
                                 numberOfPCs=inNumberOfPCs, fixedTag=inSubstrateFrame,
                                 N=subCountsESM, fixedSubs=True, figSize=inFigureSize,
                                 saveTag=inSubstrateFrame)

        # Plot: Substrate clusters
        if (inPlotEnrichmentMapPCA or inPlotEnrichmentMotifPCA
                or inPredictSubstrateActivityPCA or inPlotWordsPCA):
            clusterCount = len(subPopulations)
            for index, subCluster in enumerate(subPopulations):
                # Plot data
                plotSubstratePopulations(clusterSubs=subCluster, clusterIndex=index,
                                         numClusters=clusterCount)



    if inPlotBinnedSubstrateES or inPlotBinnedSubstratePrediction:
        # Verify that the file exists
        if os.path.exists(filePathBinnedSubsES):
            # Load the data
            with open(filePathBinnedSubsES, 'rb') as file:
                enrichedFrames = pk.load(file)
        else:
            # Obtain data: Initial sort
            if os.path.exists(filePathFixedFrameInitial):
                # Load the data
                with open(filePathFixedFrameInitial, 'rb') as file:
                    frameProbInitial = pk.load(file)

                # Calculate: Sample size
                frameTotalCountsInitial = pd.read_csv(filePathFixedFrameInitialTotalCounts,
                                                      header=None)
                frameTotalCountsInitial = int(frameTotalCountsInitial.iloc[0, 0])
            else:
                # Fix frame
                frameCountsInitial, frameTotalCountsInitial = fixInitialSubs(
                    substrates=substratesInitial)

                # Evaluate: Probability
                frameProbInitial = substrateProbability(substrates=frameCountsInitial,
                                                        N=frameTotalCountsInitial,
                                                        sortType='Initial Sort')

                # Save the data
                with open(filePathFixedFrameInitial, 'wb') as file:
                    pk.dump(frameProbInitial, file)
                frameTotalCountsInitial = pd.DataFrame([frameTotalCountsInitial])
                frameTotalCountsInitial.to_csv(filePathFixedFrameInitialTotalCounts,
                                               index=False, header=False)
                frameTotalCountsInitial = int(frameTotalCountsInitial.iloc[0, 0])

            # Obtain data: Final sort
            if os.path.exists(filePathFixedFrameFinal):
                # Load the data
                with open(filePathFixedFrameFinal, 'rb') as file:
                    frameProbFinal = pk.load(file)
            else:
                # Evaluate: Probability
                frameProbFinal = substrateProbability(substrates=binnedSubs,
                                                      N=frameTotalCountsInitial,
                                                      sortType='Final Sort')
                with open(filePathFixedFrameFinal, 'wb') as file:
                    pk.dump(frameProbFinal, file)


            # Calculate: ES
            enrichedFrames = subFrameEnrichment(substratesInitial=frameProbInitial,
                                                initialN=frameTotalCountsInitial,
                                                substratesFinal=frameProbFinal)

            # Save the fixed substrate dataset
            if inSaveBinnedSubES:
                with open(filePathBinnedSubsES, 'wb') as file:
                    pk.dump(enrichedFrames, file)
                print(f'Binned substrate ES saved at:\n'
                      f'     {filePathBinnedSubsES}\n\n')

    # Plot: Binned enrichemt bar graph
    if inPlotBinnedSubstrateES:
        ngs.plotBinnedSubstrates(substrates=binnedSubs,
                                 countsTotal=frameTotalCountsFinal,
                                 datasetTag=inSubstrateFrame,
                                 dataType='ES',
                                 title=f'{inEnzymeName}\n{inSubstrateFrame}\n'
                                       f'Top {inPlotBinnedSubNumber} Substrates',
                                 numDatapoints=inPlotBinnedSubNumber,
                                 barColor=inBarColor,
                                 barWidth=inBarWidth)

    # Predict: Binned enrichemt
    if inPlotBinnedSubstratePrediction:
        matrixTypeEnrichment = 'Enrichment Scores Test' # 'log2(RF Final / RF Initial)'
        predictSubstrateEnrichment(substratesEnriched=enrichedFrames,
                                   matrix=fixedFrameES,
                                   matrixType=matrixTypeEnrichment)
