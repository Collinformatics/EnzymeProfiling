from functions import getFileNames, NGS
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pk
import random
from sklearn.metrics import r2_score
import sys



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inPathFolder = f'Enzymes/{inEnzymeName}'
inSaveFigures = True
inSetFigureTimer = False

# Input 2: Experimental Parameters
inMotifPositions = ['P4','P3','P2','P1','P1\''] # ,'P2\''
# inMotifPositions = ['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']
# inMotifPositions = ['Pos 1', 'Pos 2', 'Pos 3', 'Pos 4', 'Pos 5', 'Pos 6', 'Pos 7']
inIndexNTerminus = 0 # Define the index if the first AA in the binned substrate

# Input 3: Computational Parameters
inPlotOnlyWords = True
inFixedResidue = ['Q']
inFixedPosition = [4,5,6]
inExcludeResidues = False
inExcludedResidue = ['A','A']
inExcludedPosition = [9,10]
inMinimumSubstrateCount = 1
inCodonSequence = 'NNS' # Baseline probs of degenerate codons (can be N, S, or K)
inUseCodonProb = False # Use AA prob from inCodonSequence to calculate enrichment
inAvgInitialProb = True

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
    #inPlotMotifEnrichment = False
    #inPlotWordCloud = True
inPlotStats = False
inPlotBarGraphs = True
inPlotPCA = False # PCA plot of the combined set of motifs
inPlotSuffixTree = True
inPlotActivityFACS = False
inPredictSubstrateActivity = False
inPredictSubstrateActivityPCA = False
inPlotBinnedSubstrateES = False
inPlotBinnedSubstratePrediction = False
inPlotCounts = False
inShowSampleSize = True # Include the sample size in your figures

# Input 5: Processing The Data
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
inLimitYAxis = True

# Input 8: Motif Enrichment
inPlotNBars = 50

# Input 9: Word Cloud
inLimitWords = True
inTotalWords = inPlotNBars

# Input 10: PCA
inNumberOfPCs = 2
inTotalSubsPCA = int(5*10**4)
inIncludeSubCountsESM = True
inPlotEntropyPCAPopulations = False

# Input 11: Predict Activity
inPredictActivity = False
inPredictionTag = 'pp1a/b Substrates'
inPredictSubstrates = ['AVLQSGFR', 'VTFQSAVK', 'ATVQSKMS', 'ATLQAIAS',
                       'VKLQNNEL', 'VRLQAGNA', 'PMLQSADA', 'TVLQAVGA',
                       'ATLQAENV', 'TRLQSLEN', 'PKLQSSQA']
inPredictionTag = 'Substrates'
inPredictSubstrates = ['AVLQSGFR', 'VILQAGFR', 'VILQAPFR', 'LVLQSNDL',
                       'ATLQGLMI', 'TVLQAAML', 'VSLQSTYK', 'VSLQGAEL']
# inPredictionTag = 'FP14-18'
# inPredictSubstrates = ['AVLQSGFR', 'TVLQAAMH', 'VLLQGCVH',
#                        'WVLQAKLL', 'AILQCMLG', 'VLLQGVVH']
# inPredictionTag = 'FP19-23'
# inPredictSubstrates = ['AVLQSGFR', 'CILQAVFH', 'VVLQAVMH',
#                        'SILQCVLM', 'VMLQAVFH', 'PLLQAILM']
# inPredictionTag = 'Heatmap Substrates'
# inPredictSubstrates = ['AVLQSGFR', 'VILQSGFR', 'VILQSPFR', 'VILHSGFR', 'VIMQSGFR',
#                        'VPLQSGFR', 'NILQSGFR', 'VILQTGFR', 'PILQSGFR', 'PIMQSGFR']
inRankScores = False
inScalePredMatrix = False # Scale EM by ΔS


# Input 12: Codon Enrichment
inPredictCodonsEnrichment = False

# Input 13: Printing The Data
inPrintLoadedSubs = True
inPrintSampleSize = True
inPrintCounts = True
inPrintRF = True
inPrintES = True
inPrintEntropy = True
inPrintMotifData = True
inPrintNumber = 10

# Input 14: Evaluate Known Substrates
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

# Input 15: Evaluate Binned Substrates
inPlotEnrichedSubstrateFrame = False
inPrintLoadedFrames = True
inPlotBinnedSubNumber = 30
inPlotBinnedSubProb = True
inPlotBinnedSubYMax = 0.07

# Input 16: Predict Binned Substrate Enrichment
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
enzymeName, filesInitial, filesFinal, labelAAPos = getFileNames(enzyme=inEnzymeName)
# inMotifPositions = labelAAPos
motifLen = len(inMotifPositions)
motifFramePos = [inIndexNTerminus, inIndexNTerminus + motifLen]



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=enzymeName, substrateLength=len(labelAAPos),
          filterSubs=True, fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          minEntropy=None, figEMSquares=inShowEnrichmentAsSquares, xAxisLabels=labelAAPos,
          xAxisLabelsMotif=inMotifPositions, printNumber=inPrintNumber,
          showNValues=inShowSampleSize, bigAAonTop=inBigLettersOnTop,
          limitYAxis=inLimitYAxis, findMotif=False, folderPath=inPathFolder,
          filesInit=filesInitial, filesFinal=filesFinal, plotPosS=inPlotEntropy,
          plotFigEM=inPlotEnrichmentMap, plotFigEMScaled=inPlotEnrichmentMapScaled,
          plotFigLogo=inPlotLogo, plotFigWebLogo=inPlotWeblogo,
          plotFigMotifEnrich=inPlotMotifEnrichment,
          plotFigMotifEnrichSelect=inPlotMotifEnrichmentNBars,
          plotFigWords=inPlotWordCloud, wordLimit=inLimitWords, wordsTotal=inTotalWords,
          plotFigBars=inPlotBarGraphs, NSubBars=inPlotNBars, plotFigPCA=inPlotPCA,
          numPCs=inNumberOfPCs, NSubsPCA=inTotalSubsPCA, plotSuffixTree=inPlotSuffixTree,
          saveFigures=inSaveFigures, setFigureTimer=inSetFigureTimer)



# =================================== Define Functions ===================================
def fixInitialSubs(substrates):
    print('============================ Fix: Initial Substrates '
          '============================')
    print(f'Substrate Frames:{purple} Initial Sort{resetColor}')

    subMotifInitial = {}
    numberOfSubs = 0

    for substrate in substrates:
        for indexFrame in range(len(inFixedPosition)):
            subMotif = substrate[motifFramePos[0] + indexFrame:
                                 motifFramePos[-1] + indexFrame]
            if subMotif in subMotifInitial:
                subMotifInitial[subMotif] += 1
            else:
                subMotifInitial[subMotif] = 1
            numberOfSubs += 1

    # Sort the substrate frames
    subMotifInitial = dict(sorted(subMotifInitial.items(),
                                  key=lambda x: x[1], reverse=True))

    # Print the substrate frames
    iteration = 0
    for substrate, counts in subMotifInitial.items():
        print(f'{green} {substrate}{resetColor}, {pink}{counts}')
        if iteration == inPrintNumber:
            print(f'{resetColor}\n')
            break
        else:
            iteration += 1

    return subMotifInitial, numberOfSubs



def substrateProbability(substrates, N, sortType):
    print('=========================== Calculate: Substrate Probability '
          '============================')
    print(f'Substrate Frames:{purple} {sortType}')

    subMotifProb = {}
    for substrate, count in substrates.items():
        subMotifProb[substrate] = count / N

    iteration = 0

    for substrate, count in subMotifProb.items():
        print(f'{green}{substrate}{resetColor}, {pink}{count}')
        if iteration == inPrintNumber:
            print(f'{resetColor}\n')
            break
        else:
            iteration += 1

    return subMotifProb



def subMotifEnrichment(substratesInitial, initialN, substratesFinal):
    print('============================ Calculate: Substrate Enrichment '
          '============================')
    decimals = 3
    enrichedSubs = {}
    minInitialProb = 1 / initialN
    print(f'Min Initial Prob:{pink} {minInitialProb}{resetColor}\n')
    for substrate, probability in substratesFinal.items():
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



def generateKinetics(predictions):
    print('========================= Evaluate Prediction Accuracy '
          '==========================')
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



def plotPredictionStats(data, predictionType):
    substrates, avgScores, stDev, accuracy = data[0], data[3], data[4], data[5]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=inFigureSize)
    bars = ax.bar(substrates, avgScores, yerr=stDev, color='white', width=inBarWidth)
    ax.errorbar(substrates, avgScores, yerr=stDev, fmt='none',
                ecolor='black', elinewidth=inLineThickness, 
                capsize=5, capthick=inLineThickness)
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
    print('=========================== Predict Enrichment Scores '
          '===========================')
    print(f'Number of Unique Enriched Subs:'
          f'{white} {len(substratesEnriched):,}{resetColor}\n\n'
          f'Matrix:{pink} {matrixType}{green}\n{matrix}{resetColor}\n\n')

    # Predict enrichment scores
    iteration = 0
    substrates = {}
    for substrate, score in substratesEnriched.items():
        substrateScorePredicted = 0
        for index, AA, in enumerate(substrate):
            valueAA = matrix.loc[AA, inMotifPositions[index]]

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
                  f'          X Max: {round(x_max, ngs.roundVal)}\n'
                  f'          X Min: {resetColor(x_min, ngs.roundVal)}\n')

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
                          f' {pink}{avgEnrichment}{resetColor}\n'
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
            if (inExperimentalESUpperLimit >= valueEnrichment >= 
                    inExperimentalESLowerLimit):
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
            ax.annotate(substrate, (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center')

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
    print('============================ Measured FACS Activity '
          '=============================')
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



# ====================================== Load Data =======================================
# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort')

# Calculate: Initial sort probabilities
if inUseCodonProb:
    # Evaluate: Degenerate codon probabilities
    rfInitial = ngs.calculateProbCodon(codonSeq=inCodonSequence)
else:
    rfInitial = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
                                fileType='Initial Sort', calcAvg=inAvgInitialProb)
    # if len(labelAAPos) == len(inMotifPositions):
    #     rfInitial = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
    #                                    fileType='Initial Sort')
    # else:
    #     rfInitial = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
    #                                    fileType='Initial Sort', calcAvg=True)



# ===================================== Run The Code =====================================
# Get dataset tag
ngs.getDatasetTag(combinedMotifs=True, useCodonProb=inUseCodonProb, codon=inCodonSequence)

# Load: Substrates
substratesInitial, totalSubsInitial = ngs.loadUnfilteredSubs(loadInitial=True)

# Load: Substrate motifs
motifs, motifsCountsTotal, substratesFiltered = ngs.loadMotifSeqs(
    motifLabel=inMotifPositions, motifIndex=motifFramePos)

# Display current sample size
ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=motifsCountsTotal,
                     NFinalUnique=len(motifs.keys()))

# Evaluate dataset
combinedMotifs = False
if len(ngs.motifIndexExtracted) > 1:
    combinedMotifs = True


# # Evaluate: Count Matrices
# Load: Motif counts
if inPlotStats:
    countsMotifs, countsRelCombined, countsRelCombinedTotal = ngs.loadMotifCounts(
        motifLabel=inMotifPositions, motifIndex=motifFramePos, returnList=True)

    # Evaluate statistics
    if len(countsMotifs) > 1:
        ngs.fixedMotifStats(countsList=countsMotifs, initialRF=rfInitial,
                            motifFrame=inMotifPositions, datasetTag=ngs.datasetTag)
else:
    countsRelCombined, countsRelCombinedTotal = ngs.loadMotifCounts(
        motifLabel=inMotifPositions, motifIndex=motifFramePos)

# Calculate: RF
rfCombinedReleasedMotif = ngs.calculateRFCombinedMotif(
    countsCombinedMotifs=countsRelCombined)

# Calculate: Positional entropy
ngs.calculateEntropy(rf=rfCombinedReleasedMotif,
                     combinedMotifs=combinedMotifs,
                     releasedCounts=True)

# Calculate enrichment scores
ngs.calculateEnrichment(rfInitial=rfInitial, rfFinal=rfCombinedReleasedMotif,
                        combinedMotifs=combinedMotifs, releasedCounts=True)


# Predict substrate activity
if inPredictActivity:
    ngs.predictActivityHeatmap(predSubstrates=inPredictSubstrates,
                               predModel=ngs.datasetTag, predLabel=inPredictionTag,
                               releasedCounts=True, rankScores=inRankScores,
                               scaleEMap=inScalePredMatrix)

if inPredictCodonsEnrichment:
# Evaluate codon
    rfInitial = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
                                     fileType='Initial Sort', calcAvg=True)
    probCodon = ngs.calculateProbCodon(codonSeq=inCodonSequence)
    ngs.codonPredictions(codon=inCodonSequence, codonProb=probCodon, substrates=motifs)


# # Evaluate: Motif Sequences
# Count fixed substrates
motifCountsFinal, motifsCountsTotal = ngs.countResidues(substrates=motifs,
                                                        datasetType='Final Sort')

# Calculate: RF
rfMotif = ngs.calculateRF(counts=motifCountsFinal, N=motifsCountsTotal,
                            fileType='Final Sort')

# Calculate: Positional entropy
ngs.calculateEntropy(rf=rfMotif, combinedMotifs=combinedMotifs)

# Calculate: AA Enrichment
ngs.calculateEnrichment(rfInitial=rfInitial, rfFinal=rfMotif,
                        combinedMotifs=combinedMotifs)

ngs.processSubstrates(subsInit=substratesInitial, subsFinal=substratesFiltered,
                      motifs=motifs, subLabel=inMotifPositions,
                      combinedMotifs=combinedMotifs)
sys.exit()

# Set flag
subsPredict = None
if inPredictSubstrateActivityPCA:
    inPlotBarGraphs = True


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
                                              predictionMatrix=fixedMotifES,
                                              normalizeValues=inNormalizePredictions,
                                              matrixType='Enrichment Scores')
    plotSubstratePrediction(substrates=subsPredict, predictValues=True, 
                            scaledMatrix=False, plotDataPCA=False, popPCA=None)

    # Predict: Scaled Enrichment
    subsPredictScaled, yMax, yMin = predictActivity(
        substrates=subsPredict.copy(), predictionMatrix=fixedMotifESScaled,
        normalizeValues=inNormalizePredictions, matrixType='Scaled Enrichment Scores')
    plotSubstratePrediction(substrates=subsPredictScaled, predictValues=True, 
                            scaledMatrix=True, plotDataPCA=False, popPCA=None)

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
    # plotSubstratePrediction(substrates=subsPredictAI, predictValues=False, 
    #                         scaledMatrix=True, plotDataPCA=False, popPCA=None)


    # Evaluate predictions
    if inEvaluatePredictions:
        simKinetics = generateKinetics(predictions=subsPredict)
        plotPredictionAccuracy(data=simKinetics, predictionType='Enrichment')
        plotPredictionStats(data=simKinetics, predictionType='Enrichment')

        simKineticsScaled = generateKinetics(predictions=subsPredictScaled)
        plotPredictionAccuracy(data=simKineticsScaled, 
                               predictionType='Scaled Enrichment')
        plotPredictionStats(data=simKineticsScaled, predictionType='Scaled Enrichment')

        simKineticsAI = generateKinetics(predictions=subsPredictAI)
        plotPredictionAccuracy(data=simKineticsAI, predictionType='SArKS - ESM')
        plotPredictionStats(data=simKineticsAI, predictionType='SArKS - ESM')



# Evaluate substrates
if (inPlotBarGraphs or inPlotBinnedSubstrateES
        or inPlotBinnedSubstratePrediction or inPlotBarGraphs
        or inPlotEnrichmentMap or inPlotLogo
        or inPredictSubstrateActivityPCA or inPlotWordCloud):

    
    # Plot: Motifs
    if inPlotBarGraphs:
        print(f'{orange}Warning: This part of the script has not been written\n'
              f'# Plot: Motifs\nif inPlotBarGraphs:{resetColor}\n\n')

    if (inPlotBarGraphs or inPlotEnrichmentMap or inPlotLogo
            or inPredictSubstrateActivityPCA or inPlotWordCloud):
        subPopulations = []

        # Convert substrate data to numerical
        tokensESM, subsESM, subCountsESM = ngs.ESM(substrates=motifs,
                                                   collectionNumber=inTotalSubsPCA,
                                                   useSubCounts=inIncludeSubCountsESM,
                                                   subPositions=inMotifPositions,
                                                   datasetTag=ngs.labelCombinedMotifs)

        # Cluster substrates
        subPopulations = ngs.plotPCA(
            substrates=motifs, data=tokensESM, indices=subsESM, numberOfPCs=inNumberOfPCs,
            fixedTag=ngs.labelCombinedMotifs, N=subCountsESM, fixedSubs=True,
            saveTag=ngs.labelCombinedMotifs)

        # Plot: Substrate clusters
        if (inPlotEnrichmentMap or inPlotLogo
                or inPredictSubstrateActivityPCA or inPlotWordCloud):
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
                enrichedMotif = pk.load(file)
        else:
            # Obtain data: Initial sort
            if os.path.exists(filePathFixedMotifInitial):
                # Load the data
                with open(filePathFixedMotifInitial, 'rb') as file:
                    frameProbInitial = pk.load(file)

                # Calculate: Sample size
                frameTotalCountsInitial = pd.read_csv(
                    filePathFixedMotifInitialTotalCounts, header=None)
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
                with open(filePathFixedMotifInitial, 'wb') as file:
                    pk.dump(frameProbInitial, file)
                frameTotalCountsInitial = pd.DataFrame([frameTotalCountsInitial])
                frameTotalCountsInitial.to_csv(filePathFixedMotifInitialTotalCounts,
                                               index=False, header=False)
                frameTotalCountsInitial = int(frameTotalCountsInitial.iloc[0, 0])

            # Obtain data: Final sort
            if os.path.exists(filePathFixedMotifFinal):
                # Load the data
                with open(filePathFixedMotifFinal, 'rb') as file:
                    frameProbFinal = pk.load(file)
            else:
                # Evaluate: Probability
                frameProbFinal = substrateProbability(substrates=motifs,
                                                      N=frameTotalCountsInitial,
                                                      sortType='Final Sort')
                with open(filePathFixedMotifFinal, 'wb') as file:
                    pk.dump(frameProbFinal, file)


            # Calculate: ES
            enrichedMotif = subMotifEnrichment(substratesInitial=frameProbInitial,
                                                initialN=frameTotalCountsInitial,
                                                substratesFinal=frameProbFinal)

            # Save the fixed substrate dataset
            if inSaveBinnedSubES:
                with open(filePathBinnedSubsES, 'wb') as file:
                    pk.dump(enrichedMotif, file)
                print('================================= Save The Data '
                      '=================================')
                print(f'Binned substrate ES saved at:\n'
                      f'     {filePathBinnedSubsES}\n\n')

    # Plot: Binned enrichment bar graph
    if inPlotBinnedSubstrateES:
        ngs.plotBinnedSubstrates(substrates=motifs,
                                 countsTotal=frameTotalCountsFinal,
                                 datasetTag=ngs.labelCombinedMotifs,
                                 dataType='ES',
                                 title=f'{inEnzymeName}\n{ngs.labelCombinedMotifs}\n'
                                       f'Top {inPlotBinnedSubNumber} Substrates',
                                 numDatapoints=inPlotBinnedSubNumber,
                                 barColor=inBarColor,
                                 barWidth=inBarWidth)

    # Predict: Binned enrichment
    if inPlotBinnedSubstratePrediction:
        matrixTypeEnrichment = 'Enrichment Scores Test' # 'log2(RF Final / RF Initial)'
        predictSubstrateEnrichment(substratesEnriched=enrichedMotif,
                                   matrix=fixedMotifES,
                                   matrixType=matrixTypeEnrichment)
