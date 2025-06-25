# PURPOSE: This code will load in your extracted substrates for processing

# IMPORTANT: Process all of your data with extractSubstrates before using this script


from functions import getFileNames, NGS
from itertools import product
import os
import sys



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inPathFolder = f'{inEnzymeName}'
inSaveFigures = True
inSetFigureTimer = False

# Input 2: Computational Parameters
inPlotOnlyWords = True
inFixResidues = True
inFixedResidue = ['Q']
inFixedPosition = [4]
inExcludeResidues = True
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inMinimumSubstrateCount = 10
inMinDeltaS = 0.6
inPrintFixedSubs = True
inShowSampleSize = True
inUseEnrichmentFactor = False
inCodonSequence = 'NNS' # Baseline probs of degenerate codons (can be N, S, or K)
inUseCodonProb = False # Use AA prob from inCodonSequence to calculate enrichment

# Input 3: Making Figures
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
    # inPlotEnrichmentMap = False
    inPlotEnrichmentMapScaled = False
    inPlotLogo = False
    inPlotWeblogo = False
    inPlotMotifEnrichment = False
    inPlotMotifEnrichmentNBars = False
    inPlotWordCloud = True
inPlotWordCloud = False # <--------------------
inPlotBarGraphs = False
inPlotPCA = False
inPlotSuffixTree = True
inPlotCounts = False
inPlotAADistribution = False
inPlotPositionalProbDist = False # For understanding shannon entropy

# Input 4: Inspecting The data
inPrintNumber = 10

# Input 6: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False
inYLabelEnrichmentMap = 2 # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter

# Input 7: Plot Sequence Motif
inNormLetters = True # Equal letter heights fixed for fixed AAs
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inBigLettersOnTop = False

# Input 8: Word Cloud
inLimitWords = True
inTotalWords = 50

# Input 9: Bar Graphs
inNSequences = 50

# Input 10: PCA
inPCAMotif = False
inNumberOfPCs = 2
inTotalSubsPCA = 10000
inIncludeSubCountsESM = False
inExtractPopulations = False
inPlotEntropyPCAPopulations = False
inAdjustZeroCounts = False # Prevent counts of 0 in PCA EM & Motif

# Input 11: Predict Activity
inPredictActivity = True
inPredictionTag = 'pp1a/b Substrates'
inPredictSubstrates = ['AVLQSGFR', 'VTFQSAVK', 'ATVQSKMS', 'ATLQAIAS',
                       'VKLQNNEL', 'VRLQAGNA', 'PMLQSADA', 'TVLQAVGA',
                       'ATLQAENV', 'TRLQSLEN', 'PKLQSSQA']

# Input 12: Optimal Substrates
inEvaluateOS = False
inPrintOSNumber = 10
inMaxResidueCount = 4

# Input 13: Evaluate Substrate Enrichment
inEvaluateSubstrateEnrichment = False # ============= Fix: Load Initial Subs =============
inSaveEnrichedSubstrates = False
inNumberOfSavedSubstrates = 10**6

# Input 14: Evaluate Positional Preferences
inPlotPosProb = False # Plot RF distributions of a given AA
inCompairAA = 'L' # Select AA of interest (different A than inFixedResidue)



# =================================== Setup Parameters ===================================
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



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=enzymeName, substrateLength=len(labelAAPos),
          filterSubs=inFixResidues, fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          minEntropy=inMinDeltaS, figEMSquares=inShowEnrichmentAsSquares,
          xAxisLabels=labelAAPos, printNumber=inPrintNumber, showNValues=inShowSampleSize,
          bigAAonTop=inBigLettersOnTop, findMotif=False, folderPath=inPathFolder,
          filesInit=filesInitial, filesFinal=filesFinal, useEF=inUseEnrichmentFactor,
          plotPosS=inPlotEntropy, plotFigEM=inPlotEnrichmentMap,
          plotFigEMScaled=inPlotEnrichmentMapScaled, plotFigLogo=inPlotLogo,
          plotFigWebLogo=inPlotWeblogo, plotFigMotifEnrich=inPlotMotifEnrichment,
          plotFigMotifEnrichSelect=inPlotMotifEnrichmentNBars,
          plotFigWords=inPlotWordCloud,  wordLimit=inLimitWords, wordsTotal=inTotalWords,
          plotFigBars=inPlotBarGraphs, NSubBars=inNSequences, plotFigPCA=inPlotPCA,
          numPCs=inNumberOfPCs, NSubsPCA=inTotalSubsPCA, plotSuffixTree=inPlotSuffixTree,
          saveFigures=inSaveFigures, setFigureTimer=inSetFigureTimer)



# ====================================== Load data =======================================
# Get dataset tag
fixedSubSeq = ngs.getDatasetTag(useCodonProb=inUseCodonProb, codon=inCodonSequence)

# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort')

# Calculate: Initial sort probabilities
if inUseCodonProb:
    # Evaluate: Degenerate codon probabilities
    probInitial = ngs.calculateProbCodon(codonSeq=inCodonSequence)
else:
    probInitial = ngs.calculateProbabilities(counts=countsInitial, N=countsInitialTotal,
                                             fileType='Initial Sort')

# substratesInitial = None
loadFilteredSubs = False
filePathFixedCountsFinal, filePathFixedSubsFinal = None, None
substratesInitial, totalSubsInitial = ngs.loadUnfilteredSubs(loadInitial=True)
countsFinal, countsFinalTotal = None, None
if inFixResidues:
    if inUseEnrichmentFactor:
        substratesInitial, totalSubsInitial = ngs.loadUnfilteredSubs(loadInitial=True)

    filePathFixedSubsFinal, filePathFixedCountsFinal = (
        ngs.getFilePath(datasetTag=fixedSubSeq))

    # Verify that the file exists
    if (os.path.exists(filePathFixedSubsFinal) and
            os.path.exists(filePathFixedCountsFinal)):

        # Load: Counts
        countsFinal, countsFinalTotal = ngs.loadCounts(filter=True,
                                                       fileType='Final Sort',
                                                       datasetTag=fixedSubSeq)

        # Load: Substrates
        substratesFinal, totalSubsFinal = ngs.loadSubstratesFiltered()

        if countsFinalTotal != totalSubsFinal:
            print(f'{orange}ERROR: '
                  f'The total number of Loaded Counts ({cyan}{countsFinalTotal:,}'
                  f'{orange}) =/= number of Total Substrates '
                  f'({cyan}{totalSubsFinal:,}{orange})\n')
            sys.exit()
    else:
        loadFilteredSubs = True

        # Load: Substrates
        substratesFinal, totalSubsFinal = ngs.loadUnfilteredSubs(loadFinal=True)
else:
    # # Load: Substrates
    # (substratesInitial, totalSubsInitial,
    #  substratesFinal, totalSubsFinal) = ngs.loadUnfilteredSubs(loadInitial=True)

    # Load: Counts
    countsFinal, countsFinalTotal = ngs.loadCounts(fileType='Final Sort', filter=False)



# ================================== Evaluate The data ===================================
if inFixResidues:
    if loadFilteredSubs:
        # Fix AA
        substratesFinal, countsFinalTotal = ngs.fixResidue(
            substrates=substratesFinal, fixedString=fixedSubSeq,
            printRankedSubs=inPrintFixedSubs, sortType='Final Sort')

        # Count fixed substrates
        countsFinal, countsFinalTotal = ngs.countResidues(substrates=substratesFinal,
                                                          datasetType='Final Sort')

    if inEvaluateSubstrateEnrichment:
        substratesInitial, totalSubsInitial = ngs.loadUnfilteredSubs(loadInitial=True)

        fixedSubsInitial, countsInitialFixedTotal = ngs.fixResidue(
            substrates=substratesInitial, fixedString=fixedSubSeq,
            printRankedSubs=inPrintFixedSubs, sortType='Initial Sort')

# Save the data
ngs.saveData(substrates=substratesFinal, counts=countsFinal)


# Display current sample size
ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalTotal,
                     NFinalUnique=len(substratesFinal.keys()))

# Calculate: RF
probFinal = ngs.calculateProbabilities(counts=countsFinal, N=countsFinalTotal,
                                       fileType='Final Sort')

# Calculate: Positional entropy
entropy = ngs.calculateEntropy(probability=probFinal)

# Calculate: Enrichment scores
enrichmentScores = ngs.calculateEnrichment(probInitial=probInitial, probFinal=probFinal)


# Evaluate: Sequences
motifs = ngs.processSubstrates(
    subsInit=substratesInitial, subsFinal=substratesFinal, motifs=substratesFinal,
    subLabel=labelAAPos)

if inPlotPCA:
    finalSubsMotif = ngs.getMotif(substrates=substratesFinal)

    if inUseEnrichmentFactor:
        if substratesInitial is None:
            print(f'{red}Write code to load in the initial substrates{resetColor}\n')
            sys.exit()

    # # Plot: Work cloud
    # if inPlotWordCloud:
    #     if inFixResidues:
    #         ngs.plotWordCloud(substrates=finalSubsMotif, plotEF=inUseEnrichmentFactor)
    #     else:
    #         ngs.plotWordCloud(substrates=substratesFinal, plotEF=inUseEnrichmentFactor)

    # Plot: PCA
    if inPlotPCA:
        if inPCAMotif:
            datasetTag = ngs.datasetTagMotif
            saveTag = ngs.saveTagMotif
            labelPos = ngs.xAxisLabelsMotif
            subsPCA = finalSubsMotif
        else:
            datasetTag =  fixedSubSeq
            saveTag = datasetTag
            labelPos = labelAAPos
            subsPCA = substratesFinal

        # Convert substrate data to numerical
        tokensESM, subsESM, subCountsESM = ngs.ESM(
            substrates=subsPCA, collectionNumber=int(inTotalSubsPCA),
            useSubCounts=inIncludeSubCountsESM, subPositions=labelPos,
            datasetTag=saveTag)

        # Cluster substrates
        subPopulations = ngs.plotPCA(substrates=finalSubsMotif, data=tokensESM,
                                     indices=subsESM, numberOfPCs=inNumberOfPCs,
                                     N=subCountsESM, fixedSubs=inFixResidues,
                                     datasetTag=datasetTag, saveTag=saveTag)

        # Plot: Substrate clusters
        if subPopulations is not None:
            clusterCount = len(subPopulations)
            for index, subCluster in enumerate(subPopulations):
                # Plot data
                ngs.plotSubstratePopulations(
                    substrates=subCluster, clusterIndex=index, numClusters=clusterCount,
                    datasetTag=datasetTag, saveTag=saveTag)

            print(f'Debug PCA')
            sys.exit()

# Predict substrate activity
if inPredictActivity:
    ngs.predictActivityHeatmap(predSubstrates=inPredictSubstrates,
                               predModel=ngs.datasetTag, predLabel=inPredictionTag)



# ========================================================================================
if inEvaluateOS:
    print('============================== Evaluate Optimal Substrates '
          '==============================')
    if inFixResidues:
        # Determine the OS
        combinations = 1
        optimalAA = []
        substratesOS = {}
        for indexColumn, column in enumerate(ngs.eMap.columns):
            # Find the best residues at this position
            optimalAAPos = ngs.eMap[column].nlargest(inMaxResidueCount)

            # Filter the data
            for rank, (AA, ES) in (
                    enumerate(zip(optimalAAPos.index, optimalAAPos.values), start=1)):
                if ES <= 0:
                    optimalAAPos = optimalAAPos.drop(index=AA)
            optimalAA.append(optimalAAPos)
        print(f'Optimal Residues: {purple}{inEnzymeName} - {fixedSubSeq}{resetColor}')
        for index, data in enumerate(optimalAA, start=1):
            # Determine the number of variable residues at this position
            numberAA = len(data)
            combinations *= numberAA

            # Define substrate position
            positionSub = labelAAPos[index-1]
            print(f'Preferred Residues:\n'
                  f'{greenLight}{data}{resetColor}\n')
        print(f'Possible Substrate Combinations: {red}{combinations:,}'
              f'{resetColor}\n\n')

        # Make all possible substrate sequences
        residueChoices = [series.index.tolist() for series in optimalAA]
        subCombos = list(product(*residueChoices))
        substrates = [''.join(combo) for combo in subCombos]

        # Predict activity
        substratesOS = {}
        for substrate in substrates:
            score = 0
            for index in range(len(substrate)):
                AA = substrate[index]
                pos = ngs.eMap.columns[index]
                score += ngs.eMap.loc[AA, pos]
            substratesOS[substrate] = score
        substratesOS = dict(sorted(substratesOS.items(),
                                   key=lambda x: x[1], reverse=True))

        print(f'Predicted Optimal Substrates:')
        for index, (substrate, ES) in enumerate(substratesOS.items()):
            print(f'     {pink} {substrate}{resetColor}, '
                  f'ES:{red} {ES:.3f}{resetColor}')
            if index >= inPrintOSNumber:
                sys.exit()

        print(f'\nNumber of substrates:{greyDark} {len(substratesOS):,}{resetColor}\n\n')



if inEvaluateSubstrateEnrichment:
    ngs.substrateEnrichment(initialSubs=substratesInitial, finalSubs=substratesFinal,
                            NSubs=inNumberOfSavedSubstrates,
                            saveData=inSaveEnrichedSubstrates)

    if inFixResidues:
        ngs.substrateEnrichment(initialSubs=substratesInitial, finalSubs=substratesFinal,
                                NSubs=inNumberOfSavedSubstrates,
                                saveData=inSaveEnrichedSubstrates)


if inPlotAADistribution:
    # Plot: AA probabilities in initial and final sorts
    ngs.plotLibraryProbDist(probInitial=probInitial, probFinal=probFinal,
                            codonType=inCodonSequence, datasetTag=fixedSubSeq)

    # Evaluate: Degenerate codon probabilities
    codonProbs = ngs.calculateProbCodon(codonSeq=inCodonSequence)
    # Plot: Codon probabilities
    ngs.plotLibraryProbDist(probInitial=probFinal, probFinal=codonProbs,
                            codonType=inCodonSequence, datasetTag=inCodonSequence,
                            skipInitial=True)

if inPlotCounts:
    # Plot the data
    ngs.plotCounts(countedData=countsFinal, totalCounts=countsFinalTotal)

if inPlotPositionalProbDist:
    ngs.plotPositionalProbDist(probability=probFinal, entropyScores=entropy,
                               sortType='Final Sort', datasetTag=fixedSubSeq)

if inPlotPosProb:
    ngs.compairRF(probInitial=probInitial, probFinal=probFinal, selectAA=inCompairAA)
    ngs.boxPlotRF(probInitial=probInitial, probFinal=probFinal, selectAA=inCompairAA)
