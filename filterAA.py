# PURPOSE: This code will load in your extracted substrates for processing

# IMPORTANT: Process all of your data with extractSubstrates before using this script


from functions import filePaths, NGS
import numpy as np
import os
import sys
import pandas as pd



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inPathFolder = f'/path/{inEnzymeName}'
inSaveFigures = True

# Input 2: Computational Parameters
inFixResidues = True
inFixedResidue = ['Q']
inFixedPosition = [4]
inExcludeResidues = False
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inMinimumSubstrateCount = 10
inPrintFixedSubs = True
inShowSampleSize = True
inEvaluateSubstrateEnrichment = False

# Input 3: Making Figures
inPlotEntropy = True
inPlotEnrichmentMap = True
inPlotEnrichmentMapScaled = True
inPlotLogo = True
inPlotWeblogo = True
inPlotWordCloud = True
inPlotPCA = False
inPlotAADistribution = False
inCodonSequence = 'NNS' # Baseline probs of degenerate codons (can be N, S, or K)
inPlotCountsAA = False
inPlotPositionalProbDist = False # For understanding shannon entropy

# Input 4: Inspecting The Data
inPrintNumber = 10

# Input 5: Motif Extraction
inMinDeltaS = 0.55

# Input 6: PCA
inPCAMotif = True
inNumberOfPCs = 2
inTotalSubsPCA = 10000
inIncludeSubCountsESM = False
inExtractPopulations = False
inPlotEntropyPCAPopulations = False
inAdjustZeroCounts = False # Prevent counts of 0 in PCA EM & Motif

# Input 7: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False
inYLabelEnrichmentMap = 2 # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter

# Input 8: Plot Sequence Motif
inNormLetters = True # Equal letter heights fixed for fixed AAs
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inBigLettersOnTop = False

# Input 9: Word Cloud
inLimitWords = True
inNWords = 100

# Input 10: Optimal Substrates
inEvaluateOS = False
inPrintOSNumber = 10
inMaxResidueCount = 4

# Input 11: Evaluate Substrate Enrichment
inEvaluateSubstrateEnrichment = False # ============= Fix: Load Initial Subs =============
inSaveEnrichedSubstrates = False
inNumberOfSavedSubstrates = 10**6

# Input 12: Evaluate Positional Preferences
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
enzymeName, filesInitial, filesFinal, labelAAPos = filePaths(enzyme=inEnzymeName)



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=enzymeName, substrateLength=len(labelAAPos),
          filterSubs=inFixResidues, fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          figEMSquares=inShowEnrichmentAsSquares, xAxisLabels=labelAAPos,
          printNumber=inPrintNumber, showNValues=inShowSampleSize,
          bigAAonTop=inBigLettersOnTop, findMotif=False, folderPath=inPathFolder,
          filesInit=filesInitial, filesFinal=filesFinal, plotPosS=inPlotEntropy,
          plotFigEM=inPlotEnrichmentMap, plotFigEMScaled=inPlotEnrichmentMapScaled,
          plotFigLogo=inPlotLogo, plotFigWebLogo=inPlotWeblogo,
          plotFigWords=inPlotWordCloud, saveFigures=inSaveFigures, setFigureTimer=None)



# ====================================== Load Data =======================================
# Get dataset tag
fixedSubSeq = ngs.getDatasetTag()

# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort')

# Calculate: RF
initialRF = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
                            fileType='Initial Sort')

loadFilteredSubs = False
filePathFixedCountsFinal, filePathFixedSubsFinal = None, None
countsFinal, countsFinalTotal = None, None
if inFixResidues:
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
    # Load: Substrates
    (substratesInitial, totalSubsInitial,
     substratesFinal, totalSubsFinal) = ngs.loadUnfilteredSubs()

    # Load: Counts
    countsFinal, countsFinalTotal = ngs.loadCounts(fileType='Final Sort', filter=False)



# ================================== Evaluate The Data ===================================
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
ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalTotal)

# Calculate: Average initial RF
initialRFAvg = np.sum(initialRF, axis=1) / len(initialRF.columns)
initialRFAvg = pd.DataFrame(initialRFAvg, index=initialRFAvg.index,
                            columns=['Average RF'])

# Calculate: RF
finalRF = ngs.calculateRF(counts=countsFinal, N=countsFinalTotal, fileType='Final Sort')


# Calculate: Positional entropy
entropy = ngs.calculateEntropy(probability=finalRF)

# Calculate: Enrichment scores
enrichmentScores = ngs.calculateEnrichment(probInitial=initialRF, probFinal=finalRF)

if inPlotWordCloud or inPlotPCA:
    # Extract motif
    ngs.identifyMotif(entropy=entropy, minEntropy=inMinDeltaS, fixFullFrame=True)
    finalSubsMotif = ngs.getMotif(substrates=substratesFinal)

    # Plot: Work cloud
    if inPlotWordCloud:
        if inFixResidues:
            ngs.plotWordCloud(substrates=finalSubsMotif, indexSet=None,
                              limitWords=inLimitWords, N=inNWords,
                              saveTag=ngs.datasetTag)
        else:
            ngs.plotWordCloud(substrates=substratesFinal, indexSet=None,
                              limitWords=inLimitWords, N=inNWords,
                              saveTag='Unfiltered')

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



# ========================================================================================
if inEvaluateOS:
    print('============================== Evaluate Optimal Substrates '
          '==============================')
    if inFixResidues:
        # Determine the OS
        combinations = 1
        optimalAA = []
        substratesOS = {}
        for indexColumn, column in enumerate(heights.columns):
            # Find the best residues at this position
            optimalAAPos = heights[column].nlargest(inMaxResidueCount)

            # Filter the data
            for rank, (AA, ES) in (
                    enumerate(zip(optimalAAPos.index, optimalAAPos.values), start=1)):
                if ES <= 0:
                    optimalAAPos = optimalAAPos.drop(index=AA)
            optimalAA.append(optimalAAPos)
        print(f'Optimal Residues: {inEnzymeName} - Fixed {fixedSubSeq}')
        for index, data in enumerate(optimalAA, start=1):
            # Determine the number of variable residues at this position
            numberAA = len(data)
            combinations *= numberAA

            # Define substrate position
            positionSub = labelAAPos[index-1]
            print(f'Position:{purple} {positionSub}{resetColor}')

            for AA, datapoint in data.items():
                print(f'     {AA}:{white} {datapoint:.6f}{resetColor}')
            print('')
        print(f'Possible Substrate Combinations:{greyDark} {combinations:,}'
              f'{resetColor}\n\n')

        # Use the optimal residues to determine OS
        substrate = ''
        score = 0
        for index, data in enumerate(optimalAA, start=1):
            # Select the top-ranked AA for each position
            topAA = data.idxmax()
            topES = data.max()

        # Construct the OS
        substrate += ''.join(topAA)
        score += topES

        # Update OS dictionary
        substratesOS[substrate] = score
        # print(f'{white}OS{resetColor}:{white} {substrate}{resetColor}, '
        #       f'{white}ES{resetColor}: {score:.6f}\n\n')

        # Create additional substrates
        # print(f'========== Create substrates ==========')
        for indexColumn, column in enumerate(heights.columns):
            # print(f'Column Index:{white} {indexColumn}{resetColor}, '
            #       f'Column:{white} {column}{resetColor}')

            # Collect new substrates to add after the iteration
            newSubstratesList = []

            for substrate, ESMax in list(substratesOS.items()):
                # print(f'Current Substrates:\n'
                #       f'     Substrate:{purple} {substrate}{resetColor}, '
                #       f'ES:{white} {ESMax}{resetColor}\n')
                AAOS = substrate[indexColumn]
                ESOS = optimalAA[indexColumn][AAOS]

                # Access the correct filtered data for the column
                optimalAAPos = optimalAA[indexColumn]
                for AA, ES in optimalAAPos.items():
                    if AA != AAOS:
                        # print(f'New Residue:{white} {AA}{resetColor}, '
                        #       f'ES:{white} {ES:.6f}{resetColor}\n'
                        #       f'     Replaced Residue:{white} {AAOS}{resetColor}, '
                        #       f'ES:{white} {ES:.6f}{resetColor}\n')

                        # Replace AAOS with AA
                        newSubstrate = substrate[:indexColumn] + AA + substrate[indexColumn + 1:]
                        newES = ESMax + (ES - ESOS)
                        # print(f'     New Substrate:{greyDark} {newSubstrate}{resetColor}, '
                        #       f'ES:{white} {newES}{resetColor}\n'
                        #       f'          ES New:{white} {ES}{resetColor}\n'
                        #       f'          ES Old:{white} {ESOS}{resetColor}\n\n')

                        # Collect new substrate and ES to add later
                        newSubstratesList.append((newSubstrate, newES))
            # Update substratesOS with new substrates after the iteration
            for newSubstrate, newES in newSubstratesList:
                substratesOS[newSubstrate] = newES

        substratesOS = sorted(substratesOS.items(), key=lambda x: x[1], reverse=True)
        print(f'Top {inPrintOSNumber} Optimal Substrates:')
        for i, (substrate, ES) in enumerate(substratesOS[:inPrintOSNumber], start=1):
            print(f'     Substrate:{white} {substrate}{resetColor}, '
                  f'ES:{white} {ES:.6f}{resetColor}')

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
    # Plot: AA probabilities in initial & final sorts
    ngs.plotLibraryProbDist(probInitial=initialRF, probFinal=finalRF,
                            codonType=inCodonSequence, datasetTag=fixedSubSeq)

    # Evaluate: Degenerate codon probabilities
    codonProbs = ngs.calculateProbCodon(codonSeq=inCodonSequence)
    # Plot: Codon probabilities
    ngs.plotLibraryProbDist(probInitial=None, probFinal=codonProbs,
                            codonType=inCodonSequence, datasetTag=inCodonSequence)

if inPlotCountsAA:
    # Plot the data
    ngs.plotCounts(countedData=countsFinal, totalCounts=countsFinalTotal)

if inPlotPositionalProbDist:
    ngs.plotPositionalProbDist(probability=finalRF, entropyScores=entropy,
                               sortType='Final Sort', datasetTag=fixedSubSeq)

if inPlotPosProb:
    ngs.compairRF(probInitial=initialRF, probFinal=finalRF, selectAA=inCompairAA)
    ngs.boxPlotRF(probInitial=initialRF, probFinal=finalRF, selectAA=inCompairAA)
