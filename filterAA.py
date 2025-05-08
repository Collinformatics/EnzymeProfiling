# PURPOSE: This code will load in your extracted substrates for processing

# IMPORTANT: Process all of your data with extractSubstrates before using this script


from functions import filePaths, NGS
import numpy as np
import os
import sys
import pickle as pk
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
inExcludeResidues = True
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inMinimumSubstrateCount = 10
inPrintFixedSubs = True
inPlotWithSampleSize = True
inEvaluateSubstrateEnrichment = False

# Input 3: Processing The Data
inPlotEntropy = True
inPlotEnrichmentMap = True
inPlotEnrichmentMotif = True
inPlotWeblogoMotif = True
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
inBinSubsPCA = True
inNumberOfPCs = 2
inTotalSubsPCA = 10000
inIncludeSubCountsESM = False
inExtractPopulations = False
inPlotEntropyPCAPopulations = False
inAdjustZeroCounts = False # Prevent counts of 0 in PCA EM & Motif

# Input 7: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False
inTitleEnrichmentMap = inEnzymeName
inYLabelEnrichmentMap = 2 # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter

# Input 8: Plot Sequence Motif
inNormLetters = True # Equal letter heights fixed for fixed AAs
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inTitleMotif = inTitleEnrichmentMap
inBigLettersOnTop = False

# Input 9: Word Cloud
inLimitWords = True
inNWords = 50

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
fileNamesInitial, fileNamesFinal, labelAAPos = filePaths(enzyme=inEnzymeName)



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=inEnzymeName, substrateLength=len(labelAAPos),
          filterData=inFixResidues, fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          figEMSquares=inShowEnrichmentAsSquares, xAxisLabels=labelAAPos,
          printNumber=inPrintNumber, showNValues=inPlotWithSampleSize, findMotif=False,
          folderPath=inPathFolder, filesInit=fileNamesInitial, filesFinal=fileNamesFinal,
          saveFigures=inSaveFigures, setFigureTimer=None)



# =================================== Define Functions ===================================
def plotSubstratePopulations(clusterSubs, clusterIndex, numClusters):
    print('==================================== Plot PC Clusters '
          '===================================')
    print(f'Cluster Number:{white} {clusterIndex + 1}{resetColor}\n'
          f'     Total Clusters:{white} {numClusters}{resetColor}\n\n')


    # Define figure titles
    if numClusters == 1:
        figureTitleEM = (f'\n{inTitleEnrichmentMap}: PCA Population\n'
                         f'{fixedSubSeq}')
        figureTitleMotif = (f'{inTitleMotif}: PCA Population\n'
                            f'{fixedSubSeq}')
        figureTitleWordCloud = (f'{inTitleEnrichmentMap}: '
                                f'PCA Population\n{fixedSubSeq}')
        datasetTag = f'PCA Pop - {fixedSubSeq}'
    else:
        figureTitleEM = (f'\n{inTitleEnrichmentMap}: PCA Population {clusterIndex + 1}\n'
                         f'{fixedSubSeq}')
        figureTitleMotif = (f'{inTitleMotif}: PCA Population {clusterIndex + 1}\n'
                            f'{fixedSubSeq}')
        figureTitleWordCloud = (f'{inTitleEnrichmentMap}: '
                                f'PCA Population {clusterIndex + 1}\n{fixedSubSeq}')
        datasetTag = f'PCA Pop {clusterIndex + 1} - {fixedSubSeq}'

    # Count fixed substrates
    countFullSubstrate = False
    if countFullSubstrate:
        countsFinal, countsTotalFinal = ngs.countResiduesBinned(
            substrates=clusterSubs,
            positions=labelAAPosMotif)
    else:
        countsFinal, countsTotalFinal = ngs.countResiduesBinned(
            substrates=clusterSubs,
            positions=labelAAPosMotif)
    ngs.sampleSizeUpdate(NSubs=countsTotalFinal, sortType='Final Sort',
                         printCounts=inPrintSampleSize, fixedTag=fixedSubSeq)

    # Adjust the zero counts at nonfixed positions
    countsFinalAdjusted = countsFinal.copy()
    if inAdjustZeroCounts:
        for indexColumn in countsFinalAdjusted.columns:
            for AA in countsFinalAdjusted.index:
                if countsFinalAdjusted.loc[AA, indexColumn] == 0:
                    countsFinalAdjusted.loc[AA, indexColumn] = 1
        print(f'Adjusted Final Counts:{pink} {inEnzymeName}\n'
              f'{red}{countsFinalAdjusted}{resetColor}\n\n')


    # Calculate: RF
    finalRF = ngs.calculateRF(counts=countsFinal, N=countsTotalFinal,
                              fileType='Final Sort')
    finalRFAdjusted = ngs.calculateRF(counts=countsFinalAdjusted, N=countsTotalFinal,
                                      fileType='Final Sort')


    if inPlotEntropyPCAPopulations:
        # Plot: Positional entropy
        ngs.plotPositionalEntropy(entropy=entropy, fixedTag=fixedSubSeq)

    # Calculate: Enrichment scores
    fixedFramePopES = ngs.calculateEnrichment(initialSortRF=initialRFAvg,
                                              finalSortRF=finalRF)
    fixedFramePopESAdjusted = ngs.calculateEnrichment(initialSortRF=initialRFAvg,
                                                      finalSortRF=finalRFAdjusted)

    # Calculate: Enrichment scores scaled
    fixedFramePCAESScaled = pd.DataFrame(0.0, index=fixedFramePopES.index,
                                        columns=fixedFramePopES.columns)
    fixedFramePCAESScaledAdjusted = pd.DataFrame(0.0, index=fixedFramePopES.index,
                                                columns=fixedFramePopES.columns)

    # Scale enrichment scores with Shannon Entropy
    for positon in fixedFramePopES.columns:
        fixedFramePCAESScaled.loc[:, positon] = (fixedFramePopES.loc[:, positon] *
                                                entropy.loc[
                                                positon, 'ΔS'])
        fixedFramePCAESScaledAdjusted.loc[:, positon] = (
                fixedFramePopESAdjusted.loc[:, positon] *
                entropy.loc[positon, 'ΔS'])
    print(f'Motif:{greenLight} Scaled{resetColor}\n{fixedFramePCAESScaled}\n\n')
    yMax = max(fixedFramePCAESScaled[fixedFramePopES > 0].sum())
    yMin = min(fixedFramePCAESScaled[fixedFramePopES < 0].sum())


    # # Plot data
    # Plot: Enrichment Map
    ngs.plotEnrichmentScores(scores=fixedFramePopESAdjusted, dataType='Enrichment',
                             title=figureTitleEM, motifFilter=False,
                             duplicateFigure=False, saveTag=datasetTag)

    # Plot: Enrichment Map Scaled
    ngs.plotEnrichmentScores(scores=fixedFramePCAESScaledAdjusted,
                             dataType='Scaled Enrichment', title=figureTitleEM,
                             motifFilter=False, duplicateFigure=False, saveTag=datasetTag)

    # Plot: Sequence Motif
    ngs.plotMotif(data=fixedFramePCAESScaled.copy(), dataType='Scaled Enrichment',
                  bigLettersOnTop=inBigLettersOnTop, title=f'{figureTitleMotif}',
                  yMax=yMax, yMin=yMin, showYTicks=False,
                  addHorizontalLines=inAddHorizontalLines, motifFilter=False,
                  duplicateFigure=False, saveTag=datasetTag)

    # Plot: Work cloud
    ngs.plotWordCloud(clusterSubs=clusterSubs,
                      clusterIndex=clusterIndex,
                      title=figureTitleWordCloud,
                      saveTag=datasetTag)



def binSubstrates(substrates, datasetTag, index):
    print('===================================== Bin Substrates '
          '====================================')
    # Define: Substrate frame indices
    startDifference = 0
    if index != 0:
        # Evaluate previous fixed frame index
        startSubPrevious = inFixedPosition[index - 1]
        diff = inFixedPosition[index] - startSubPrevious - 1
        startDifference = inFixedPosition[index] - inFixedPosition[index - 1] + diff
    startSub = inFramePositions[0] + startDifference
    endSub = startSub + inMotifLength


    print(f'Fixed frame:{purple} {datasetTag}{resetColor}')
    iteration = 0
    for substrate, count in substrates.items():
        print(f'Substrate:{pink} {substrate}{resetColor}\n'
              f'     Frame:{greenLight} {substrate[startSub:endSub]}{resetColor}\n'
              f'     Count:{red} {count:,}{resetColor}')
        iteration += 1
        if iteration == inPrintNumber:
            print('\n')
            break

    # Bin the substrate frame
    binnedSubstrates = {}
    binnedSubstratesTotal = 0
    collectedSubs = {}
    for substrate, count in substrates.items():
        if substrate not in collectedSubs.keys():
            collectedSubs[substrate] = 1
            binnedSubstratesTotal += count
            sub = substrate[startSub:endSub]
        else:
            print(f'Drop duplicate:{white} {substrate}{resetColor},'
                  f'{red} {count}{resetColor}')

        # Add substrate frame to the substrate dictionary
        if sub in binnedSubstrates:
            binnedSubstrates[sub] += count
        else:
            binnedSubstrates[sub] = count

    return binnedSubstrates, binnedSubstratesTotal



# ====================================== Load Data =======================================
if inFixResidues:
    fixedSubSeq = ngs.genDatasetTag()
else:
    fixedSubSeq = 'Unfiltered'

# Load: Counts
countsInitial, countsTotalInitial = ngs.loadCounts(filter=False, fileType='Initial Sort')

# Calculate: RF
initialRF = ngs.calculateRF(counts=countsInitial, N=countsTotalInitial,
                            fileType='Initial Sort')

filePathFixedSubsFinal = ''
loadFilteredSubs = False
if inFixResidues:
    filePathFixedSubsFinal, filePathFixedCountsFinal = (
        ngs.getFilePath(datasetTag=fixedSubSeq))

    # Verify that the file exists
    if (os.path.exists(filePathFixedSubsFinal) and
            os.path.exists(filePathFixedCountsFinal)):

        # Load: Counts
        countsFinal, countsTotalFinal = ngs.loadCounts(filter=True,
                                                       fileType='Final Sort',
                                                       datasetTag=fixedSubSeq)

        # Load: Substrates
        substratesFinal, totalSubsFinal = ngs.loadSubstratesFiltered()

        if countsTotalFinal != totalSubsFinal:
            print(f'{orange}ERROR: '
                  f'The total number of Loaded Counts ({cyan}{countsTotalFinal:,}'
                  f'{orange}) =/= number of Total Substrates '
                  f'({cyan}{totalSubsFinal:,}{orange})\n')
            sys.exit()
    else:
        loadFilteredSubs = True

        # Load: Substrates
        substratesFinal, totalSubsFinal = ngs.loadUnfilteredSubs()
else:
    # Load: Substrates
    (substratesInitial, totalSubsInitial,
     substratesFinal, totalSubsFinal) = ngs.loadUnfilteredSubs(loadInitial=True)

    # Load: Counts
    countsFinal, countsTotalFinal = ngs.loadCounts(fileType='Final Sort', filter=False)



# ================================== Evaluate The Data ===================================
if inFixResidues:
    if loadFilteredSubs:
        # Fix AA
        substratesFinal, countsTotalFinal = ngs.fixResidue(
            substrates=substratesFinal, fixedString=fixedSubSeq,
            printRankedSubs=inPrintFixedSubs, sortType='Final Sort')

        # Count fixed substrates
        countsFinal, countsTotalFinal = ngs.countResidues(substrates=substratesFinal,
                                                          datasetType='Final Sort')

    if inEvaluateSubstrateEnrichment:
        fixedSubsInitial, totalFixedSubstratesInitial = ngs.fixResidue(
            substrates=substratesInitial, fixedString=fixedSubSeq,
            printRankedSubs=inPrintFixedSubs, sortType='Initial Sort')



# Display current sample size
ngs.sampleSizeDisplay(sortType=None, datasetTag=fixedSubSeq)

# Calculate: Average initial RF
RFsum = np.sum(initialRF, axis=1)
initialRFAvg = RFsum / len(initialRF.columns)
initialRFAvg = pd.DataFrame(initialRFAvg, index=initialRFAvg.index,
                            columns=['Average RF'])

# Calculate: RF
finalRF = ngs.calculateRF(counts=countsFinal, N=countsTotalFinal,
                          fileType='Final Sort')

# Calculate: Positional entropy
entropy = ngs.calculateEntropy(RF=finalRF, datasetTag=fixedSubSeq)


# Save the data
if inFixResidues:
    if (not os.path.exists(filePathFixedSubsFinal) or
            not os.path.exists(filePathFixedCountsFinal)):
        print(f'Filtered substrates saved at:\n'
              f'     {greenDark}{filePathFixedSubsFinal}{resetColor}\n'
              f'     {greenDark}{filePathFixedCountsFinal}{resetColor}\n\n')

        # Save the fixed substrate dataset
        with open(filePathFixedSubsFinal, 'wb') as file:
            pk.dump(substratesFinal, file)

        # Save the fixed substrate counts dataset
        countsFinal.to_csv(filePathFixedCountsFinal)



# ==================================== Plot The Data =====================================
# Plot: Positional entropy
if inPlotEntropy:
    ngs.plotPositionalEntropy(entropy=entropy, datasetTag=fixedSubSeq)


if inPlotEnrichmentMap:
    # Calculate: Enrichment scores
    enrichmentScores = ngs.calculateEnrichment(initialSortRF=initialRF,
                                               finalSortRF=finalRF)

    # Plot: Enrichment Map
    ngs.plotEnrichmentScores(scores=enrichmentScores, dataType='Enrichment',
                             title=inTitleEnrichmentMap, motifFilter=False,
                             duplicateFigure=False, saveTag=fixedSubSeq)


if inPlotEnrichmentMotif:
    # Calculate: Enrichment scores
    heights, yMax, yMin = ngs.enrichmentMatrix(
        counts=countsFinal.copy(), N=countsTotalFinal, baselineProb=initialRF,
        baselineType='Initial Sort', scaleData=True, normalizeFixedScores=inNormLetters)


    # Plot: Sequence Motif
    ngs.plotMotif(data=heights, dataType='Scaled Enrichment',
                  bigLettersOnTop=inBigLettersOnTop, title=f'{inTitleMotif}',
                  yMax=yMax, yMin=yMin, showYTicks=False,
                  addHorizontalLines=inAddHorizontalLines, motifFilter=False,
                  duplicateFigure=False, saveTag=fixedSubSeq)


if inPlotWeblogoMotif:
    weblogo, yMax, yMin = ngs.calculateWeblogo(probability=finalRF, entropy=entropy)

    if inShowWeblogoYTicks:
        # Plot: Sequence Motif
        ngs.plotMotif(
            data=weblogo, dataType='WebLogo', bigLettersOnTop=inBigLettersOnTop,
            title=inTitleMotif, yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
            addHorizontalLines=inAddHorizontalLines, motifFilter=False,
            duplicateFigure=False, saveTag=fixedSubSeq)
    else:
        ngs.plotMotif(
            data=weblogo, dataType='WebLogo', bigLettersOnTop=inBigLettersOnTop,
            title=inTitleMotif, yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
            addHorizontalLines=inAddHorizontalLines, motifFilter=False,
            duplicateFigure=False, saveTag=fixedSubSeq)


if inPlotWordCloud or inBinSubsPCA:
    if inFixResidues:
        # Extract motif
        ngs.identifyMotif(entropy=entropy, minEntropy=inMinDeltaS,
                          fixFullFrame=True, getIndices=True)
        finalSubsMotif = ngs.getMotif(substrates=substratesFinal)

        # Plot: Work cloud
        if inPlotWordCloud:
            titleWordCloud = f'{inEnzymeName}: Motif {fixedSubSeq}'
            ngs.plotWordCloud(substrates=finalSubsMotif, indexSet=None,
                              limitWords=inLimitWords, N=inNWords,
                              title=titleWordCloud, saveTag=ngs.datasetTagMotif)
    else:
        # Plot: Work cloud
        if inPlotWordCloud:
            titleWordCloud = f'{inEnzymeName}: Unfiltered'
            ngs.plotWordCloud(substrates=substratesFinal, indexSet=None,
                              limitWords=inLimitWords, N=inNWords,
                              title=titleWordCloud, saveTag='Unfiltered')



# ========================================================================================
if inEvaluateOS:
    print('============================== Evaluate Optimal Substrates '
          '==============================')
    if inFixResidues:
        # Calculate: Enrichment scores and scale with Shannon Entropy
        heights, yMax, yMin = ngs.enrichmentMatrix(
            counts=countsFinal.copy(), N=countsTotalFinal, baselineProb=initialRF,
            baselineType='Initial Sort', scaleData=True,
            normalizeFixedScores=inNormLetters)

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
                            saveData=inSaveEnrichedSubstrates, savePath=inFilePath)

    if inFixResidues:
        ngs.substrateEnrichment(initialSubs=substratesInitial, finalSubs=substratesFinal,
                                NSubs=inNumberOfSavedSubstrates,
                                saveData=inSaveEnrichedSubstrates, savePath=inFilePath)


if inPlotPCA:
    # Bin substrates before PCA, or don't
    if inBinSubsPCA:
        # Convert substrate data to numerical
        tokensESM, subsESM, subCountsESM = ngs.ESM(substrates=finalSubsMotif,
                                                   collectionNumber=int(inTotalSubsPCA),
                                                   useSubCounts=inIncludeSubCountsESM,
                                                   subPositions=labelAAPosMotif,
                                                   datasetTag=fixedSubSeqMotif)

        # Cluster substrates
        subPopulations = ngs.plotPCA(substrates=finalSubsMotif, data=tokensESM,
                                     indices=subsESM, numberOfPCs=inNumberOfPCs,
                                     fixedTag=fixedSubSeq, N=subCountsESM,
                                     fixedSubs=True, saveTag=fixedSubSeqMotif)
    else:
        # Convert substrate data to numerical
        tokensESM, subsESM, subCountsESM = ngs.ESM(substrates=substratesFinal,
                                                   collectionNumber=int(inTotalSubsPCA),
                                                   useSubCounts=inIncludeSubCountsESM,
                                                   subPositions=labelAAPos,
                                                   datasetTag=fixedSubSeq)

        # Cluster substrates
        subPopulations = ngs.plotPCA(substrates=substratesFinal, data=tokensESM,
                                 indices=subsESM, numberOfPCs=inNumberOfPCs,
                                 fixedTag=fixedSubSeq, N=subCountsESM, fixedSubs=True,
                                 saveTag=fixedSubSeq)

    # Plot: Substrate clusters
    if subPopulations != None:
        clusterCount = len(subPopulations)
        for index, subCluster in enumerate(subPopulations):
            # Plot data
            plotSubstratePopulations(clusterSubs=subCluster, clusterIndex=index,
                                     numClusters=clusterCount)
        print(f'Debug PCA')


if inPlotAADistribution:
    # Plot: AA probabilities in initial & final sorts
    ngs.plotLibraryProbDist(probInitial=initialRF, probFinal=finalRF,
                            codonType=inCodonSequence, fixedTag=fixedSubSeq)

    # Evaluate: Degenerate codon probabilities
    codonProbs = ngs.calculateProbCodon(codonSeq=inCodonSequence)
    # Plot: Codon probabilities
    ngs.plotLibraryProbDist(probInitial=None, probFinal=codonProbs,
                            codonType=inCodonSequence, fixedTag=inCodonSequence)

if inPlotCountsAA:
    # Plot the data
    ngs.plotCounts(countedData=countsFinal, totalCounts=countsTotalFinal,
                   datasetTag=fixedSubSeq)

if inPlotPositionalProbDist:
    ngs.plotPositionalProbDist(probability=finalRF, entropyScores=entropy,
                               sortType='Final Sort', datasetTag=fixedSubSeq)

if inPlotPosProb:
    ngs.compairRF(probInitial=initialRF, probFinal=finalRF, selectAA=inCompairAA)
    ngs.boxPlotRF(probInitial=initialRF, probFinal=finalRF, selectAA=inCompairAA)
