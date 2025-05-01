from functions import filePaths, NGS
import os
import pandas as pd
import pickle as pk
import sys
import threading



# When fixing substrates in the initial sort, use substrates with counts < 10

# Coordinate Descent & Optimization Framework



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inBasePath = f'/path/{inEnzymeName}'
inFilePath = os.path.join(inBasePath, 'Extracted Data')
inSavePathFigures = os.path.join(inBasePath, 'Figures')
inFileNamesInitialSort, inFileNamesFinalSort, inAAPositions = (
    filePaths(enzyme=inEnzymeName))
inSaveData = True
inSaveFigures = True
inSetFigureTimer = True

# Input 2: Experimental Parameters
inSubstrateLength = len(inAAPositions)
inShowSampleSize = True  # Include the sample size in your figures
inSaveScaledEnrichment = False  # Saving Scaled Enrichment Values
inSubstratePositions = inAAPositions # ['P4', 'P3', 'P2', 'P1', 'P1\'']

# Input 3: Computational Parameters
inRefixMotif = False
inFixResidues = True
inFixedResidue = ['Q'] # Only use 1 AA
inFixedPosition = [7]
inExcludeResidues = False
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inDatasetTag = f'Motif {inFixedResidue[0]}@R{inFixedPosition[0]}'
inManualEntropy = False
inManualFrame = ['R4', 'R5', 'R6', 'R2']
inFixEntireSubstrateFrame = False
inMinDeltaS = 0.6
inMinimumSubstrateCount = 10
inSetMinimumESFixAA = 0
inSetMinimumESReleaseAA = -1
inPrintFixedSubs = True
inCombineFixedMotifs = False
inPredictSubstrateEnrichmentScores = False
inDuplicateFigure = True

# Input 5: Figure Parameters
inPlotPositionalEntropy = True
inPlotEnrichmentMap = True
inPlotEnrichmentMotif = False
inPlotUnscaledScatter = False
inPlotScaledScatter = False

# Input 4: Processing The Data
inPrintLoadedSubs = True
inPrintSampleSize = True
inPrintCounts = True
inPlotCounts = False
inPrintRF = True
inPrintEntropy = True
inShowEnrichmentData = True
inShowMotifData = True
inPrintNumber = 10
inCodonSequence = 'NNS'  # Base probabilities of degenerate codons (can be N, S, or K)
inUseCodonProb = False  # If True: use "inCodonSequence" for baseline probabilities
# If False: use "inFileNamesInitialSort" for baseline probabilities

# Input 6: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False
inTitleEnrichmentMap = f'{inEnzymeName}: {inDatasetTag}'
inYLabelEnrichmentMap = 2  # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter
inPrintSelectedSubstrates = 1  # Set = 1, to print substrates with fixed residue

# Input 7: Plot Sequence Motif
inTitleMotif = inTitleEnrichmentMap
inNormLetters = False  # Normalize fixed letter heights
inPlotWeblogoMotif = False
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inPlotNegativeWeblogoMotif = False
inBigLettersOnTop = False

# Input 8: Substrate Enrichment
inBinSubstrates = False
inSaveEnrichedSubs = False
inPredictionDatapointColor = '#CC5500'
inSetAxisLimits = False

# Input 9: Predict Optimal substrates
inSetMinimumES = True
inPrintES = True
inPlotSubsetOfSubstrates = True
inShowKLDivergenceProb = False
inDatapointColor = '#CC5500'
inPlotSubstrateText = False
inMiniumSubstrateScoreLimit = False
inMiniumSubstrateScore = -55
inNormalizeValues = False
inPrintNormalizedValues = False
inPrintBinnedSubstrates = True
inMinX, inMaxX = -0.05, 1.05
inMinY, inMaxY = -0.05, 1.05  # -7.5, 15
inInspectDataUpperValue = 0.4
inInspectDataLowerValue = 0.2
inDifferenceThreshold = 0
inBinPositionRange = [1, 5]
inOverlapOSPredictions = False
if inOverlapOSPredictions:
    fixedPositions = []
    inBinPositions = []
    inDatapointColor = ['#FF0000', '#16FFD4', '#FF00F2', '#FF6200']  # '#F79620'
    for index in range(inFixedPosition[0], inFixedPosition[0] + 4):
        fixedPositions.append(index)
        addValue = index - 4
        inBinPositions.append((inBinPositionRange[0] - 1 + addValue,
                               inBinPositionRange[1] + addValue))
    inFixedPosition = fixedPositions
else:
    if inFixedPosition[0] != 4:
        addValue = inFixedPosition[0] - 4
        inBinPositionRange[0] += addValue
        inBinPositionRange[1] += addValue
inMatrixESLabel = r'Enrichment Scores'  # - log5()'
inMatrixScaledESLabel = r'ΔS * Enrichment Scores'  # - log5()'



# =============================== Setup Figure Parameters ================================
global fixedSubSeq

# Colors:
white = '\033[38;2;255;255;255m'
silver = '\033[38;2;204;204;204m'
purple = '\033[38;2;189;22;255m'
pink = '\033[38;2;255;0;242m'
cyan = '\033[38;2;22;255;212m'
green = '\033[38;2;5;232;49m'
greenLight = '\033[38;2;204;255;188m'
greenDark = '\033[38;2;30;121;13m'
yellow = '\033[38;2;255;217;24m'
orange = '\033[38;2;247;151;31m'
red = '\033[91m'
resetColor = '\033[0m'



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=inEnzymeName, substrateLength=inSubstrateLength,
          fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          figEMSquares=inShowEnrichmentAsSquares, xAxisLabels=inAAPositions,
          xAxisLabelsBinned=None, residueLabelType=inYLabelEnrichmentMap,
          printNumber=inPrintNumber, showNValues=inShowSampleSize,
          idMotif=True, savePath=inFilePath, saveFigures=inSaveFigures,
          savePathFigs=inSavePathFigures, setFigureTimer=inSetFigureTimer)



# ====================================== Load Data =======================================
def loadSubstrates(filePath, fileNames, fileType, printLoadedData, result):
    subsLoaded, totalSubs = ngs.loadSubstrates(filePath=filePath,
                                               fileNames=fileNames,
                                               fileType=fileType,
                                               printLoadedData=printLoadedData)
    result[fileType] = (subsLoaded, totalSubs)

# Initialize result dictionary
loadedResults = {}

# Create threads for loading initial and final substrates
# threadInitial = threading.Thread(target=loadSubstrates,
#                                  args=(inFilePath, inFileNamesInitialSort,
#                                        'Initial Sort', inPrintCounts, loadedResults))
threadFinal = threading.Thread(target=loadSubstrates,
                               args=(inFilePath, inFileNamesFinalSort, 'Final Sort',
                                     inPrintCounts, loadedResults))

# Start the threads
# threadInitial.start()
threadFinal.start()

# Wait for the threads to complete
# threadInitial.join()
threadFinal.join()
# time.sleep(0.5)

# Retrieve the loaded substrates
# substratesInitial, totalSubsInitial = loadedResults['Initial Sort']
substratesFinal, totalSubsFinal = loadedResults['Final Sort']


# Load Data: Initial sort
filePathsInitial = []
if '/' in inFilePath:
    for fileName in inFileNamesInitialSort:
        filePathsInitial.append(f'{inFilePath}/counts_{fileName}')
else:
    for fileName in inFileNamesInitialSort:
        filePathsInitial.append(f'{inFilePath}\\counts_{fileName}')

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
    print(f'\nMake sure your path is correctly named, and that you have already '
          f'extracted and counted your NGS data\n')
    sys.exit()
else:
    countsInitial, totalSubsInitial = ngs.loadCounts(filePath=filePathsInitial,
                                                     files=inFileNamesInitialSort,
                                                     printLoadedData=inPrintCounts,
                                                     fileType='Initial Sort')
    # Calculate RF
    initialRF = ngs.calculateRF(counts=countsInitial, N=totalSubsInitial,
                                fileType='Initial Sort', printRF=inPrintRF)

# Update: Current Sample Size
ngs.updateSampleSize(NSubs=totalSubsInitial, sortType='Initial Sort',
                     printCounts=inPrintSampleSize, fixedTag=None)
ngs.updateSampleSize(NSubs=totalSubsFinal, sortType='Final Sort',
                     printCounts=inPrintSampleSize, fixedTag=None)



# =================================== Define Functions ===================================
def fixSubstrate(subs, fixedAA, fixedPosition, exclude, excludeAA, excludePosition,
                 sortType, fixedTag, initialFix):
    print('==================================== Fix Substrates '
          '=====================================')
    print(f'Substrate Dataset:'
          f'{purple} {inEnzymeName}{resetColor} - {purple}{sortType}{resetColor}\n')
    print(f'Selecting substrates with:')
    for index in range(len(fixedAA)):
        AA = ', '.join(fixedAA[index])
        print(f'     {purple}{AA}{resetColor}@{purple}R{fixedPosition[index]}{resetColor}')
    print('')
    if exclude:
        print(f'Excluding substrates with:')
        for index in range(len(excludeAA)):
            AA = ', '.join(excludeAA[index])
            print(f'     {pink}{AA}{resetColor}@{pink}R{excludePosition[index]}'
                  f'{resetColor}')
        print('')


    # Initialize data structures
    fixedSubs = {}
    fixedSubsTotal = 0

    # Prepare dataset type label
    sortTypePathTag = sortType.replace(' ', '')
    fixedTag = fixedTag.replace(' ', '')


    # # Load Data
    # Define: File path
    filePathFixedSubs = os.path.join(
        inFilePath, f'fixedSubs - {inEnzymeName} - {fixedTag} - '
                    f'{sortTypePathTag} - MinCounts_{inMinimumSubstrateCount}')
    filePathFixedCounts = os.path.join(
        inFilePath, f'counts - {inEnzymeName} - {fixedTag} - '
                    f'{sortTypePathTag} - MinCounts_{inMinimumSubstrateCount}')

    # Determine if the fixed substrate file exists
    loadFiles = False
    if os.path.exists(filePathFixedSubs) and os.path.exists(filePathFixedCounts):
        loadFiles = True
        print(f'Loading Substrates at Path:\n'
              f'     {filePathFixedSubs}\n'
              f'     {filePathFixedCounts}\n\n')

        # Load Data: Fixed substrates
        with open(filePathFixedSubs, 'rb') as file:
            fixedSubs = pk.load(file)

        # Load Data: Fixed counts
        fixedCounts = pd.read_csv(filePathFixedCounts, index_col=0)

        # Calculate total counts
        fixedCountsTotal = sum(fixedCounts.iloc[:, 0])
    else:
        # Fix the substrates if the files were not found
        print(f'Fixing substrates...\n\n')
        if exclude:
            # Fix AA
            if len(fixedAA) == 1:
                for substrate, count in subs.items():
                    keepSub = True

                    # Evaluate substrate
                    for indexExclude, AAExclude in enumerate(excludeAA):
                        if len(AAExclude) == 1:
                            indexRemoveAA = excludePosition[indexExclude] - 1

                            # Is the AA acceptable?
                            if substrate[indexRemoveAA] == AAExclude:
                                keepSub = False
                                continue
                        else:
                            # Remove Multiple AA at a specific position
                            for AAExcludeMulti in AAExclude:
                                indexRemoveAA = excludePosition[indexExclude] - 1
                                for AAExclude in AAExcludeMulti:

                                    # Is the AA acceptable?
                                    if substrate[indexRemoveAA] == AAExclude:
                                        keepSub = False
                                        continue

                    # If the substrate is accepted, look for the desired AA
                    if keepSub:
                        if len(fixedAA) == 1 and len(fixedAA[0]) == 1:
                            # Fix only one AA
                            if substrate[fixedPosition[0] - 1] != fixedAA[0]:
                                keepSub = False
                        else:
                            for indexFixed, fixAA in enumerate(fixedAA):
                                indexFixAA = fixedPosition[indexFixed] - 1
                                if len(fixAA) == 1:
                                    # Fix one AA at a given position
                                    if substrate[indexFixAA] != fixAA:
                                        keepSub = False
                                        break
                                else:
                                    # Fix multiple AAs at a given position
                                    if substrate[indexFixAA] not in fixAA:
                                        keepSub = False
                                        break

                    # Select the substrate
                    if keepSub:
                        fixedSubs[substrate] = count
                        fixedSubsTotal += count
            else:
                for substrate, count in subs.items():
                    keepSub = False
                    saveSeq = []

                    # Evaluate substrate
                    for indexExclude, AAExclude in enumerate(excludeAA):
                        if len(AAExclude) == 1:
                            indexRemoveAA = excludePosition[indexExclude] - 1

                            # Is the AA acceptable?
                            if substrate[indexRemoveAA] == AAExclude:
                                saveSeq.append(False)
                                continue
                        else:
                            # Remove Multiple AA at a specific position
                            for AAExcludeMulti in AAExclude:
                                indexRemoveAA = excludePosition[indexExclude] - 1
                                for AAExclude in AAExcludeMulti:

                                    # Is the AA acceptable?
                                    if substrate[indexRemoveAA] == AAExclude:
                                        saveSeq.append(False)
                                        continue

                    # If the substrate is accepted, look for the desired AA
                    if False not in saveSeq:
                        for index in range(len(fixedAA)):
                            foundAA = False
                            for multiAA in fixedAA[index]:
                                if substrate[fixedPosition[index] - 1] == multiAA:
                                    foundAA = True
                                    break

                            # Discard Substrate
                            if not foundAA:
                                saveSeq.append(False)
                                break

                        if False not in saveSeq:
                            if count >= inMinimumSubstrateCount:
                                fixedSubs[str(substrate)] = count
                                fixedSubsTotal += count
        else:
            # Fix AA
            if len(fixedAA) == 1:
                for substrate, count in subs.items():
                    if substrate[fixedPosition[0] - 1] == fixedAA[0]:
                        if count >= inMinimumSubstrateCount:
                            fixedSubs[substrate] = count
                            fixedSubsTotal += count
                            continue
            else:
                for substrate, count in subs.items():
                    saveSeq = []
                    for index in range(len(fixedAA)):
                        foundAA = False
                        for multiAA in fixedAA[index]:
                            if substrate[fixedPosition[index] - 1] == multiAA:
                                foundAA = True
                                break

                        # Discard Substrate
                        if not foundAA:
                            saveSeq.append(False)
                            break

                    if False not in saveSeq:
                        if count >= inMinimumSubstrateCount:
                            fixedSubs[str(substrate)] = count
                            fixedSubsTotal += count
                            continue

        # Print fixed substrates
        if inPrintFixedSubs:
            iteration = 0
            print(f'Selected Substrates:')
            for substrate, count in fixedSubs.items():
                print(f'     Substrate:{silver} {substrate}{resetColor}\n'
                      f'         Count:{red} {count:,}{resetColor}')
                iteration += 1
                if iteration == inPrintNumber:
                    break
        print('\n')

        # Count fixed substrates
        fixedCounts, fixedCountsTotal = ngs.countResidues(substrates=fixedSubs,
                                                          datasetType='Final Sort',
                                                          printCounts=inPrintCounts)

        # Save the fixed substrate dataset
        with open(filePathFixedSubs, 'wb') as file:
            pk.dump(fixedSubs, file)

        # Save the counted substrate data
        fixedCounts.to_csv(filePathFixedCounts, index=True, float_format='%.0f')

        # Print your save location
        print(f'Fixed substrate data saved at:\n'
              f'     {filePathFixedSubs}\n'
              f'     {filePathFixedCounts}\n\n')

    return fixedSubs, fixedCounts, fixedCountsTotal



def fixFrame(substrates, fixRes, fixPos, sortType, datasetTag):
    print('============================== Fix Substrate Frame '
          '==============================')
    print(f'Save Iteration: {ngs.saveFigureIteration}')
    print(f'Dataset:{purple} {inEnzymeName}{resetColor} - '
          f'{purple}{sortType}{resetColor}\n'
          f'Fixed Residue:{white} {fixRes}{resetColor}\n'
          f'Fixed Position:{white} {fixPos}{resetColor}\n'
          f'Minimum Substrate Count:{white} {inMinimumSubstrateCount}{resetColor}\n'
          f'Starting with:\n'
          f'     {white}{totalSubsFinal:,}{resetColor} total substrates\n'
          f'     {white}{len(substrates):,}{resetColor} unique substrates\n\n')

    if len(fixRes) != 1:
        print(f'{orange}ERROR:\n'
              f'     You can only fix 1 AA in the first iteration\n'
              f'     You attempted to fix{resetColor}:{white} {fixRes}{orange}\n\n'
              f'Trim the inFixedResidue list down to 1 AA & try again.\n')
        sys.exit()

    if inCombineFixedMotifs:
        print(f'{orange}ERROR: inCombinedFixedMotifs ='
              f'{white} {inCombineFixedMotifs}{orange}]n'
              f'Write this code!\n')
        sys.exit()
    else:
        # Make fixed seq tag
        fixedSubSeq = ngs.genDatasetTag()

    # # Fix The First Set Of Substrates
    fixedSubsFinal, fixedCountsFinal, countsTotalFixedMotif = fixSubstrate(
        subs=substrates, fixedAA=fixRes, fixedPosition=fixPos,
        exclude=inExcludeResidues, excludeAA=inExcludedResidue,
        excludePosition=inExcludedPosition, sortType='Final Sort',
        fixedTag=fixedSubSeq, initialFix=True)

    initialFixedPos = inAAPositions[inFixedPosition[0] - 1]

    # Calculate RF
    finalFixedRF = ngs.calculateRF(counts=fixedCountsFinal, N=countsTotalFixedMotif,
                                   fileType='Final Sort', printRF=inPrintRF)

    # Determine substrate frame
    positionalEntropy = ngs.calculateEntropy(RF=finalFixedRF,
                                             printEntropy=inPrintEntropy)
    if inManualEntropy:
        substrateFrameSorted = pd.DataFrame(1, index=inManualFrame,
                                            columns=['ΔEntropy'])
        print(f'Ranked Substrate Frame:{green} User Defined{purple}\n'
              f'{substrateFrameSorted}{resetColor}\n\n')
    else:
        substrateFrameSorted = ngs.findSubstrateFrame(entropy=positionalEntropy,
                                                      minEntropy=inMinDeltaS,
                                                      fixFullFrame=
                                                      inFixEntireSubstrateFrame,
                                                      getIndices=False)

    if inPlotPositionalEntropy:
        # Visualize: Change in Entropy
        ngs.plotPositionalEntropy(entropy=positionalEntropy, fixedDataset=inFixResidues,
                                  fixedTag=fixedSubSeq, avgDelta=False)

    # Update: Current Sample Size
    ngs.updateSampleSize(NSubs=countsTotalFixedMotif,
                         sortType=f'{purple}Final Sort{resetColor} - {purple}Fixed',
                         printCounts=inPrintSampleSize, fixedTag=fixedSubSeq)

    # Calculate enrichment scores
    finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                           finalSortRF=finalFixedRF,
                                           printES=inShowEnrichmentData)

    # Create released matrix df
    releasedCounts = pd.DataFrame(0.0, index=finalFixedES.index,
                                  columns=finalFixedES.columns)

    if inPlotEnrichmentMap:
        # Plot: Enrichment Map
        ngs.plotEnrichmentScores(scores=finalFixedES, dataType='Enrichment',
                                 title=inTitleEnrichmentMap, motifFilter=True,
                                 duplicateFigure=inDuplicateFigure, saveTag=datasetTag)
    # sys.exit()

    if inPlotEnrichmentMotif:
        # Calculate enrichment scores and scale with Shannon Entropy
        pType = 'Initial Sort'
        heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
            counts=fixedCountsFinal, N=countsTotalFixedMotif, baselineProb=initialRF,
            baselineType=pType, printRF=inShowMotifData, scaleData=True,
            normalizeFixedScores=inNormLetters)

        # Plot: Sequence Motif
        ngs.plotMotif(
            data=heights, dataType='Scaled Enrichment', bigLettersOnTop=inBigLettersOnTop,
            title=inTitleMotif, yMax=yMax,yMin=yMin, showYTicks=inShowWeblogoYTicks,
            addHorizontalLines=inAddHorizontalLines, motifFilter=True,
            duplicateFigure=inDuplicateFigure, saveTag=datasetTag)


    # # Determine: Other Important Residues
    # Initialize variables used for determining the preferred residues
    preferredPositions = [inFixedPosition[0]]
    preferredResidues = [inFixedResidue]

    # # Fix The Next Set Of Substrates
    # Cycle through the substrate and fix AA
    for iteration, position in enumerate(substrateFrameSorted.index):
        if position == initialFixedPos:
            # Skip the position that was already fixed
            continue

        # Update iteration number
        ngs.saveFigureIteration += 1

        # Add the position from this iteration to the list of inspected locations
        if 'R' in position:
            preferredPositions.append(int(position.split('R')[1]))
        else:
            preferredPositions.append(int(position))
            print(f'     Prefered: {preferredPositions}\n\n')

        # for x in range(0, 9):
        # Record preferred residues
        preferredAA = []
        for AA in finalFixedES.index:
            ES = finalFixedES.loc[AA, position]
            if ES >= inSetMinimumESFixAA:
                preferredAA.append(AA)
        preferredResidues.append(preferredAA)

        # Sort preferredPositions and keep preferredResidues in sync
        sortedLists = sorted(zip(preferredPositions, preferredResidues))
        preferredPositions, preferredResidues = zip(*sortedLists)

        # Convert tuples back to lists
        preferredResidues = list(preferredResidues)
        preferredPositions = list(preferredPositions)
        print(f'Preferred Residues:')
        for index in range(len(preferredResidues)):
            AA = ', '.join(preferredResidues[index])
            print(f'     {pink}{AA}{resetColor}@{pink}R{preferredPositions[index]}'
                  f'{resetColor}')
        print('\n')

        # Update NGS attributes
        ngs.fixedAA = preferredResidues
        ngs.fixedPosition = preferredPositions

        # Make fixed seq tag
        fixedSubSeq = ngs.genDatasetTag()

        # Fix Substrates
        fixedSubsFinal, fixedCountsFinal, countsTotalFixedMotif = fixSubstrate(
            subs=substrates, fixedAA=preferredResidues, fixedPosition=preferredPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType='Final Sort',
            fixedTag=fixedSubSeq, initialFix=False)


        # # Process Data
        # Update: Current Sample Size
        ngs.updateSampleSize(NSubs=countsTotalFixedMotif,
                             sortType=f'{purple}Final Sort{resetColor} - {purple}Fixed',
                             printCounts=inPrintSampleSize, fixedTag=fixedSubSeq)

        # Calculate: RF
        finalFixedRF = ngs.calculateRF(counts=fixedCountsFinal, N=countsTotalFixedMotif,
                                       fileType='Fixed Final Sort', printRF=inPrintRF)

        # Visualize: Change in Entropy
        if inPlotPositionalEntropy:
            substrateFrameSorted = ngs.findSubstrateFrame(entropy=positionalEntropy,
                                                          minEntropy=inMinDeltaS,
                                                          fixFullFrame=
                                                          inFixEntireSubstrateFrame,
                                                          getIndices=False)
            if inManualEntropy:
                substrateFrameSorted = pd.DataFrame(1, index=inManualFrame,
                                                    columns=['ΔEntropy'])
                print(f'Ranked Substrate Frame:{green} User Defined{purple}\n'
                      f'{substrateFrameSorted}{resetColor}\n\n')

        # Calculate enrichment scores
        finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                               finalSortRF=finalFixedRF,
                                               printES=inShowEnrichmentData)

        # Inspect enrichment scores
        for index, indexPosition in enumerate(preferredPositions):
            position = f'R' + str(indexPosition)
            for AA in finalFixedES.index:
                ES = finalFixedES.loc[AA, position]
                if ES <= inSetMinimumESFixAA and ES != float('-inf'):
                    if AA in preferredResidues[index]:
                        preferredResidues[index].remove(AA)


        # # Plot the data
        if inPlotEnrichmentMap:
            # Plot: Enrichment Map
            ngs.plotEnrichmentScores(scores=finalFixedES, dataType='Enrichment',
                                     title=inTitleEnrichmentMap, motifFilter=True,
                                     duplicateFigure=inDuplicateFigure,
                                     saveTag=datasetTag)

        if inPlotEnrichmentMotif:
            # Calculate enrichment scores and scale with Shannon Entropy
            pType = 'Initial Sort'
            heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
                counts=fixedCountsFinal, N=countsTotalFixedMotif, baselineProb=initialRF,
                baselineType=pType, printRF=inShowMotifData, scaleData=True,
                normalizeFixedScores=inNormLetters)

            # Plot: Sequence Motif
            ngs.plotMotif(
                data=heights, dataType='Scaled Enrichment',
                bigLettersOnTop=inBigLettersOnTop,  title=inTitleMotif,
                yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
                addHorizontalLines=inAddHorizontalLines, motifFilter=True,
                duplicateFigure=inDuplicateFigure, saveTag=datasetTag)


    # # Release and fix each position
    # Initialize the list of residues that will be fixed
    keepResidues = preferredResidues.copy()
    keepPositions = preferredPositions.copy()
    for iteration, position in enumerate(substrateFrameSorted.index):
        print(f'=================================== Release Residues '
              f'===================================')
        print(f'Save Iteration: {ngs.saveFigureIteration}')

        # Update iteration number
        ngs.saveFigureIteration += 1

        # Determine which residues will be released
        for indexDrop, indexPos in enumerate(preferredPositions):
            if str(indexPos) in position:
                # Drop the element at indexDrop
                keepResidues.pop(indexDrop)
                keepPositions.pop(indexDrop)
                print(f'Dropped Substrate Restriction:\n'
                      f'     Release:{purple} {position}{resetColor}\n\n'
                      f'Fixing Substrates with:')
                for index in range(len(keepResidues)):
                    AA = ', '.join(keepResidues[index])
                    print(f'     {purple}{AA}{resetColor}@{purple}R{keepPositions[index]}'
                          f'{resetColor}')
                print('\n')
                break

        # Update NGS attributes
        ngs.fixedAA = keepResidues
        ngs.fixedPosition = keepPositions


        # # Fix Substrates with released position
        # Make fixed seq tag
        fixedSubSeq = ngs.genDatasetTag()

        # Fix Substrates: Release position
        fixedSubsFinal, fixedCountsFinal, countsTotalFixedMotif = fixSubstrate(
            subs=substrates, fixedAA=keepResidues, fixedPosition=keepPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType='Final Sort',
            fixedTag=fixedSubSeq, initialFix=False)


        # Record counts at released position
        releasedCounts.loc[:, position] = fixedCountsFinal.loc[:, position]

        # # Process Data
        # Update: Current Sample Size
        ngs.updateSampleSize(NSubs=countsTotalFixedMotif,
                             sortType=f'{purple}Final Sort{resetColor} - {purple}Fixed',
                             printCounts=inPrintSampleSize, fixedTag=fixedSubSeq)

        # Calculate: RF
        finalFixedRF = ngs.calculateRF(counts=fixedCountsFinal, N=countsTotalFixedMotif,
                                       fileType='Fixed Final Sort', printRF=inPrintRF)

        # Visualize: Change in Entropy
        if inPlotPositionalEntropy:
            substrateFrameSorted = ngs.findSubstrateFrame(entropy=positionalEntropy,
                                                          minEntropy=inMinDeltaS,
                                                          fixFullFrame=
                                                          inFixEntireSubstrateFrame,
                                                          getIndices=False)
            if inManualEntropy:
                substrateFrameSorted = pd.DataFrame(1, index=inManualFrame,
                                                    columns=['ΔEntropy'])
                print(f'Ranked Substrate Frame:{green} User Defined{purple}\n'
                      f'{substrateFrameSorted}{resetColor}\n\n')

        # Calculate enrichment scores
        finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                               finalSortRF=finalFixedRF,
                                               printES=inShowEnrichmentData)

        # Inspect enrichment scores
        for index, indexPosition in enumerate(keepPositions):
            positionSub = f'R' + str(indexPosition)
            for AA in finalFixedES.index:
                ES = finalFixedES.loc[AA, positionSub]
                if ES <= inSetMinimumESFixAA and ES != float('-inf'):
                    if AA in keepResidues[index]:
                        keepResidues[index].remove(AA)


        # # Plot the data
        if inPlotEnrichmentMap:
            # Plot: Enrichment Map
            ngs.plotEnrichmentScores(scores=finalFixedES, dataType='Enrichment',
                                     title=inTitleEnrichmentMap, motifFilter=True,
                                     duplicateFigure=inDuplicateFigure,
                                     saveTag=datasetTag)

        if inPlotEnrichmentMotif:
            # Calculate enrichment scores and scale with Shannon Entropy
            pType = 'Initial Sort'
            heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
                counts=fixedCountsFinal, N=countsTotalFixedMotif, baselineProb=initialRF,
                baselineType=pType, printRF=inShowMotifData, scaleData=True,
                normalizeFixedScores=inNormLetters)

            # Plot: Sequence Motif
            ngs.plotMotif(
                data=heights, dataType='Scaled Enrichment',
                bigLettersOnTop=inBigLettersOnTop, title=inTitleMotif,
                yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
                addHorizontalLines=inAddHorizontalLines, motifFilter=True,
                duplicateFigure=inDuplicateFigure, saveTag=datasetTag)


        # # Refix Dropped Position
        # Add the dropped position from this iteration to the list of fixed residues
        preferredAA = []
        for AA in finalFixedES.index:
            ES = finalFixedES.loc[AA, position]
            if ES >= inSetMinimumESReleaseAA:
                preferredAA.append(AA)
        keepResidues.insert(indexDrop, preferredAA)
        keepPositions.insert(indexDrop, int(position.split('R')[1]))


        # Make fixed seq tag
        fixedSubSeq = ngs.genDatasetTag()

        # Fix Substrates
        fixedSubsFinal, fixedCountsFinal, countsTotalFixedMotif = fixSubstrate(
            subs=substrates, fixedAA=keepResidues, fixedPosition=keepPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType='Final Sort',
            fixedTag=fixedSubSeq, initialFix=False)


        # # Process Data
        # Update: Current Sample Size
        ngs.updateSampleSize(NSubs=countsTotalFixedMotif,
                             sortType=f'{purple}Final Sort{resetColor} - {purple}Fixed',
                             printCounts=inPrintSampleSize, fixedTag=fixedSubSeq)


        # Here

        # Calculate: RF
        finalFixedRF = ngs.calculateRF(counts=fixedCountsFinal, N=countsTotalFixedMotif,
                                       fileType='Fixed Final Sort', printRF=inPrintRF)

        # Visualize: Change in Entropy
        if inPlotPositionalEntropy:
            substrateFrameSorted = ngs.findSubstrateFrame(entropy=positionalEntropy,
                                                          minEntropy=inMinDeltaS,
                                                          fixFullFrame=
                                                          inFixEntireSubstrateFrame,
                                                          getIndices=False)
            if inManualEntropy:
                substrateFrameSorted = pd.DataFrame(1, index=inManualFrame,
                                                    columns=['ΔEntropy'])
                print(f'Ranked Substrate Frame:{green} User Defined{purple}\n'
                      f'{substrateFrameSorted}{resetColor}\n\n')

        # Calculate enrichment scores
        finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                               finalSortRF=finalFixedRF,
                                               printES=inShowEnrichmentData)

        # Display loop status
        if position == substrateFrameSorted.index[-1]:
            # Update iteration number
            ngs.saveFigureIteration += 1

            print(f'{red}This is the final figure{resetColor}\n\n')
            datasetTag = f'{datasetTag} - Final'


        if inPlotEnrichmentMap:
            # Plot: Enrichment Map
            ngs.plotEnrichmentScores(scores=finalFixedES, dataType='Enrichment',
                                     title=inTitleEnrichmentMap, motifFilter=True,
                                     duplicateFigure=inDuplicateFigure,
                                     saveTag=datasetTag)


        if inPlotEnrichmentMotif:
            # Calculate enrichment scores and scale with Shannon Entropy
            pType = 'Initial Sort'
            heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
                counts=fixedCountsFinal, N=countsTotalFixedMotif,
                baselineProb=initialRF, baselineType=pType, printRF=inShowMotifData,
                scaleData=True, normalizeFixedScores=inNormLetters)

            # Plot: Sequence Motif
            ngs.plotMotif(
                data=heights, dataType='Scaled Enrichment',
                bigLettersOnTop=inBigLettersOnTop, title=inTitleMotif,
                yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
                addHorizontalLines=inAddHorizontalLines, motifFilter=True,
                duplicateFigure=inDuplicateFigure,saveTag=datasetTag)


    # Initialize matrix
    releasedRF = pd.DataFrame(0.0, index=finalFixedES.index,
                              columns=finalFixedES.columns)
    releasedRFScaled = releasedRF.copy()

    # Fill in missing columns in the released counts matrix and calculate RF
    for position in releasedCounts.columns:
        if position not in substrateFrameSorted.index:
            releasedCounts.loc[:, position] = fixedCountsFinal.loc[:, position]

        releasedRF.loc[:, position] = (releasedCounts.loc[:, position] /
                                       countsTotalFixedMotif)
        releasedRFScaled.loc[:, position] = (releasedRF.loc[:, position] *
                                             positionalEntropy.loc[position, 'ΔEntropy'])
    print(f'Scaled RF:{purple} Fixed Frame{resetColor}\n{releasedRFScaled}\n\n')

    # Calculate enrichment scores
    releasedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                         finalSortRF=releasedRF,
                                         printES=inShowEnrichmentData)
    releasedESScaled = releasedES.copy()
    for position in releasedES.columns:
        releasedESScaled.loc[:, position] = (releasedES.loc[:, position] *
                                             positionalEntropy.loc[position, "ΔEntropy"])


    # Save the data
    if inSaveData:
        # Save the fixed substrate dataset
        with open(filePathFixedMotifSubs, 'wb') as file:
            pk.dump(fixedSubsFinal, file)

        # Save the counted substrate data
        fixedCountsFinal.to_csv(filePathFixedMotifCounts,
                                index=True, float_format='%.0f')
        releasedCounts.to_csv(filePathFixedMotifReleasedCounts,
                              index=True, float_format='%.0f')

        # Print the save paths
        print(f'Fixed substrate data saved at:\n'
              f'     {filePathFixedMotifSubs}\n'
              f'     {filePathFixedMotifCounts}\n'
              f'     {filePathFixedMotifReleasedCounts}\n\n')

    return (fixedSubsFinal, fixedCountsFinal, countsTotalFixedMotif,
            releasedRF, releasedRFScaled, releasedCounts)



# ===================================== Run The Code =====================================
# # Fix AA at the important positions in the substrate
fixedSubSeq = ngs.genDatasetTag()

# Define: File path
filePathFixedMotifSubs = os.path.join(
    inFilePath, f'fixedMotif - {inEnzymeName} - {fixedSubSeq} - '
                f'FinalSort - MinCounts {inMinimumSubstrateCount}')
filePathFixedMotifCounts = os.path.join(
    inFilePath, f'fixedMotifCounts - {inEnzymeName} - {fixedSubSeq} - '
                f'FinalSort - MinCounts {inMinimumSubstrateCount}')
filePathFixedMotifReleasedCounts = os.path.join(
    inFilePath, f'fixedMotifCountsRel - {inEnzymeName} - {fixedSubSeq} - '
                f'FinalSort - MinCounts {inMinimumSubstrateCount}')



# Load the fixed frame if the file can be found
if (os.path.exists(filePathFixedMotifSubs) and
        os.path.exists(filePathFixedMotifCounts) and not inRefixMotif):
    print('============================== Load: Fixed Substrate Frame '
          '==============================')
    print(f'File found:{purple} {inDatasetTag}\n'
          f'     {greenDark}{filePathFixedMotifSubs}\n'
          f'     {greenDark}{filePathFixedMotifCounts}{resetColor}\n\n')

    # Disable: Autosave figures
    if inSaveFigures:
        ngs.saveFigures = False
        print(f'{orange}NGS Parameter Update{resetColor}:'
              f'{orange} inSaveFigures was Disabled{resetColor}\n\n')

    # Load Data: Fixed substrates
    with open(filePathFixedMotifSubs, 'rb') as file:
        fixedMotifFinalSubs = pk.load(file)

    # Load Data: Fixed counts
    fixedCountsFinal = pd.read_csv(filePathFixedMotifCounts, index_col=0)

    # Calculate: Total counts
    countsTotalFixedMotif = sum(fixedCountsFinal.iloc[:, 0])

    # Print loaded data
    iteration = 0
    print(f'Loaded Fixed Frame:{purple} {fixedSubSeq}{resetColor}\n' # {substrateFrame}
          f'Number of Total Substrates:'
          f'{white} {countsTotalFixedMotif:,}{resetColor}\n'
          f'Number of Unique Substrates:'
          f'{white} {len(fixedMotifFinalSubs):,}{resetColor}\n')
    for substrate, count in fixedMotifFinalSubs.items():
        print(f'Substrate:{green} {substrate}{resetColor}\n'
              f'     Count:{pink} {count:,}{resetColor}')
        iteration += 1
        if iteration == inPrintNumber:
            break
    print('\n')

    # Update sample size
    ngs.updateSampleSize(NSubs=countsTotalFixedMotif,
                         sortType=f'Final Sort - Fixed Frame:',
                         printCounts=inPrintSampleSize, fixedTag=fixedSubSeq)

    # Calculate: RF
    finalFixedRF = ngs.calculateRF(counts=fixedCountsFinal, N=countsTotalFixedMotif,
                                   fileType='Fixed Final Sort', printRF=inPrintRF)

    # Calculate enrichment scores
    finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                           finalSortRF=finalFixedRF,
                                           printES=inShowEnrichmentData)
    if inPlotEnrichmentMap:
        # Plot: Enrichment Map
        ngs.plotEnrichmentScores(scores=finalFixedES, dataType='Enrichment',
                                 title=inTitleEnrichmentMap, motifFilter=True,
                                 duplicateFigure=inDuplicateFigure,
                                 saveTag=f'Fixed Motif {fixedSubSeq}')

    if inPlotEnrichmentMotif:
        # Calculate enrichment scores and scale with Shannon Entropy
        pType = 'Initial Sort'
        heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
            counts=fixedCountsFinal, N=countsTotalFixedMotif,
            baselineProb=initialRF, baselineType=pType, printRF=inShowMotifData,
            scaleData=True, normalizeFixedScores=inNormLetters)

        # Plot: Sequence Motif
        ngs.plotMotif(
            data=heights, dataType='Scaled Enrichment',
            bigLettersOnTop=inBigLettersOnTop, title=inTitleMotif,
            yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
            addHorizontalLines=inAddHorizontalLines, motifFilter=True,
            duplicateFigure=inDuplicateFigure, saveTag=f'Fixed Motif {fixedSubSeq}')
else:
    # Fix the substrate frame
    (fixedMotifFinalSubs,
     countsFixedMotif,
     countsTotalFixedMotif,
     fixedMotifReleased,
     fixedMotifReleasedRFScaled,
     fixedMotifReleasedCounts) = fixFrame(substrates=substratesFinal,
                                          fixRes=inFixedResidue,
                                          fixPos=inFixedPosition,
                                          sortType='Final Sort',
                                          datasetTag=f'Motif {fixedSubSeq}')
