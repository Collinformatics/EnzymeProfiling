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
inPathFolder = f'/path/{inEnzymeName}'
inSaveData = False
inSaveFigures = False
inSetFigureTimer = True

# Input 2: Computational Parameters
inMinDeltaS = 0.6
inRefixMotif = True
inFixedResidue = ['Q'] # Only use 1 AA
inFixedPosition = [4]
inExcludeResidues = False
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inManualEntropy = False
inManualFrame = ['R4', 'R5', 'R6', 'R2']
inFixFullMotifSeq = False
inMinimumSubstrateCount = 10
inSetMinimumESFixAA = 0
inSetMinimumESReleaseAA = -1
inPrintFixedSubs = True
inCombineFixedMotifs = False
inPredictSubstrateEnrichmentScores = False
inDuplicateFigure = True

# Input 3: Figure Parameters
inShowSampleSize = True
inPlotPositionalEntropy = True
inPlotEnrichmentMap = True
inPlotEnrichmentMotif = False
inPlotUnscaledScatter = True
inPlotScaledScatter = True

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
# If False: use "inFileNamesInitial" for baseline probabilities

# Input 5: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False
inYLabelEnrichmentMap = 2  # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter
inPrintSelectedSubstrates = 1  # Set = 1, to print substrates with fixed residue

# Input 6: Plot Sequence Motif
inNormLetters = False  # Normalize fixed letter heights
inPlotWeblogoMotif = False
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inPlotNegativeWeblogoMotif = False
inBigLettersOnTop = False

# Input 7: Substrate Enrichment
inBinSubstrates = False
inSaveEnrichedSubs = False
inPredictionDatapointColor = '#CC5500'
inSetAxisLimits = False

# Input 8: Predict Optimal substrates
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
inMatrixESLabel = r'Enrichment Scores' # - log5()'
inMatrixScaledESLabel = r'ΔS * Enrichment Scores' # - log5()'



# =================================== Setup Parameters ===================================
inFileNamesInitial, inFileNamesFinal, inAAPositions = (filePaths(enzyme=inEnzymeName))

# Colors:
white = '\033[38;2;255;255;255m'
greyDark = '\033[38;2;144;144;144m'
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
ngs = NGS(enzymeName=inEnzymeName, substrateLength=len(inAAPositions),
          filterData=True, fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          figEMSquares=inShowEnrichmentAsSquares, xAxisLabels=inAAPositions,
          xAxisLabelsBinned=None, residueLabelType=inYLabelEnrichmentMap,
          printNumber=inPrintNumber, showNValues=inShowSampleSize,
          findMotif=True, folderPath=inPathFolder,
          filesInit=inFileNamesInitial, filesFinal=inFileNamesFinal,
          saveFigures=inSaveFigures,
          setFigureTimer=inSetFigureTimer)



# ====================================== Load Data =======================================
# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort')



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
        print(f'     {purple}{AA}{resetColor}@{purple}R{fixedPosition[index]}'
              f'{resetColor}')
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
        inPathFolder, f'fixedSubs - {inEnzymeName} - {fixedTag} - '
                    f'{sortTypePathTag} - MinCounts_{inMinimumSubstrateCount}')
    filePathFixedCounts = os.path.join(
        inPathFolder, f'counts - {inEnzymeName} - {fixedTag} - '
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
                print(f'     Substrate:{greyDark} {substrate}{resetColor}\n'
                      f'         Count:{red} {count:,}{resetColor}')
                iteration += 1
                if iteration == inPrintNumber:
                    break
        print('\n')

        # Count fixed substrates
        fixedCounts, fixedCountsTotal = ngs.countResidues(substrates=fixedSubs,
                                                          datasetType='Final Sort')

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
                                   fileType='Final Sort')

    # Determine substrate frame
    entropy = ngs.calculateEntropy(RF=finalFixedRF, datasetTag=fixedSubSeq)
    if inManualEntropy:
        motifPos = pd.DataFrame(1, index=inManualFrame, columns=['ΔEntropy'])
        print(f'{pink}Ranked Substrate Frame{resetColor}:{yellow} User Defined\n'
              f'{cyan}{motifPos}{resetColor}\n\n')
    else:
        motifPos = ngs.identifyMotif(entropy=entropy, minEntropy=inMinDeltaS,
                                     fixFullFrame=inFixFullMotifSeq, getIndices=False)
    print(f'Motif:\n{motifPos}\n\nExit here')
    sys.exit()

    if inPlotPositionalEntropy:
        # Visualize: Change in Entropy
        ngs.plotPositionalEntropy(entropy=entropy, fixedTag=fixedSubSeq)

    # Display current sample size
    ngs.sampleSizeDisplay(sortType=None, datasetTag=fixedSubSeq)

    # Calculate enrichment scores
    finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                           finalSortRF=finalFixedRF)

    # Create released matrix df
    releasedCounts = pd.DataFrame(0.0, index=finalFixedES.index,
                                  columns=finalFixedES.columns)

    if inPlotEnrichmentMap:
        # Plot: Enrichment Map
        ngs.plotEnrichmentScores(scores=finalFixedES, dataType='Enrichment',
                                 title=inFigureTitle, motifFilter=True,
                                 duplicateFigure=inDuplicateFigure, saveTag=datasetTag)
    # sys.exit()

    if inPlotEnrichmentMotif:
        # Calculate enrichment scores and scale with Shannon Entropy
        pType = 'Initial Sort'
        heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
            counts=fixedCountsFinal, N=countsTotalFixedMotif, baselineProb=initialRF,
            baselineType=pType, scaleData=True, normalizeFixedScores=inNormLetters)

        # Plot: Sequence Motif
        ngs.plotMotif(
            data=heights, dataType='Scaled Enrichment', bigLettersOnTop=inBigLettersOnTop,
            title=inFigureTitle, yMax=yMax,yMin=yMin, showYTicks=inShowWeblogoYTicks,
            addHorizontalLines=inAddHorizontalLines, motifFilter=True,
            duplicateFigure=inDuplicateFigure, saveTag=datasetTag)


    # # Determine: Other Important Residues
    # Initialize variables used for determining the preferred residues
    preferredPositions = [inFixedPosition[0]]
    preferredResidues = [inFixedResidue]

    # # Fix The Next Set Of Substrates
    # Cycle through the substrate and fix AA
    for iteration, position in enumerate(motifPos.index):
        if position == initialFixedPos:
            # Skip the position that was already fixed
            continue


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
                                       fileType='Fixed Final Sort')

        # Calculate enrichment scores
        finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                               finalSortRF=finalFixedRF)

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
                                     title=inFigureTitle, motifFilter=True,
                                     duplicateFigure=inDuplicateFigure,
                                     saveTag=datasetTag)

        if inPlotEnrichmentMotif:
            # Calculate enrichment scores and scale with Shannon Entropy
            pType = 'Initial Sort'
            heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
                counts=fixedCountsFinal, N=countsTotalFixedMotif, baselineProb=initialRF,
                baselineType=pType, scaleData=True, normalizeFixedScores=inNormLetters)

            # Plot: Sequence Motif
            ngs.plotMotif(
                data=heights, dataType='Scaled Enrichment',
                bigLettersOnTop=inBigLettersOnTop,  title=inFigureTitle,
                yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
                addHorizontalLines=inAddHorizontalLines, motifFilter=True,
                duplicateFigure=inDuplicateFigure, saveTag=datasetTag)


    # # Release and fix each position
    # Initialize the list of residues that will be fixed
    keepResidues = preferredResidues.copy()
    keepPositions = preferredPositions.copy()
    for iteration, position in enumerate(motifPos.index):
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
        ngs.sampleSizeUpdate(NSubs=countsTotalFixedMotif,
                             sortType=f'{purple}Final Sort{resetColor} - {purple}Fixed')

        # Calculate: RF
        finalFixedRF = ngs.calculateRF(counts=fixedCountsFinal, N=countsTotalFixedMotif,
                                       fileType='Fixed Final Sort')

        # Calculate enrichment scores
        finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                               finalSortRF=finalFixedRF)

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
                                     title=inFigureTitle, motifFilter=True,
                                     duplicateFigure=inDuplicateFigure,
                                     saveTag=datasetTag)

        if inPlotEnrichmentMotif:
            # Calculate enrichment scores and scale with Shannon Entropy
            pType = 'Initial Sort'
            heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
                counts=fixedCountsFinal, N=countsTotalFixedMotif, baselineProb=initialRF,
                baselineType=pType, scaleData=True, normalizeFixedScores=inNormLetters)

            # Plot: Sequence Motif
            ngs.plotMotif(
                data=heights, dataType='Scaled Enrichment',
                bigLettersOnTop=inBigLettersOnTop, title=inFigureTitle,
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
        ngs.sampleSizeUpdate(NSubs=countsTotalFixedMotif,
                             sortType=f'{purple}Final Sort{resetColor} - {purple}Fixed')

        # Calculate: RF
        finalFixedRF = ngs.calculateRF(counts=fixedCountsFinal, N=countsTotalFixedMotif,
                                       fileType='Fixed Final Sort')

        # Calculate enrichment scores
        finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                               finalSortRF=finalFixedRF)

        # Display loop status
        if position == motifPos.index[-1]:
            # Update iteration number
            ngs.saveFigureIteration += 1

            print(f'{red}This is the final figure{resetColor}\n\n')
            datasetTag = f'{datasetTag} - Final'


        if inPlotEnrichmentMap:
            # Plot: Enrichment Map
            ngs.plotEnrichmentScores(scores=finalFixedES, dataType='Enrichment',
                                     title=inFigureTitle, motifFilter=True,
                                     duplicateFigure=inDuplicateFigure,
                                     saveTag=datasetTag)


        if inPlotEnrichmentMotif:
            # Calculate enrichment scores and scale with Shannon Entropy
            pType = 'Initial Sort'
            heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
                counts=fixedCountsFinal, N=countsTotalFixedMotif,
                baselineProb=initialRF, baselineType=pType, scaleData=True,
                normalizeFixedScores=inNormLetters)

            # Plot: Sequence Motif
            ngs.plotMotif(
                data=heights, dataType='Scaled Enrichment',
                bigLettersOnTop=inBigLettersOnTop, title=inFigureTitle,
                yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
                addHorizontalLines=inAddHorizontalLines, motifFilter=True,
                duplicateFigure=inDuplicateFigure,saveTag=datasetTag)


    # Initialize matrix
    releasedRF = pd.DataFrame(0.0, index=finalFixedES.index,
                              columns=finalFixedES.columns)
    releasedRFScaled = releasedRF.copy()

    # Fill in missing columns in the released counts matrix and calculate RF
    for position in releasedCounts.columns:
        if position not in motifPos.index:
            releasedCounts.loc[:, position] = fixedCountsFinal.loc[:, position]

        releasedRF.loc[:, position] = (releasedCounts.loc[:, position] /
                                       countsTotalFixedMotif)
        releasedRFScaled.loc[:, position] = (releasedRF.loc[:, position] *
                                             entropy.loc[position, 'ΔEntropy'])
    print(f'Scaled RF:{purple} Fixed Frame{resetColor}\n{releasedRFScaled}\n\n')

    # Calculate enrichment scores
    releasedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                         finalSortRF=releasedRF)
    releasedESScaled = releasedES.copy()
    for position in releasedES.columns:
        releasedESScaled.loc[:, position] = (releasedES.loc[:, position] *
                                             entropy.loc[position, "ΔEntropy"])


    # Save the data
    if inSaveData and not inRefixMotif:
        # Save the fixed substrate dataset
        with open(filePathFixedMotifSubs, 'wb') as file:
            pk.dump(fixedSubsFinal, file)

        # Save the counted substrate data
        fixedCountsFinal.to_csv(filePathFixedMotifCounts,
                                index=True, float_format='%.0f')
        releasedCounts.to_csv(filePathFixedMotifReleasedCounts,
                              index=True, float_format='%.0f')

        # Print the save paths
        print(f'Fixed substrate data saved at:{greenDark}\n'
              f'     {filePathFixedMotifSubs}\n'
              f'     {filePathFixedMotifCounts}\n'
              f'     {filePathFixedMotifReleasedCounts}{resetColor}\n\n')

    return (fixedSubsFinal, fixedCountsFinal, countsTotalFixedMotif,
            releasedRF, releasedRFScaled, releasedCounts)



# ===================================== Run The Code =====================================
# # Fix AA at the important positions in the substrate
fixedSubSeq = ngs.genDatasetTag()
inDatasetTag = f'Motif {fixedSubSeq}'
inFigureTitle = f'{inEnzymeName}: {inDatasetTag}'

# Calculate RF
initialRF = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
                            fileType='Initial Sort')

# Define: File paths
(filePathFixedMotifSubs,
 filePathFixedMotifCounts,
 filePathFixedMotifReleasedCounts) = ngs.getFilePath(datasetTag=fixedSubSeq,
                                                     motifPath=True)

# Load the fixed frame if the file can be found
if (os.path.exists(filePathFixedMotifSubs) and
        os.path.exists(filePathFixedMotifCounts) and not inRefixMotif):
    print('============================== Load: Fixed Substrate Frame '
          '==============================')
    print(f'File found:{purple} {inDatasetTag}\n'
          f'     {greenDark}{filePathFixedMotifSubs}\n'
          f'     {greenDark}{filePathFixedMotifCounts}{resetColor}\n\n')

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
                                   fileType='Fixed Final Sort')

    # Calculate enrichment scores
    finalFixedES = ngs.calculateEnrichment(initialSortRF=initialRF,
                                           finalSortRF=finalFixedRF)
    if inPlotEnrichmentMap:
        # Plot: Enrichment Map
        ngs.plotEnrichmentScores(scores=finalFixedES, dataType='Enrichment',
                                 title=inFigureTitle, motifFilter=True,
                                 duplicateFigure=inDuplicateFigure, saveTag=inDatasetTag)

        # Re-enable: Autosave figures
        if inSaveFigures:
            ngs.saveFigures = True

    if inPlotEnrichmentMotif:
        # Calculate enrichment scores and scale with Shannon Entropy
        pType = 'Initial Sort'
        heights, fixedAA, yMax, yMin = ngs.enrichmentMatrix(
            counts=fixedCountsFinal, N=countsTotalFixedMotif, baselineProb=initialRF,
            baselineType=pType, scaleData=True, normalizeFixedScores=inNormLetters)

        # Plot: Sequence Motif
        ngs.plotMotif(
            data=heights, dataType='Scaled Enrichment',
            bigLettersOnTop=inBigLettersOnTop, title=inFigureTitle,
            yMax=yMax, yMin=yMin, showYTicks=inShowWeblogoYTicks,
            addHorizontalLines=inAddHorizontalLines, motifFilter=True,
            duplicateFigure=inDuplicateFigure, saveTag=inDatasetTag)
else:
    # Load: Unfiltered substates
    substratesFinal, totalSubsFinal, = ngs.loadUnfilteredSubs()

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
                                          datasetTag=inDatasetTag)
