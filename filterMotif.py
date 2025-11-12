from functions import getFileNames, NGS
import os
import pandas as pd
import pickle as pk
import sys


# Coordinate Descent & Optimization Framework



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inPathFolder = f'Enzymes/{inEnzymeName}'
inSaveData = False
inSaveFigures = True
inSetFigureTimer = True

# Input 2: Computational Parameters
inMinDeltaS = 0.6
inRefixMotif = True
inFixedResidue = ['Q']
inFixedPosition = [5] # Fix only at 1 position in the substrate
inExcludeResidues = False
inExcludedResidue = ['A','A']
inExcludedPosition = [9,10]
inManualEntropy = False
inManualFrame = ['R4','R5','R3','R6']
inFixFullMotifSeq = False
inMinimumSubstrateCount = 1
inSetMinimumESFixAA = 0
inSetMinimumESReleaseAA = -1
inPrintFixedSubs = True
inCombineFixedMotifs = False
inPredictSubstrateEnrichmentScores = False
inDuplicateFigure = True
inShowSampleSize = True
inDropResidue = [] # To drop: inDropResidue = ['R9'], For nothing: inDropResidue = []

# Input 3: Figures
inPlotOnlyWords = True
inPlotEntropy = True
inPlotEnrichmentMap = True
inPlotEnrichmentMapScaled = True
inPlotLogo = True
inPlotWordCloud = True
if inPlotOnlyWords:
    #inPlotEntropy = False
    #inPlotEnrichmentMap = False
    inPlotEnrichmentMapScaled = False
    inPlotLogo = False
    inPlotWeblogo = False
    inPlotMotifEnrichment = False
    inPlotWordCloud = True
inPlotWeblogo = False
inPlotUnscaledScatter = False
inPlotScaledScatter = False

# Input 4: Processing The Data
inPrintNumber = 10

# Input 5: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False

# Input 6: Plot Sequence Motif
inBigLettersOnTop = False
inLimitYAxis = False

# Input 8: Substrate Enrichment
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
inMatrixESLabel = r'Enrichment Scores'
inMatrixScaledESLabel = r'ΔS * Enrichment Scores'



# =================================== Setup Parameters ===================================
# Colors:
greyDark = '\033[38;2;144;144;144m'
purple = '\033[38;2;189;22;255m'
magenta = '\033[38;2;255;0;128m'
pink = '\033[38;2;255;0;242m'
cyan = '\033[38;2;22;255;212m'
blue = '\033[38;5;51m'
green = '\033[38;2;5;232;49m'
greenLight = '\033[38;2;204;255;188m'
greenLightB = '\033[38;2;204;255;188m'
greenDark = '\033[38;2;30;121;13m'
yellow = '\033[38;2;255;217;24m'
orange = '\033[38;2;247;151;31m'
red = '\033[91m'
resetColor = '\033[0m'

# Load: Dataset labels
enzymeName, filesInitial, filesFinal, labelAAPos = getFileNames(enzyme=inEnzymeName)



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=enzymeName, substrateLength=len(labelAAPos),
          filterSubs=True, fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          minEntropy=inMinDeltaS, figEMSquares=inShowEnrichmentAsSquares,
          xAxisLabels=labelAAPos, printNumber=inPrintNumber, showNValues=inShowSampleSize,
          bigAAonTop=inBigLettersOnTop, findMotif=True, folderPath=inPathFolder,
          filesInit=filesInitial, filesFinal=filesFinal, plotPosS=inPlotEntropy,
          plotFigEM=inPlotEnrichmentMap, plotFigEMScaled=inPlotEnrichmentMapScaled,
          plotFigLogo=inPlotLogo, plotFigWebLogo=inPlotWeblogo, plotFigWords=False,
          wordLimit=None, wordsTotal=None, plotFigBars=False, NSubBars=None,
          plotFigPCA=False, numPCs=None, NSubsPCA=None, plotSuffixTree=False,
          motifFilter=True, saveFigures=inSaveFigures, setFigureTimer=inSetFigureTimer)



# ====================================== Load Data =======================================
# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort',
                                                   dropColumn=inDropResidue)



# =================================== Define Functions ===================================
def fixSubstrate(subs, fixedAA, fixedPosition, exclude, excludeAA, excludePosition,
                 sortType, posFilter=False):
    print('==================================== Fix Substrates '
          '=====================================')
    print(f'Substrate Dataset:'
          f'{purple} {inEnzymeName} - {sortType}{resetColor}\n')
    print(f'Selecting substrates with:{magenta}')
    for index in range(len(fixedAA)):
        AA = ','.join(fixedAA[index])
        print(f'     {AA}{resetColor}@{magenta}R{fixedPosition[index]}')
    print(f'{resetColor}\n')
    if exclude:
        print(f'Excluding substrates with:{magenta}')
        for index in range(len(excludeAA)):
            AA = ','.join(excludeAA[index])
            print(f'     {AA}@R{excludePosition[index]}')
        print(f'{resetColor}\n')


    # Initialize data structures
    fixedSubs = {}
    fixedSubsTotal = 0


    # # Load Data
    # Define: File path
    filePathFixedSubs, filePathFixedCounts = (
        ngs.getFilePath(datasetTag=ngs.datasetTag))
    print('================================ Filtering Motif '
          '================================')


    # Determine if the fixed substrate file exists
    if os.path.exists(filePathFixedSubs) and os.path.exists(filePathFixedCounts):
        print(f'Loading Substrates at path:\n'
              f'     {greenDark}{filePathFixedSubs}\n'
              f'     {filePathFixedCounts}{resetColor}\n\n')

        # Load Data: Fixed substrates
        with open(filePathFixedSubs, 'rb') as file:
            fixedSubs = pk.load(file)

        # Load Data: Fixed counts
        fixedCounts = pd.read_csv(filePathFixedCounts, index_col=0)

        # Calculate total counts
        fixedCountsTotal = sum(fixedCounts.iloc[:, 0])
    else:
        # Fix the substrates if the files were not found
        print(f'Fixing substrates at {magenta}{posFilter}{resetColor}\n\n')
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
                # Fix AA
                if len(fixedAA) == 1:
                    if isinstance(fixedAA[0], list):
                        for substrate, count in subs.items():
                            if substrate[fixedPosition[0] - 1] in fixedAA[0]:
                                if count >= inMinimumSubstrateCount:
                                    fixedSubs[substrate] = count
                                    fixedSubsTotal += count
                                    continue
                    else:
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
            if fixedSubs:
                for substrate, count in fixedSubs.items():
                    print(f'     {pink}{substrate}{resetColor}, '
                          f'Counts: {red}{count:,}{resetColor}')
                    iteration += 1
                    if iteration == inPrintNumber:
                        break
            else:
                print(
                    f'{orange}ERROR: No substrates we selected.{resetColor}')
        print('\n')

        # Count fixed substrates
        fixedCounts, fixedCountsTotal = ngs.countResidues(substrates=fixedSubs,
                                                          datasetType=sortType)
        # Inspect data for 0 counts
        for position in fixedCounts.columns:
            if (fixedCounts.iloc[:, 0] == 0).all():
                print(f'{orange}ERROR: Zeros counts at residue {cyan}{position}\n')
                sys.exit()


        # Save the fixed substrate dataset
        if inSaveData and not inRefixMotif:
            print('================================= Save The Data '
                  '=================================')
            print(f'Fixed substrate data saved at:\n'
                  f'     {filePathFixedSubs}\n'
                  f'     {filePathFixedCounts}\n\n')

            with open(filePathFixedSubs, 'wb') as file:
                pk.dump(fixedSubs, file)

            # Save the counted substrate data
            fixedCounts.to_csv(filePathFixedCounts, index=True, float_format='%.0f')

    # Remove column(s) from the matrix
    if inDropResidue:
        fixedCounts = ngs.dropColumnsFromMatrix(countMatrix=fixedCounts,
                                                 datasetType=sortType,
                                                 dropColumn=inDropResidue)

    return fixedSubs, fixedCounts, fixedCountsTotal



def fixFrame(substrates, fixRes, fixPos, sortType, datasetTag):
    print('============================ Filter Substrate Motif '
          '=============================')
    print(f'Dataset:{purple} {inEnzymeName} - {sortType} - {datasetTag}{resetColor}\n'
          f'Minimum Substrate Count:{red} {inMinimumSubstrateCount}{resetColor}\n'
          f'Starting with:\n'
          f'     {red}{totalSubsFinal:,}{resetColor} total substrates\n'
          f'     {red}{len(substrates):,}{resetColor} unique substrates\n\n')

    def dispPreferredAA(tag=None):
        if tag is None:
            print(f'Preferred Residues:')
        else:
            print(f'Preferred Residues: {tag}')
        for index in range(len(preferredResidues)):
            AA = ','.join(preferredResidues[index])
            print(f'     {pink}{AA}{resetColor}@{pink}R{preferredPositions[index]}'
                  f'{resetColor}')
        print()
        if tag is not None:
            print()


    if inCombineFixedMotifs:
        print(f'{orange}ERROR: inCombinedFixedMotifs ='
              f'{greenLightB} {inCombineFixedMotifs}{orange}]n'
              f'Write this code!\n')
        sys.exit()
    else:
        # Update: Dataset tag
        ngs.getDatasetTag()


    # # Fix The First Set Of Substrates
    substratesFinalFixed, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
        subs=substrates, fixedAA=fixRes, fixedPosition=fixPos,
        exclude=inExcludeResidues, excludeAA=inExcludedResidue,
        excludePosition=inExcludedPosition, sortType=sortType)

    initialFixedPos = [labelAAPos[pos - 1] for pos in inFixedPosition]

    # Display current sample size
    ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalFixedTotal,
                         NFinalUnique=len(substratesFinalFixed.keys()))

    # Calculate RF
    rfFinalFixed = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                   fileType=sortType)

    # Calculate: Entropy
    ngs.calculateEntropy(rf=rfFinalFixed,
                         fixFullFrame=inFixFullMotifSeq)

    # Overwrite substrate frame
    if inManualEntropy:
        ngs.subFrame = pd.DataFrame(1, index=inManualFrame, columns=['ΔS'])
        print(f'{pink}Ranked Substrate Frame{resetColor}:{yellow} User Defined\n'
              f'{cyan}{ngs.subFrame}{resetColor}\n\n')


    # Calculate enrichment scores
    ngs.calculateEnrichment(rfInitial=rfInitial, rfFinal=rfFinalFixed)

    # Save the data
    if inSaveData:
        ngs.saveData(substrates=substratesFinalFixed, counts=countsFinalFixed)

    # Update: Algorithm parameter
    ngs.motifInit = False

    # # Determine: Other Important Residues
    # Initialize variables used for determining the preferred residues
    preferredPositions = inFixedPosition
    preferredResidues = inFixedResidue

    print(f'Fixing:\n'
          f'     {preferredResidues}\n'
          f'     {preferredPositions}\n')
    print(f'Initial Fix Pos: {initialFixedPos}\n\n')

    # # Fix The Next Set Of Substrates
    # Cycle through the substrate and fix AA
    for iteration, position in enumerate(ngs.subFrame.index):
        print('=============================== Positional Filter '
              '===============================')
        print(f'Iteration: {red}{iteration}{resetColor}\n'
              f' Position: {red}{position}{resetColor}\n\n')
        if position in initialFixedPos:
            # Skip the position that was already fixed
            continue


        # Update: Figure label
        ngs.saveFigureIteration += 1

        # Add the position from this iteration to the list of inspected locations
        if 'R' in position:
            pos = int(position.replace('R', ''))
            preferredPositions.append(pos)
        else:
            preferredPositions.append(int(position))
            print(f'     Prefered: {preferredPositions}\n\n')

        # Record preferred residues
        preferredAA = []
        for AA in ngs.eMap.index:
            ES = ngs.eMap.loc[AA, position]
            if ES >= inSetMinimumESFixAA:
                preferredAA.append(AA)
        preferredResidues.append(preferredAA)



        # Sort preferredPositions and keep preferredResidues in sync
        sortedLists = sorted(zip(preferredPositions, preferredResidues))
        preferredPositions, preferredResidues = zip(*sortedLists)

        # Convert tuples back to lists
        preferredResidues = list(preferredResidues)
        preferredPositions = list(preferredPositions)
        dispPreferredAA()

        # Update NGS attributes
        ngs.fixedAA = preferredResidues
        ngs.fixedPos = preferredPositions

        # Update: Dataset tag
        ngs.getDatasetTag()

        # Fix Substrates
        substratesFinalFixed, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
            subs=substrates, fixedAA=preferredResidues, fixedPosition=preferredPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType=sortType, posFilter=position)


        # # Process Data
        # Display current sample size
        ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalFixedTotal,
                             NFinalUnique=len(substratesFinalFixed.keys()))

        # Calculate: RF
        rfFinalFixed = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                       fileType=f'Fixed Final Sort')

        # Calculate enrichment scores
        ngs.calculateEnrichment(rfInitial=rfInitial, rfFinal=rfFinalFixed,
                                posFilter=position)

        # Save the data
        if inSaveData:
            ngs.saveData(substrates=substratesFinalFixed, counts=countsFinalFixed)

        # Inspect enrichment scores
        print('================================ Inspect Filter '
              '=================================')
        dispPreferredAA()
        for index, indexPosition in enumerate(preferredPositions):
            position = f'R' + str(indexPosition)
            for AA in ngs.eMap.index:
                ES = ngs.eMap.loc[AA, position]
                if ES < inSetMinimumESFixAA and ES != float('-inf'):
                    if AA in preferredResidues[index]:
                        preferredResidues[index].remove(AA)
        dispPreferredAA(tag=f'{greenLight}Filtered{resetColor}')


    print(f'Finish Fixing Iter: {ngs.saveFigureIteration}\n\n')
    # # Release and fix each position
    # Initialize the list of residues that will be fixed
    keepResidues = preferredResidues.copy()
    keepPositions = preferredPositions.copy()
    for iteration, position in enumerate(ngs.subFrame.index):
        print(f'=================================== Release Residues '
              f'===================================')
        # Update: Figure label
        ngs.saveFigureIteration += 1

        # Determine which residues will be released
        indexDrop = None
        for indexDrop, indexPos in enumerate(preferredPositions):
            if str(indexPos) in position:
                # Drop the element at indexDrop
                keepResidues.pop(indexDrop)
                keepPositions.pop(indexDrop)
                print(f'Release Position: {purple}{position}{resetColor}\n'
                      f'Released Filter:')
                for index in range(len(keepResidues)):
                    AA = ','.join(keepResidues[index])
                    print(f'     {magenta}{AA}{resetColor}@'
                          f'{magenta}R{keepPositions[index]}'
                          f'{resetColor}')
                print('\n')
                break

        # Update NGS attributes
        ngs.fixedAA = keepResidues
        ngs.fixedPos = keepPositions


        # # Fix Substrates with released position
        # Update: Dataset tag
        ngs.getDatasetTag()

        # Fix Substrates: Release position
        substratesFinalFixed, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
            subs=substrates, fixedAA=keepResidues, fixedPosition=keepPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType=sortType, posFilter=position)


        # # Process Data
        # Display current sample size
        ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalFixedTotal,
                             NFinalUnique=len(substratesFinalFixed.keys()))

        # Calculate: RF
        rfFinalFixed = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                       fileType='Fixed Final Sort')

        # Calculate enrichment scores
        ngs.calculateEnrichment(rfInitial=rfInitial, rfFinal=rfFinalFixed,
                                posFilter=position, relFilter=True)

        # Save the data
        if inSaveData:
            ngs.saveData(substrates=substratesFinalFixed, counts=countsFinalFixed)

        # Inspect enrichment scores
        print('================================ Inspect Filter '
              '=================================')
        dispPreferredAA()
        for index, indexPosition in enumerate(keepPositions):
            positionSub = f'R' + str(indexPosition)
            for AA in ngs.eMap.index:
                ES = ngs.eMap.loc[AA, positionSub]
                if ES < inSetMinimumESFixAA and ES != float('-inf'):
                    if AA in keepResidues[index]:
                        keepResidues[index].remove(AA)
        dispPreferredAA(tag=f'{greenLight}Filtered{resetColor}')


        # # Refix Dropped Position
        # Add the dropped position from this iteration to the list of fixed residues
        preferredAA = []
        for AA in ngs.eMap.index:
            ES = ngs.eMap.loc[AA, position]
            if ES >= inSetMinimumESReleaseAA:
                preferredAA.append(AA)
        keepResidues.insert(indexDrop, preferredAA)
        keepPositions.insert(indexDrop, int(position.split('R')[1]))

        # Update: Dataset tag
        ngs.getDatasetTag()

        # Fix Substrates
        substratesFinalFixed, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
            subs=substrates, fixedAA=keepResidues, fixedPosition=keepPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType=sortType, posFilter=position)

        # Update: Figure label
        ngs.saveFigureIteration += 1


        # # Process Data
        # Display current sample size
        ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalFixedTotal,
                             NFinalUnique=len(substratesFinalFixed.keys()))

        # Calculate: RF
        rfFinalFixed = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                       fileType='Fixed Final Sort')

        # Calculate enrichment scores
        ngs.calculateEnrichment(rfInitial=rfInitial, rfFinal=rfFinalFixed,
                                posFilter=position)

        # Save the data
        if inSaveData:
            ngs.saveData(substrates=substratesFinalFixed, counts=countsFinalFixed)

        # Display loop status
        if position == ngs.subFrame.index[-1]:
            print(f'{red}This is the final figure{resetColor}\n\n')
            datasetTag = f'{datasetTag} - Final'

    
    # Extract motif
    ngs.getMotif(substrates=substratesFinalFixed)

    # Release the filter
    countsReleased, releasedRF = releaseCounts(substrates=substrates,
                                               countsFiltered=countsFinalFixed,
                                               keepResidues=keepResidues,
                                               keepPositions=keepPositions,
                                               sortType=sortType)

    # Save the data
    if inSaveData:
        ngs.saveData(substrates=substratesFinalFixed, counts=countsFinalFixed,
                     countsReleased=countsReleased)

    return (substratesFinalFixed, countsFinalFixed, countsFinalFixedTotal,
            countsReleased, releasedRF)



def releaseCounts(substrates, countsFiltered, keepResidues, keepPositions, sortType):
    print('================================ Release Counts '
          '=================================')
    print(f'Filter:{purple}')
    for index in range(len(keepResidues)):
        print(f'     {keepResidues[index]}{resetColor}@{purple}R{keepPositions[index]}')
    print(f'{resetColor}\n')

    # Initialize matrices
    countsReleased = pd.DataFrame(0, index=ngs.eMap.index,
                                  columns=ngs.eMap.columns)
    countsTotal = pd.DataFrame(0, index=ngs.eMap.columns, columns=['Total Counts'])
    releasedRF = pd.DataFrame(0.0, index=ngs.eMap.index,
                              columns=ngs.eMap.columns)


    def populateMatrix(counts, popPosition, idxRel):
        # Record counts at released position
        print('======================== Populate Released Count Matrix '
              '=========================')
        print(f'Populate Position: {magenta}{popPosition}{resetColor}\n'
              f'Dataset: {purple}{ngs.datasetTag}{resetColor}\n')
        if isinstance(popPosition, list):
            for pos in popPosition:
                countsReleased.loc[:, pos] = counts.loc[:, pos]
                N = sum(counts.loc[:, pos])
                countsTotal.loc[pos, 'Total Counts'] = N

                # Calculate RF
                releasedRF.loc[:, pos] = (countsReleased.loc[:, pos] /
                                               countsTotal.loc[pos, 'Total Counts'])
        else:
            countsReleased.loc[:, popPosition] = counts.loc[:, popPosition]
            N = sum(counts.loc[:, popPosition])
            countsTotal.loc[popPosition, 'Total Counts'] = N

            # Calculate RF
            releasedRF.loc[:, popPosition] = (
                    countsReleased.loc[:, popPosition] /
                    countsTotal.loc[popPosition, 'Total Counts']
            )

        print(f'Released Counts: {purple}Released counts{resetColor}\n'
              f'{countsReleased}\n\n'
              f'Total Counts: {purple}Released counts{resetColor}'
              f'\n{countsTotal}\n\n'
              f'Relative Frequency: {purple}Released counts{resetColor}\n'
              f'{releasedRF}\n\n')

        # Calculate enrichment scores
        ngs.calculateEnrichment(rfInitial=rfInitial, rfFinal=releasedRF,
                                releasedCounts=True, relIteration=idxRel)


    # Determine which residues will be released
    indexDrop = None
    populatedPositions = []
    for indexRel, posDrop in enumerate(ngs.subFrame.index):
        populatedPositions.append(posDrop)
        position = posDrop
        if 'R' in posDrop:
            posDrop = int(posDrop.replace('R', ''))
        fixAA = keepResidues.copy()
        fixPos = keepPositions.copy()

        # Identify drop
        if posDrop in fixPos:
            indexDrop = fixPos.index(posDrop)
        else:
            print(f'{orange}ERROR: The drop position {cyan}{posDrop}{orange} was not '
                  f'found in the list of fixed positions {cyan}{fixPos}{orange}\n\n')
            sys.exit()
        print(f'Removing Filter: {magenta}{position}{resetColor}\n'
              f'New Filter:')

        # Drop the element at indexDrop
        fixAA.pop(indexDrop)
        fixPos.pop(indexDrop)
        for index in range(len(fixAA)):
            AA = ','.join(fixAA[index])
            print(f'     {magenta}{AA}{resetColor}@{magenta}R{fixPos[index]}'
                  f'{resetColor}')
        print('\n')

        # Update NGS attributes
        ngs.fixedAA = fixAA
        ngs.fixedPos = fixPos

        # # Fix Substrates with released position
        # Update: Dataset tag
        ngs.getDatasetTag()

        # Fix Substrates: Release position
        substratesFinalFixed, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
            subs=substrates, fixedAA=ngs.fixedAA, fixedPosition=ngs.fixedPos,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType=sortType, posFilter=position)

        # Save the data
        if inSaveData:
            ngs.saveData(substrates=substratesFinalFixed, counts=countsFinalFixed)

        # Record counts
        populateMatrix(counts=countsFinalFixed, popPosition=position, idxRel=indexRel)

    # Populate remaining columns
    fillPos = []
    for position in countsReleased.columns:
        if position not in populatedPositions:
            fillPos.append(position)
    populateMatrix(counts=countsFiltered,
                   popPosition=fillPos,
                   idxRel=len(populatedPositions))

    return countsReleased, releasedRF




# ===================================== Run The Code =====================================
# # Fix AA at the important positions in the substrate
fixedSubSeq = ngs.getDatasetTag()
inDatasetTag = f'Motif {fixedSubSeq}'
inFigureTitle = f'{inEnzymeName}: {inDatasetTag}'

# Calculate RF
rfInitial = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
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

    # Dont save figure
    if ngs.saveFigures:
        ngs.saveFigures= False
        print(f'{yellow}Warning: The option that enables saving figures was '
              f'turned off{resetColor}\n\n')

    # Load Data: Fixed substrates
    with open(filePathFixedMotifSubs, 'rb') as file:
        substratesFinalFixed = pk.load(file)

    # Load Data: Fixed counts
    countsFinalFixed = pd.read_csv(filePathFixedMotifCounts, index_col=0)

    # Calculate: Total counts
    countsFinalFixedTotal = sum(countsFinalFixed.iloc[:, 0])

    # Print loaded data
    iteration = 0
    print(f'Loaded Fixed Frame:{purple} {fixedSubSeq}{resetColor}\n' # {substrateFrame}
          f'Number of Total Substrates:'
          f'{greenLightB} {countsFinalFixedTotal:,}{resetColor}\n'
          f'Number of Unique Substrates:'
          f'{greenLightB} {len(substratesFinalFixed):,}{resetColor}\n')

    print(f'Substrates:')
    for substrate, count in substratesFinalFixed.items():
        print(f'     {pink}{substrate}{resetColor}, Count:{red} {count:,}{resetColor}')
        iteration += 1
        if iteration == inPrintNumber:
            break
    print('\n')

    # Display current sample size
    ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalFixedTotal,
                             NFinalUnique=len(substratesFinalFixed.keys()))

    # Calculate: RF
    rfFinalFixed = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                   fileType='Fixed Final Sort')

    # Calculate: Positional entropy
    ngs.calculateEntropy(rf=rfFinalFixed,
                         fixFullFrame=inFixFullMotifSeq)

    # Calculate enrichment scores
    ngs.calculateEnrichment(rfInitial=rfInitial, rfFinal=rfFinalFixed)

    # Extract motif
    finalSubsMotif = ngs.getMotif(substrates=substratesFinalFixed)
else:
    # Load: Unfiltered substates
    substratesFinal, totalSubsFinal = ngs.loadUnfilteredSubs(loadFinal=True)

    # Fix the substrate frame
    fixFrame(substrates=substratesFinal, fixRes=inFixedResidue,
             fixPos=inFixedPosition, sortType='Final Sort', datasetTag=inDatasetTag)
