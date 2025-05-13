from functions import filePaths, NGS
import os
import pandas as pd
import pickle as pk
import sys


# Coordinate Descent & Optimization Framework



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inPathFolder = f'/path/{inEnzymeName}'
inSaveData = False
inSaveFigures = False
inSetFigureTimer = True

# Input 2: Computational Parameters
inMinDeltaS = 0.55
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
inPlotEntropy = True
inPlotEnrichmentMap = True
inPlotEnrichmentMapScaled = False
inPlotLogo = False
inPlotWeblogo = True
inPlotWordCloud = False
inPlotUnscaledScatter = True
inPlotScaledScatter = True

# Input 4: Processing The Data
inPrintNumber = 10

# Input 5: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False

# Input 6: Plot Sequence Motif
inNormLetters = False  # Normalize fixed letter heights
inPlotWeblogoMotif = False
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inPlotNegativeWeblogoMotif = False
inBigLettersOnTop = False

# Input 7: Word Cloud
inLimitWords = True
inNWords = 50

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
pink = '\033[38;2;255;0;242m'
cyan = '\033[38;2;22;255;212m'
green = '\033[38;2;5;232;49m'
greenLight = '\033[38;2;204;255;188m'
greenLightB = '\033[38;2;204;255;188m'
greenDark = '\033[38;2;30;121;13m'
yellow = '\033[38;2;255;217;24m'
orange = '\033[38;2;247;151;31m'
red = '\033[91m'
resetColor = '\033[0m'

# Load: Dataset labels
(enzymeName, inFileNamesInitial,
 inFileNamesFinal, inAAPositions) = filePaths(enzyme=inEnzymeName)



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=enzymeName, substrateLength=len(inAAPositions),
          filterData=True, fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          figEMSquares=inShowEnrichmentAsSquares, xAxisLabels=inAAPositions,
          printNumber=inPrintNumber, showNValues=inShowSampleSize,
          bigAAonTop=inBigLettersOnTop, findMotif=True, folderPath=inPathFolder,
          filesInit=inFileNamesInitial, filesFinal=inFileNamesFinal,
          plotPosS=inPlotEntropy, plotFigEM=inPlotEnrichmentMap,
          plotFigEMScaled=inPlotEnrichmentMapScaled, plotFigLogo=inPlotLogo,
          plotFigWebLogo=inPlotWeblogo, plotFigWords=inPlotWordCloud,
          saveFigures=inSaveFigures, setFigureTimer=inSetFigureTimer)



# ====================================== Load Data =======================================
# Load: Counts
countsInitial, countsInitialTotal = ngs.loadCounts(filter=False, fileType='Initial Sort')



# =================================== Define Functions ===================================
def fixSubstrate(subs, fixedAA, fixedPosition, exclude, excludeAA, excludePosition,
                 sortType, datasetTag):
    print('==================================== Fix Substrates '
          '=====================================')
    print(f'Substrate Dataset:'
          f'{purple} {inEnzymeName} - {sortType}{resetColor}\n')
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
    datasetTag = datasetTag.replace(' ', '')


    # # Load Data
    # Define: File path
    filePathFixedSubs, filePathFixedCounts = (
        ngs.getFilePath(datasetTag=datasetTag))

    # Determine if the fixed substrate file exists
    if os.path.exists(filePathFixedSubs) and os.path.exists(filePathFixedCounts):
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
                print(f'     {pink}{substrate}{resetColor}, '
                      f'Counts: {red}{count:,}{resetColor}')
                iteration += 1
                if iteration == inPrintNumber:
                    break
        print('\n')

        # Count fixed substrates
        fixedCounts, fixedCountsTotal = ngs.countResidues(substrates=fixedSubs,
                                                          datasetType=sortType)

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

    return fixedSubs, fixedCounts, fixedCountsTotal



def fixFrame(substrates, fixRes, fixPos, sortType, datasetTag):
    print('============================ Filter Substrate Motif '
          '=============================')
    print(f'Dataset:{purple} {inEnzymeName} - {sortType} - {datasetTag}{resetColor}\n'
          f'Minimum Substrate Count:{red} {inMinimumSubstrateCount}{resetColor}\n'
          f'Starting with:\n'
          f'     {red}{totalSubsFinal:,}{resetColor} total substrates\n'
          f'     {red}{len(substrates):,}{resetColor} unique substrates\n\n')

    if len(fixRes) != 1:
        print(f'{orange}ERROR:\n'
              f'     You can only fix 1 AA in the first iteration\n'
              f'     You attempted to fix{resetColor}:{greenLightB} {fixRes}{orange}\n\n'
              f'Trim the inFixedResidue list down to 1 AA & try again.\n')
        sys.exit()

    if inCombineFixedMotifs:
        print(f'{orange}ERROR: inCombinedFixedMotifs ='
              f'{greenLightB} {inCombineFixedMotifs}{orange}]n'
              f'Write this code!\n')
        sys.exit()
    else:
        # Make fixed seq tag
        fixedSubSeq = ngs.getDatasetTag()

    # # Fix The First Set Of Substrates
    fixedSubsFinal, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
        subs=substrates, fixedAA=fixRes, fixedPosition=fixPos,
        exclude=inExcludeResidues, excludeAA=inExcludedResidue,
        excludePosition=inExcludedPosition, sortType=sortType,
        datasetTag=fixedSubSeq)

    initialFixedPos = inAAPositions[inFixedPosition[0] - 1]

    # Calculate RF
    finalFixedRF = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                   fileType=sortType)

    # Determine substrate frame
    entropy = ngs.calculateEntropy(probability=finalFixedRF)
    if inManualEntropy:
        motifPos = pd.DataFrame(1, index=inManualFrame, columns=['ΔS'])
        print(f'{pink}Ranked Substrate Frame{resetColor}:{yellow} User Defined\n'
              f'{cyan}{motifPos}{resetColor}\n\n')
    else:
        motifPos = ngs.identifyMotif(entropy=entropy, minEntropy=inMinDeltaS,
                                     fixFullFrame=inFixFullMotifSeq)

    # Display current sample size
    ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalFixedTotal)

    # Calculate enrichment scores
    ngs.calculateEnrichment(probInitial=initialRF, probFinal=finalFixedRF)

    # Create released matrix df
    releasedCounts = pd.DataFrame(0.0, index=ngs.eMap.index,
                                  columns=ngs.eMap.columns)


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
        fixedSubSeq = ngs.getDatasetTag()

        # Fix Substrates
        fixedSubsFinal, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
            subs=substrates, fixedAA=preferredResidues, fixedPosition=preferredPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType=sortType,
            datasetTag=fixedSubSeq)


        # # Process Data
        # Display current sample size
        ngs.recordSampleSize(NInitial=countsInitialTotal, NFinal=countsFinalFixedTotal)

        # Calculate: RF
        finalFixedRF = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                       fileType=f'Fixed Final Sort')

        # Calculate enrichment scores
        ngs.calculateEnrichment(probInitial=initialRF, probFinal=finalFixedRF)

        # Inspect enrichment scores
        for index, indexPosition in enumerate(preferredPositions):
            position = f'R' + str(indexPosition)
            for AA in ngs.eMap.index:
                ES = ngs.eMap.loc[AA, position]
                if ES <= inSetMinimumESFixAA and ES != float('-inf'):
                    if AA in preferredResidues[index]:
                        preferredResidues[index].remove(AA)


    # # Release and fix each position
    # Initialize the list of residues that will be fixed
    keepResidues = preferredResidues.copy()
    keepPositions = preferredPositions.copy()
    for iteration, position in enumerate(motifPos.index):
        print(f'=================================== Release Residues '
              f'===================================')
        print(f'Iteration: {iteration}')

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
        fixedSubSeq = ngs.getDatasetTag()

        # Fix Substrates: Release position
        fixedSubsFinal, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
            subs=substrates, fixedAA=keepResidues, fixedPosition=keepPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType=sortType,
            datasetTag=fixedSubSeq)


        # Record counts at released position
        releasedCounts.loc[:, position] = countsFinalFixed.loc[:, position]

        # # Process Data
        # Update: Current Sample Size
        ngs.sampleSizeUpdate(NSubs=countsFinalFixedTotal,
                             sortType=f'{purple}Final Sort{resetColor} - {purple}Fixed',
                             datasetTag=fixedSubSeq)

        # Calculate: RF
        finalFixedRF = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                       fileType='Fixed Final Sort')

        # Calculate enrichment scores
        ngs.eMap = ngs.enrichmentMatrix(initialRF=initialRF,
                                               finalRF=finalFixedRF)

        # Inspect enrichment scores
        for index, indexPosition in enumerate(keepPositions):
            positionSub = f'R' + str(indexPosition)
            for AA in ngs.eMap.index:
                ES = ngs.eMap.loc[AA, positionSub]
                if ES <= inSetMinimumESFixAA and ES != float('-inf'):
                    if AA in keepResidues[index]:
                        keepResidues[index].remove(AA)


        # # Refix Dropped Position
        # Add the dropped position from this iteration to the list of fixed residues
        preferredAA = []
        for AA in ngs.eMap.index:
            ES = ngs.eMap.loc[AA, position]
            if ES >= inSetMinimumESReleaseAA:
                preferredAA.append(AA)
        keepResidues.insert(indexDrop, preferredAA)
        keepPositions.insert(indexDrop, int(position.split('R')[1]))


        # Make fixed seq tag
        fixedSubSeq = ngs.getDatasetTag()

        # Fix Substrates
        fixedSubsFinal, countsFinalFixed, countsFinalFixedTotal = fixSubstrate(
            subs=substrates, fixedAA=keepResidues, fixedPosition=keepPositions,
            exclude=inExcludeResidues, excludeAA=inExcludedResidue,
            excludePosition=inExcludedPosition, sortType=sortType,
            datasetTag=fixedSubSeq)


        # # Process Data
        # Update: Current Sample Size
        ngs.sampleSizeUpdate(NSubs=countsFinalFixedTotal,
                             sortType=f'{purple}Final Sort{resetColor} - {purple}Fixed',
                             datasetTag=fixedSubSeq)

        # Calculate: RF
        finalFixedRF = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                       fileType='Fixed Final Sort')

        # Calculate enrichment scores
        ngs.calculateEnrichment(probInitial=initialRF, probFinal=finalFixedRF)

        # Display loop status
        if position == motifPos.index[-1]:
            # Update iteration number
            ngs.saveFigureIteration += 1

            print(f'{red}This is the final figure{resetColor}\n\n')
            datasetTag = f'{datasetTag} - Final'


        if inPlotEnrichmentMap:
            # Plot: Enrichment Map
            ngs.plotEnrichmentScores(scores=ngs.eMap, dataType='Enrichment',
                                     title=inFigureTitle, motifFilter=True,
                                     duplicateFigure=inDuplicateFigure,
                                     saveTag=datasetTag)


        if inPlotLogo:
            # Calculate enrichment scores and scale with Shannon Entropy
            pType = 'Initial Sort'
            heights, fixedAA, yMax, yMin = ngs.makeLogo(
                counts=countsFinalFixed, N=countsFinalFixedTotal,
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
    releasedRF = pd.DataFrame(0.0, index=ngs.eMap.index,
                              columns=ngs.eMap.columns)
    releasedRFScaled = releasedRF.copy()

    # Fill in missing columns in the released counts matrix and calculate RF
    for position in releasedCounts.columns:
        if position not in motifPos.index:
            releasedCounts.loc[:, position] = countsFinalFixed.loc[:, position]

        releasedRF.loc[:, position] = (releasedCounts.loc[:, position] /
                                       countsFinalFixedTotal)
        releasedRFScaled.loc[:, position] = (releasedRF.loc[:, position] *
                                             entropy.loc[position, 'ΔS'])
    print(f'Scaled RF:{purple} Fixed Frame{resetColor}\n{releasedRFScaled}\n\n')

    # Calculate enrichment scores
    releasedES = ngs.enrichmentMatrix(initialRF=initialRF, finalRF=releasedRF)
    releasedESScaled = releasedES.copy()
    for position in releasedES.columns:
        releasedESScaled.loc[:, position] = (releasedES.loc[:, position] *
                                             entropy.loc[position, "ΔS"])


    # Save the data
    if inSaveData and not inRefixMotif:
        # Save the fixed substrate dataset
        with open(filePathFixedMotifSubs, 'wb') as file:
            pk.dump(fixedSubsFinal, file)

        # Save the counted substrate data
        countsFinalFixed.to_csv(filePathFixedMotifCounts,
                                index=True, float_format='%.0f')
        releasedCounts.to_csv(filePathFixedMotifReleasedCounts,
                              index=True, float_format='%.0f')

        # Print the save paths
        print(f'Fixed substrate data saved at:{greenDark}\n'
              f'     {filePathFixedMotifSubs}\n'
              f'     {filePathFixedMotifCounts}\n'
              f'     {filePathFixedMotifReleasedCounts}{resetColor}\n\n')

    return (fixedSubsFinal, countsFinalFixed, countsFinalFixedTotal,
            releasedRF, releasedRFScaled, releasedCounts)



# ===================================== Run The Code =====================================
# # Fix AA at the important positions in the substrate
fixedSubSeq = ngs.getDatasetTag()
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
    countsFinalFixed = pd.read_csv(filePathFixedMotifCounts, index_col=0)

    # Calculate: Total counts
    countsFinalFixedTotal = sum(countsFinalFixed.iloc[:, 0])

    # Print loaded data
    iteration = 0
    print(f'Loaded Fixed Frame:{purple} {fixedSubSeq}{resetColor}\n' # {substrateFrame}
          f'Number of Total Substrates:'
          f'{greenLightB} {countsFinalFixedTotal:,}{resetColor}\n'
          f'Number of Unique Substrates:'
          f'{greenLightB} {len(fixedMotifFinalSubs):,}{resetColor}\n')
    for substrate, count in fixedMotifFinalSubs.items():
        print(f'Substrate:{green} {substrate}{resetColor}\n'
              f'     Count:{pink} {count:,}{resetColor}')
        iteration += 1
        if iteration == inPrintNumber:
            break
    print('\n')

    # Update sample size
    ngs.sampleSizeUpdate(NSubs=countsFinalFixedTotal,
                         sortType=f'Final Sort - Fixed Frame:',
                         datasetTag=fixedSubSeq)

    # Calculate: RF
    finalFixedRF = ngs.calculateRF(counts=countsFinalFixed, N=countsFinalFixedTotal,
                                   fileType='Fixed Final Sort')

    # Calculate enrichment scores
    ngs.eMap = ngs.enrichmentMatrix(initialRF=initialRF, finalRF=finalFixedRF)
    if inPlotEnrichmentMap:
        # Plot: Enrichment Map
        ngs.plotEnrichmentScores(scores=ngs.eMap, dataType='Enrichment',
                                 title=inFigureTitle, motifFilter=True,
                                 duplicateFigure=inDuplicateFigure, saveTag=inDatasetTag)

        # Re-enable: Autosave figures
        if inSaveFigures:
            ngs.saveFigures = True
else:
    # Load: Unfiltered substates
    substratesFinal, totalSubsFinal = ngs.loadUnfilteredSubs(loadFinal=True)

    # Fix the substrate frame
    fixFrame(substrates=substratesFinal, fixRes=inFixedResidue, 
             fixPos=inFixedPosition, sortType='Final Sort', datasetTag=inDatasetTag)
