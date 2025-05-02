from functions import filePaths, NGS
import numpy as np
import os
import pickle as pk
import pandas as pd
import threading
import time
import sys



# ADD: Word cloud



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inBasePath = f'/path/{inEnzymeName}'
inFilePath = os.path.join(inBasePath, 'Extracted Data')
inSavePathFigures = '/Users/ca34522/Documents/Classes/Bioinformatics/Project/Figures'
# os.path.join(inBasePath, 'Figures')
inFileNamesInitial, inFileNamesFinal, inAAPositions = filePaths(enzyme=inEnzymeName)
inSaveFigures = True

# Input 2: Processing The Data
inPlotentropy = False
inPlotEnrichmentMap = True
inPlotEnrichmentMotif = True
inPlotWeblogoMotif = False
inPlotWordCloud = False
inPlotAADistribution = False
inPlotPositionalRF = False # For understanding shannon entropy
inPrintLoadedSubs = True
inPrintSampleSize = True
inPrintCounts = True
inPlotCounts = False
inCountsColorMap = ['white','white','lightcoral','red','firebrick','darkred']
inStDevColorMap = ['white','white','#FF76FA','#FF50F9','#FF00F2','#CA00DF','#BD16FF']
inPrintRF = True
inPrintES = True
inPrintEntopy = True
inPrintMotifData = True
inPrintNumber = 10
inCodonSequence = 'NNS' # Base probabilities of degenerate codons (can be N, S, or K)
inUseCodonProb = False # If True: use "inCodonSequence" for baseline probabilities
                       # If False: use "inFileNamesInitial" for baseline probabilities

# Input 3: Computational Parameters
inFilterSubstrates = True
inFixedResidue = ['Q']
inFixedPosition = [4]
inExcludeResidues = False # Do you want to remove any AAs from your collection of substrate
inExcludedResidue = ['A','A']
inExcludedPosition = [9,10]
inMinimumSubstrateCount = 10
inPrintFixedSubs = True
inMinDeltaS = 0.6

# Input 4: Suffix Tree
inNumberOfMotifs = 30
inFigureSizeTrie = (12, 8)

# Input 5: Plot Heatmap
inTitleEnrichmentMap = inEnzymeName
inYLabelEnrichmentMap = 2 # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter
inPrintSelectedSubstrates = 1 # Set = 1, to print substrates with fixed residue
inFigureSize = (9.5, 8) # (width, height)
inFigureBorders = [0.882, 0.075, 0.117, 0.998]

inEnrichmentColorMap = ['navy','royalblue','dodgerblue','lightskyblue','white','white',
                        'lightcoral','red','firebrick','darkred']

# Input 6: Plot Sequence Motif
inNormLetters = True # Normalize fixed letter heights
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inTitleMotif = inTitleEnrichmentMap
inBigLettersOnTop = False
inFigureSizeMotif = inFigureSize
inFigureBordersMotifYTicks = [0.882, 0.075, 0.07, 0.98] # [top, bottom, left, right]
inFigureBordersMotifMaxYTick = [0.882, 0.075, 0.102, 0.98]
inFigureBordersEnrichmentMotif = [0.882, 0.075, 0.138, 0.98]
inLetterColors = ['darkgreen','firebrick','deepskyblue','pink','navy','black','gold']
                  # Aliphatic, Acidic, Basic, Hydroxyl, Amide, Aromatic, Sulfur

# Input 7: Figure Parameters
inFigureTitleSize = 18
inFigureLabelSize = 16
inFigureTickSize = 13
inShowSampleSize = True # Include the sample size in your figures



# =================================== Setup Parameters ===================================
# Figure parameters
figSizeEM = inFigureSize
figBordersEM = inFigureBorders
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

# Colors: Console
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



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=inEnzymeName, substrateLength=len(inAAPositions),
          fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=inExcludeResidues, excludeAA=inExcludedResidue,
          excludePosition=inExcludedPosition, minCounts=inMinimumSubstrateCount,
          xAxisLabels=inAAPositions, xAxisLabelsBinned=None,
          residueLabelType=inYLabelEnrichmentMap, printNumber=inPrintNumber,
          showNValues=inShowSampleSize, findMotif=False, saveFigures=inSaveFigures,
          savePath=inFilePath, savePathFigs=inSavePathFigures, setFigureTimer=None)



# ====================================== Load Data =======================================
startTimeLoadData = time.time()
if inFilterSubstrates:
    fixedSubSeq = ngs.fixSubstrateSequence()
    datasetTag = f'Fixed {fixedSubSeq}'
else:
    fixedSubSeq = None
    datasetTag = None


loadUnfixedSubstrates = True
if inFilterSubstrates:
    filePathFixedCountsFinal = os.path.join(inFilePath,
                                            f'counts_{inEnzymeName}_'
                                            f'FinalSort_{fixedSubSeq}_'
                                            f'MinCounts_{inMinimumSubstrateCount}')
    filePathFixedSubsFinal = os.path.join(inFilePath,
                                          f'fixedSubs_{inEnzymeName}_'
                                          f'FinalSort_{fixedSubSeq}_'
                                          f'MinCounts_{inMinimumSubstrateCount}')


    # Verify that the files exists
    if (os.path.exists(filePathFixedSubsFinal) and
            os.path.exists(filePathFixedCountsFinal)):
        loadUnfixedSubstrates = False

        # # Load the data: Initial sort
        filePathsInitial = []
        for fileName in inFileNamesInitial:
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
            print(
                f'\nMake sure your path is correctly named, and that you have already '
                f'extracted and counted your NGS data\n')
            sys.exit()
        else:
            countsInitial, countsInitialTotal = ngs.loadCounts(
                filePath=filePathsInitial, files=inFileNamesInitial,
                printLoadedData=inPrintCounts, fileType='Initial Sort')
            # Calculate RF
            initialRF = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
                                        fileType='Initial Sort', printRF=inPrintRF)


        # # Load the data: Final sort
        print('================================= Load: Substrate Files '
              '=================================')
        print(f'Loading file at path:\n'
              f'     {greenDark}{filePathFixedSubsFinal}{resetColor}\n\n')
        with open(filePathFixedSubsFinal, 'rb') as file:
            substratesFinal = pk.load(file)
        print(f'Loaded Substrates:{purple} {inEnzymeName} Fixed {fixedSubSeq}')
        iteration = 0
        for substrate, count in substratesFinal.items():
            print(f'     {silver}{substrate}{resetColor}, Count:{red} {count:,}'
                  f'{resetColor}')
            iteration += 1
            if iteration >= 10:
                print('\n')
                break

        # Load: Fixed Counts
        print('================================== Load: Counted Files '
              '==================================')
        print(f'Loading file at path:\n'
              f'     {greenDark}{filePathFixedCountsFinal}{resetColor}\n\n')
        countsFinal = pd.read_csv(filePathFixedCountsFinal, index_col=0)
        countsFinalTotal = sum(countsFinal.iloc[:, 0])
        print(f'Loaded Counts:{purple} {inEnzymeName} Fixed {fixedSubSeq}'
              f'\n{red}{countsFinal}{resetColor}\n\n'
              f'Total substrates:{white} {countsFinalTotal:,}{resetColor}\n\n')
    else:
        print(f'File not found:\n'
              f'     {filePathFixedSubsFinal}'
              f'\n\nLoading substrates and fixing the residue(s):'
              f'{purple} {fixedSubSeq}{resetColor}\n\n')


if loadUnfixedSubstrates:
    def loadSubstrates(filePath, fileNames, fileType, printLoadedData, result):
        subsLoaded, totalSubs = ngs.loadSubstrates(filePath=filePath,
                                                   fileNames=fileNames,
                                                   fileType=fileType,
                                                   printLoadedData=printLoadedData)
        result[fileType] = (subsLoaded, totalSubs)


    # Initialize result dictionary
    loadedResults = {}

    # Create threads for loading initial and final substrates
    threadInitial = threading.Thread(target=loadSubstrates,
                                     args=(inFilePath, inFileNamesInitial,
                                           'Initial Sort', inPrintCounts, loadedResults))
    threadFinal = threading.Thread(target=loadSubstrates,
                                   args=(inFilePath, inFileNamesFinal,
                                         'Final Sort', inPrintCounts, loadedResults))

    # Start the threads
    threadInitial.start()
    threadFinal.start()

    # Wait for the threads to complete
    threadInitial.join()
    threadFinal.join()
    time.sleep(0.5)

    # Retrieve the loaded substrates
    substratesInitial, totalSubsInitial = loadedResults['Initial Sort']
    substratesFinal, totalSubsFinal = loadedResults['Final Sort']

    # Load the data: Initial sort
    filePathsInitial = []
    for fileName in inFileNamesInitial:
        filePathsInitial.append(os.path.join(inFilePath, f'counts_{fileName}'))
    # if '/' in inFilePath:
    #     for fileName in inFileNamesInitial: # _MinCounts_{inMinimumSubstrateCount}
    #         filePathsInitial.append(f'{inFilePath}/counts_{fileName}')
    # else:
    #     for fileName in inFileNamesInitial:
    #         filePathsInitial.append(f'{inFilePath}\\counts_{fileName}')

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
        print(f'\nMake sure your path is correctly named, and that you '
              f'have already extracted and counted your NGS data\n')
        sys.exit()
    else:
        countsInitial, countsInitialTotal = ngs.loadCounts(filePath=filePathsInitial,
                                                           files=inFileNamesInitial,
                                                           printLoadedData=inPrintCounts,
                                                           fileType='Initial Sort')
        # Calculate RF
        initialRF = ngs.calculateRF(counts=countsInitial, N=countsInitialTotal,
                                    fileType='Initial Sort', printRF=inPrintRF)

    # Load the data: final sort
    filePathsFinal = []
    for fileName in inFileNamesFinal:
        filePathsFinal.append(os.path.join(inFilePath, f'counts_{fileName}'))

    # Verify that all files exist
    missingFile = False
    for indexFile, path in enumerate(filePathsFinal):
        print(path)
        if os.path.exists(path):
            continue
        else:
            missingFile = True
            indexMissingFile.append(indexFile)
    if missingFile:
        print('\033[91mERROR: File not found at path:')
        for indexMissing in indexMissingFile:
            print(filePathsFinal[indexMissing])
        print(f'\nMake sure your path is correctly named, and that you '
              f'have already extracted and counted your NGS data\n')
        sys.exit()
    else:
        countsFinal, countsFinalTotal = ngs.loadCounts(filePath=filePathsFinal,
                                                       files=inFileNamesFinal,
                                                       printLoadedData=inPrintCounts,
                                                       fileType='Final Sort')

# Time
endTime = time.time()
runTime = endTime - startTimeLoadData
if inFilterSubstrates:
    print(f'Runtime:{white} Loading {fixedSubSeq} substrates\n'
          f'     {red} {np.round(runTime, 3)}s{resetColor}\n\n')
else:
    print(f'Runtime:{white} Loading unfixed substrates\n'
          f'     {red} {np.round(runTime, 3)}s{resetColor}\n\n')



# ==================================== Evaluate Data =====================================
# Calculate: Average initial RF
RFsum = np.sum(initialRF, axis=1)
initialRFAvg = RFsum / len(initialRF.columns)
initialRFAvg = pd.DataFrame(initialRFAvg, index=initialRFAvg.index,
                            columns=['Average RF'])

if inFilterSubstrates:
    titleSuffixTree = f'{inEnzymeName}: {fixedSubSeq}'
    datasetTag = f'{fixedSubSeq}'

    # Fix AA
    substratesFinal, counFinalTotal = ngs.fixResidue(
        substrates=substratesFinal, fixedString=fixedSubSeq,
        printRankedSubs=inPrintFixedSubs, sortType='Final Sort')

    # Count fixed substrates
    countsFinal, countsFinalTotal = ngs.countResidues(substrates=substratesFinal,
                                       datasetType='Final Sort',
                                       printCounts=inPrintCounts)


# Calculate RF
finalRF = ngs.calculateRF(counts=countsFinal, N=countsFinalTotal,
                          fileType='Final Sort', printRF=inPrintRF)


# Calculate: Positional entropy
entropy = ngs.calculateEntropy(RF=finalRF, printEntropy=inPrintEntopy)
entropySubFrame, indexSubFrame = ngs.findSubstrateFrame(entropy=entropy,
                                                        minEntropy=0.6,
                                                        fixFullFrame=False,
                                                        getIndices=True)

ngs.suffixTree(substrates=substratesFinal, N=inNumberOfMotifs,
               entropySubFrame=entropySubFrame, indexSubFrame=indexSubFrame,
               entropyMin=inMinDeltaS, datasetTag=datasetTag,
               dataType='Final Sort', figSize=inFigureSizeTrie)
