# PURPOSE: This script contains the functions that you will need to process your NGS data



from Bio import SeqIO
from Bio.Seq import Seq
from Bio import BiopythonWarning
import esm
import gzip
import joblib
import math
from itertools import combinations, product
import logomaker
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.pyplot import ylabel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import RectangleSelector
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pandas as pd
import pickle as pk
import random
import seaborn as sns
import sys
import threading
import time
import torch
import warnings
from wordcloud import WordCloud



# ================================== Setup Residue List ==================================
defaultResidues = (('Alanine', 'Ala', 'A'), ('Arginine', 'Arg', 'R'),
                   ('Asparagine', 'Asn', 'N'), ('Aspartic Acid', 'Asp', 'D'),
                   ('Cysteine', 'Cys', 'C'),  ('Glutamic Acid', 'Glu', 'E'),
                   ('Glutamine', 'Gln', 'Q'),('Glycine', 'Gly', 'G'),
                   ('Histidine', 'His ', 'H'),('Isoleucine', 'Ile', 'I'),
                   ('Leucine', 'Leu', 'L'), ('Lysine', 'Lys', 'K'),
                   ('Methionine', 'Met', 'M'), ('Phenylalanine', 'Phe', 'F'),
                   ('Proline', 'Pro', 'P'), ('Serine', 'Ser', 'S'),
                   ('Threonine', 'Thr', 'T'),('Tryptophan', 'Typ', 'W'),
                   ('Tyrosine', 'Tyr', 'Y'), ('Valine', 'Val', 'V'))



# ===================================== Set Options ======================================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.5f}'.format)

# Colors: Console
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



# =================================== Define Functions ===================================
def filePaths(enzyme):
    if enzyme.lower() == 'eln' or enzyme.lower() == 'hne':
        enzyme = 'ELN'
        inFileNamesInitialSort = ['ELN-I_S1_L001', 'ELN-I_S1_L002']
        inFileNamesFinalSort = ['ELN-R4_S2_L001', 'ELN-R4_S2_L002']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme.lower() == 'ide':
        enzyme = 'IDE'
        inFileNamesInitialSort = ['IDE-I_S3_L001', 'IDE-I_S3_L002']
        inFileNamesFinalSort = ['IDE-F_S5_L001', 'IDE-F_S5_L002']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme.lower() == 'ide prev':
        enzyme = 'IDE Prev'
        inFileNamesInitialSort = ['IDE-S1_L001', 'IDE-S1_L002',
                                  'IDE-S1_L003', 'IDE-S1_L004']
        inFileNamesFinalSort = ['IDE-S2_L001', 'IDE-S2_L002',
                                'IDE-S2_L003', 'IDE-S2_L004']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme.lower() == 'mpro':
        enzyme = f'SARS-CoV M{'ᵖʳᵒ'}'

        inFileNamesInitialSort = ['Mpro-I_S1_L001', 'Mpro-I_S1_L002',
                                  'Mpro-I_S1_L003', 'Mpro-I_S1_L004']
        inFileNamesFinalSort = ['Mpro-R4_S3_L001', 'Mpro-R4_S3_L002',
                                'Mpro-R4_S3_L003', 'Mpro-R4_S3_L004']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme.lower() == 'mpro2':
        enzyme = f'SARS-CoV-2 M{'ᵖʳᵒ'}'
        inFileNamesInitialSort = ['Mpro2-I_S1_L001', 'Mpro2-I_S1_L002',
                                  'Mpro2-I_S1_L003', 'Mpro2-I_S1_L004']
        inFileNamesFinalSort = ['Mpro2-R4_S3_L001', 'Mpro2-R4_S3_L002',
                                'Mpro2-R4_S3_L003', 'Mpro2-R4_S3_L004']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme.lower() == 'mmp7':
        enzyme = 'MMP7'
        inFileNamesInitialSort = ['MMP7-I_S3_L001', 'MMP7-I_S3_L002']
        inFileNamesFinalSort = ['MMP7-R4_S4_L001', 'MMP7-R4_S4_L002']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme.lower() == 'fyn':
        enzyme = 'Fyn'
        inFileNamesInitialSort = ['Fyn-I_S6_L001', 'Fyn-I_S6_L002']
        inFileNamesFinalSort = ['Fyn-F_S1_L001', 'Fyn-F_S1_L002']
        inAAPositions = ['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']
    elif enzyme.lower() == 'src':
        enzyme = 'Src'
        inFileNamesInitialSort = ['Src-I_S4_L001', 'Src-I_S4_L002']
        inFileNamesFinalSort = ['Src-F_S2_L001', 'Src-F_S2_L002']
        inAAPositions = ['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']
    elif enzyme.lower() == 'veev' or enzyme.lower() == 've':
        enzyme = 'VEEV'
        inFileNamesInitialSort = ['VE-I_S1_L001']
        inFileNamesFinalSort = ['VE-R4_S2_L001']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']
    else:
        print(f'{orange}ERROR: There are no file names for {cyan}{enzyme}{orange}\n'
              f'       Add information to the "filePaths" function in '
              f'{os.path.basename(__file__)}\n')
        sys.exit(1)

    return enzyme, inFileNamesInitialSort, inFileNamesFinalSort, inAAPositions



def pressKey(event):
    if event.key == 'escape':
        plt.close()
    elif event.key == 'e':
        sys.exit()
    elif event.key == 'r':
        python = sys.executable # Doesnt seem to work on windows?
        os.execl(python, python, *sys.argv)



def includeCommas(x):
    return f'{x:,}'



class NGS:
    def __init__(self, enzymeName, substrateLength, filterSubs, fixedAA, fixedPosition,
                 excludeAAs, excludeAA, excludePosition, minCounts, minEntropy,
                 figEMSquares, xAxisLabels, printNumber, showNValues, bigAAonTop,
                 findMotif, folderPath, filesInit, filesFinal, plotPosS, plotFigEM,
                 plotFigEMScaled, plotFigLogo, plotFigWebLogo, plotFigWords, wordLimit,
                 wordsTotal, plotFigBars, NSubBars, plotFigPCA, numPCs, NSubsPCA,
                 plotSuffixTree, saveFigures, setFigureTimer, expressDNA=False,
                 xAxisLabelsMotif=None, motifFilter=False, plotFigMotifEnrich=False,
                 plotFigMotifEnrichSelect=False):
        # Parameters: Dataset
        self.enzymeName = enzymeName
        self.filterSubs = filterSubs
        self.fixedAA = fixedAA
        self.fixedPos = fixedPosition
        self.excludeAAs = excludeAAs
        self.excludeAA = excludeAA
        self.excludePosition = excludePosition
        self.minSubCount = minCounts
        self.minEntropy = minEntropy
        self.xAxisLabels = xAxisLabels
        self.xAxisLabelsMotif = xAxisLabelsMotif
        self.motifLen = None
        self.printNumber = printNumber
        self.selectedSubstrates = []
        self.selectedDatapoints = []
        self.rectangles = []
        self.initialize = True
        
        # Parameters: DNA Processing
        self.expressDNA = expressDNA # Only set as True when processing DNA seqs
        self.minQS = 20 # Minium Phred quality score for extracted substrates
        self.fileSize = []
        self.countExtractedSubs = []
        self.percentUnusableDNASeqs = []

        # Parameters: Figures
        self.entropy, self.entropyMax = None, None
        self.eMap, self.eMapScaled = None, None
        self.eMapReleased, self.eMapReleasedScaled = None, None
        self.heights, self.weblogo = None, None
        self.plotFigEntropy = plotPosS
        self.plotFigEM = plotFigEM
        self.plotFigEMScaled = plotFigEMScaled
        self.plotFigLogo = plotFigLogo
        self.plotFigWebLogo = plotFigWebLogo
        self.plotFigMotifEnrich = plotFigMotifEnrich
        self.plotFigMotifEnrichSelect = plotFigMotifEnrich # plotFigMotifEnrichSelect
        self.plotFigWords = plotFigWords
        self.wordsLimit = wordLimit
        self.wordsTotal = wordsTotal
        self.plotFigBars = plotFigBars
        self.NSubBars = NSubBars
        self.plotFigPCA = plotFigPCA
        self.NPCs=numPCs
        self.NSubsPCA = NSubsPCA
        self.plotSuffixTree = plotSuffixTree
        self.datasetTag = None
        self.datasetTagMotif = None
        self.title = ''
        self.titleCombined = ''
        self.titleReleased = ''
        self.titleWeblogo = ''
        self.titleWeblogoCombined = ''
        self.titleWeblogoReleased = ''
        self.titleWords = ''
        self.titleWordsCombined = ''
        self.substrateLength = substrateLength
        self.figEMSquares = figEMSquares
        if figEMSquares:
            self.figSizeEM = (5, 8) # (width, height)
        else:
            self.figSizeEM = (9.5, 8)
        self.figSize = (9.5, 8)
        self.figSizeMini = (self.figSize[0], 6)
        self.residueLabelType = 2 # 0 = full AA name, 1 = 3-letter code, 2 = 1 letter
        self.labelSizeTitle = 18
        self.labelSizeAxis = 16
        self.labelSizeTicks = 13
        self.lineThickness = 1.5
        self.tickLength = 4
        self.residues = defaultResidues
        self.letters = [residue[2] for residue in self.residues]
        self.colorsAA = NGS.residueColors()
        self.showSampleSize = showNValues
        self.nSubsInitial = 0
        self.nSubsFinal = 0
        self.bigAAonTop = bigAAonTop
        self.findMotif = findMotif
        self.motifFilter = motifFilter
        self.initialize = True # filterMotif.py: Set to False after NGS.calculateEntropy()
        self.motifTag = ''
        self.subFrame = None
        self.motifIndex = None
        self.motifIndexExtracted = []
        self.saveFigures = saveFigures
        self.setFigureTimer = setFigureTimer
        self.figureTimerDuration = 0.5
        self.saveFigureIteration = 0
        self.figureResolution = 300

        # Parameters: Files
        self.filesInit = filesInit
        self.filesFinal = filesFinal
        self.pathFolder = folderPath
        self.pathData = os.path.join(self.pathFolder, 'Data')
        self.pathSaveFigs = os.path.join(self.pathFolder, 'Figures')
        self.pathFilteredSubs = None
        self.pathFilteredCounts = None

        # Parameters: Misc
        self.roundVal = 3
        np.set_printoptions(suppress=True) # Prevent data from printing in sci notation
        np.seterr(divide='ignore')


        # Verify directory paths
        if not os.path.exists(self.pathFolder):
            print(f'{orange}ERROR: Folder not found\n'
                  f'Check input: "{cyan}inPathFolder{orange}"\n'
                  f'     inPathFolder = {self.pathFolder}\n')
            sys.exit(1)
        if self.pathData is not None:
            if not os.path.exists(self.pathData):
                os.makedirs(self.pathData, exist_ok=True)
        if self.pathSaveFigs is not None:
            if not os.path.exists(self.pathSaveFigs):
                os.makedirs(self.pathSaveFigs, exist_ok=True)



    @staticmethod
    def alert(soundPath):
        # This function can be used to play .mp3 files
        # Used it to let you know when a process has been completed
        from playsound import playsound

        if os.path.exists(soundPath):
            threading.Thread(target=playsound, args=(soundPath,)).start()
        else:
            print(f'{orange}ERROR: The alerts sound was not found at\n'
                  f'     {soundPath}{resetColor}\n\n')



    @staticmethod
    def residueColors():
        color = ['darkgreen', 'firebrick', 'deepskyblue', 'pink', 'navy', 'black', 'gold']
                 # Aliphatic, Acidic, Basic, Hydroxyl, Amide, Aromatic, Sulfur

        return {
            'A': color[0],
            'R': color[2],
            'N': color[4],
            'D': color[1],
            'C': color[6],
            'E': color[1],
            'Q': color[4],
            'G': color[0],
            'H': color[2],
            'I': color[0],
            'L': color[0],
            'K': color[2],
            'M': color[6],
            'F': color[5],
            'P': color[0],
            'S': color[3],
            'T': color[3],
            'W': color[5],
            'Y': color[5],
            'V': color[0]
        }



    @staticmethod
    def createCustomColorMap(colorType):
        colorType = colorType.lower()
        if colorType == 'counts':
            useGreen = True
            if useGreen:
                # Green
                colors = ['#FFFFFF','#ABFF9B','#39FF14','#2E9418','#2E9418',
                          '#005000']
            else:
                # Orange
                colors = ['white','white','#FF76FA','#FF50F9','#FF00F2',
                          '#CA00DF','#BD16FF']
        elif colorType == 'stdev':
            colors = ['white','white','#FF76FA','#FF50F9','#FF00F2','#CA00DF','#BD16FF']
        elif colorType == 'word cloud':
            # ,'#F2A900','#2E8B57','black'
            colors = ['#CC5500','#CC5500','#F79620','#FAA338',
                      '#00C01E','#1D680D','#003000','black']
            # colors = ['#008631','#39E75F','#CC5500','#F79620','black']
        elif colorType == 'em':
            colors = ['navy','royalblue','dodgerblue','lightskyblue','white','white',
                      'lightcoral','red','firebrick','darkred']
        else:
            print(f'{orange}ERROR: Cannot create colormap. '
                  f'Unrecognized colorType parameter: {colorType}{resetColor}\n')
            sys.exit(1)

        # Create colormap
        if len(colors) == 1:
            colorList = [(0, colors[0]), (1, colors[0])]
        else:
            colorList = [(i / (len(colors) - 1), color) for i, color in enumerate(colors)]
        return LinearSegmentedColormap.from_list('custom_colormap', colorList)



    def loadAndTranslate(self, filePath, fileName, fileType, fixedSubs,
                         startSeq, endSeq, printQS, forwardRead):
        if forwardRead is None:
            print(f'{orange}ERROR: The file {cyan}{fileName}{orange} does not contain '
                  f'a forward read (R1) or reverse read (R2) label.\n\n'
                  f'Please update the file name, or expand the conditional statements '
                  f'in extractSubs.py that call the function NGS.loadAndTranslate() '
                  f'so that the file can be processed.')
            sys.exit(1)


        # Define file location
        fileLocation = os.path.join(filePath, f'{fileName}.{fileType}')

        # Determine the read direction
        subSequence = None
        # Extract the substrates
        subSequence = self.translate(path=fileLocation, fileName=fileName,
                                     fileType=fileType, fixData=fixedSubs,
                                     startSeq=startSeq, endSeq=endSeq,
                                     printQS=printQS, forwardRead=forwardRead)

        return subSequence



    def translate(self, path, fileName, fileType, fixData, startSeq, endSeq, printQS,
                  forwardRead):
        if forwardRead:
            read = 'R1'
            print('============================ Translate: Forward Read '
                  '============================')
        else:
            read = 'R2'
            print('============================ Translate: Reverse Read '
                  '============================')
        subSequence = {}
        totalSeqsDNA = 0
        printedSeqs = 0


        # Evaluate the file path
        gZipped = False
        if not os.path.isfile(path):
            pathZipped = path + '.gz'
            if os.path.isfile(pathZipped):
                gZipped = True
                path = pathZipped
            else:
                print(f'{orange}ERROR: File location does not lead to a file\n'
                      f'     {path}\n'
                      f'     {pathZipped}\n')
                sys.exit(1)
        print(f'File Location:\n'
              f'     {greenDark}{path}{resetColor}\n\n')

        # Define fixed substrate tag
        if fixData:
            self.getDatasetTag()
            print(f'Evaluating fixed library: {purple}{self.datasetTag}{resetColor}')
        else:
            print(f'Evaluating library: {purple}{self.enzymeName}{resetColor}')


        def printDNA(printedSeqs):
            # Select full DNA seq
            DNA = str(datapoint.seq)
            if not forwardRead:
                DNA = Seq(DNA).reverse_complement()

            # Get: Quality score
            QS = datapoint.letter_annotations['phred_quality']
            if printQS:
                print(f'DNA sequence: {DNA}\n'
                      f'     QS - Forward: {QS}')
            else:
                print(f'DNA sequence: {DNA}')

            # Inspect full DNA seq
            if startSeq in DNA and endSeq in DNA:
                # Find: Substrate indices
                start = DNA.find(startSeq) + len(startSeq)
                end = DNA.find(endSeq)

                # Extract substrate DNA seq
                substrateDNA = DNA[start:end].strip()
                print(f'     Inspected substrate: '
                      f'{greenLightB}{substrateDNA}{resetColor}')
                if len(substrateDNA) == self.substrateLength * 3:

                    # Express substrate
                    substrate = str(Seq.translate(substrateDNA))
                    print(f'     Inspected Substrate:'
                          f'{greenLightB} {substrate}{resetColor}')

                    # Inspect substrate seq: PRINT ONLY
                    if 'X' not in substrate and '*' not in substrate:
                        print(f'     Extracted substrate: '
                              f'{pink}{substrate}{resetColor}')
                        if printQS:
                            QS = QS[start:end]
                            print(f'     QS Substrate: {QS}')
                        printedSeqs += 1

            return printedSeqs


        def inspectDNA():
            # Inspect full DNA seq
            if startSeq in DNA and endSeq in DNA:
                # Find: Substrate indices
                start = DNA.find(startSeq) + len(startSeq)
                end = DNA.find(endSeq)

                # Extract substrate DNA seq
                substrate = DNA[start:end].strip()
                if len(substrate) == self.substrateLength * 3:
                    # Express substrate
                    substrate = str(Seq.translate(substrate))

                    # Inspect substrate seq: Keep good fixed datapoints
                    if 'X' not in substrate and '*' not in substrate:
                        # Inspect quality score
                        QS = datapoint.letter_annotations['phred_quality']
                        QS = QS[start:end]
                        if any(score < self.minQS for score in QS):
                            return

                        # Record datapoint
                        if substrate in subSequence:
                            subSequence[substrate] += 1
                        else:
                            subSequence[substrate] = 1


        def inspectDNAFixed():
            # Inspect full DNA seq
            if startSeq in DNA and endSeq in DNA:
                # Find: Substrate indices
                start = DNA.find(startSeq) + len(startSeq)
                end = DNA.find(endSeq)

                # Extract substrate DNA seq
                substrate = DNA[start:end].strip()
                if len(substrate) == self.substrateLength * 3:
                    # Express substrate
                    substrate = str(Seq.translate(substrate))

                    # Inspect substrate seq: Keep good fixed datapoints
                    if 'X' not in substrate and '*' not in substrate:
                        # Inspect quality score
                        QS = datapoint.letter_annotations['phred_quality']
                        QS = QS[start:end]
                        if any(score < self.minQS for score in QS):
                            return

                        if len(self.fixedAA[0]) == 1:
                            if substrate[self.fixedPos[0] - 1] in self.fixedAA:
                                if substrate in subSequence:
                                    subSequence[substrate] += 1
                                else:
                                    subSequence[substrate] = 1
                        else:
                            if substrate[self.fixedPos[0] - 1] in self.fixedAA[0]:
                                if substrate in subSequence:
                                    subSequence[substrate] += 1
                                else:
                                    subSequence[substrate] = 1


        def evaluateDNAQuality(throwaway, read):
            throwawayPercent = (throwaway / totalSeqsDNA) * 100
            print(f'\nExtraction Efficiency: '
                  f'{greenLight}{fileName}{resetColor}\n'
                  f'     Number of discarded sequences until {red}'
                  f'{self.printNumber} substrates{resetColor} were found in '
                  f'{purple}{read}{resetColor}: {red}{throwaway:,}{resetColor}\n'
                  f'     {yellow}Percent throwaway{resetColor}:'
                  f'{red} {round(throwawayPercent, self.roundVal)} %'
                  f'{resetColor}')


        # Load the data
        if gZipped:
            # Open the file
            with (gzip.open(path, 'rt', encoding='utf-8') as file):
                data = SeqIO.parse(file, fileType)
                warnings.simplefilter('ignore', BiopythonWarning)

                for datapoint in data:
                    printedSeqs = printDNA(printedSeqs)
                    totalSeqsDNA += 1
                    if printedSeqs == self.printNumber:
                        throwaway = totalSeqsDNA - self.printNumber
                        evaluateDNAQuality(throwaway, read)
                        break
                if fixData:
                    print(f'\nNote: The displayed substrates were not filtered for '
                          f'{purple}{self.datasetTag}{resetColor}\n'
                          f'      The filter will be applied to the extracted substrates')
                print('')


                # Extract the substrates
                totalSeqsDNA = 0
                if forwardRead:
                    if fixData:
                        for datapoint in data:
                            # Select full DNA seq
                            DNA = str(datapoint.seq)
                            totalSeqsDNA += 1
                            inspectDNAFixed()
                    else:
                        for datapoint in data:
                            # Select full DNA seq
                            DNA = str(datapoint.seq)
                            totalSeqsDNA += 1
                            inspectDNA()
                else:
                    if fixData:
                        for datapoint in data:
                            # Select full DNA seq
                            DNA = str(datapoint.seq)
                            DNA = Seq(DNA).reverse_complement()
                            totalSeqsDNA += 1
                            inspectDNAFixed()
                    else:
                        for datapoint in data:
                            # Select full DNA seq
                            DNA = str(datapoint.seq)
                            DNA = Seq(DNA).reverse_complement()
                            totalSeqsDNA += 1
                            inspectDNA()
        else:
            # Open the file
            with open(path, 'r') as file:
                data = SeqIO.parse(file, fileType)
                warnings.simplefilter('ignore', BiopythonWarning)

                for datapoint in data:
                    printedSeqs = printDNA(printedSeqs)
                    if printedSeqs == self.printNumber:
                        throwaway = totalSeqsDNA - self.printNumber
                        evaluateDNAQuality(throwaway, read)
                        break
                if fixData:
                    print(f'\nNote: The displayed substrates were not filtered for '
                          f'{purple}{self.datasetTag}{resetColor}\n'
                          f'      The filter will be applied to the extracted substrates')
                print('') # print('\n')

                # Extract the substrates
                totalSeqsDNA = 0
                if forwardRead:
                    if fixData:
                        for datapoint in data:
                            # Select full DNA seq
                            DNA = str(datapoint.seq)
                            totalSeqsDNA += 1
                            inspectDNAFixed()
                    else:
                        for datapoint in data:
                            # Select full DNA seq
                            DNA = str(datapoint.seq)
                            totalSeqsDNA += 1
                            inspectDNA()
                else:
                    if fixData:
                        for datapoint in data:
                            # Select full DNA seq
                            DNA = str(datapoint.seq)
                            DNA = Seq(DNA).reverse_complement()
                            totalSeqsDNA += 1
                            inspectDNAFixed()
                    else:
                        for datapoint in data:
                            # Select full DNA seq
                            DNA = str(datapoint.seq)
                            DNA = Seq(DNA).reverse_complement()
                            totalSeqsDNA += 1
                            inspectDNA()


        # Verify if substrates have been extracted
        if len(subSequence) == 0:
            print(f'\nNo substrates were extracted from file at:\n{path}\n\n'
                  f'Recommend: adjust variables\n'
                  f'     startSeq: {red}{startSeq}{resetColor}\n'
                  f'     endSeq: {red}{endSeq}{resetColor}')
            sys.exit(1)
        else:
            extractionCount = sum(subSequence.values())
            throwaway = (totalSeqsDNA - extractionCount)
            throwawayPercent = (throwaway / totalSeqsDNA) * 100
            self.fileSize.append(totalSeqsDNA)
            self.countExtractedSubs.append(extractionCount)
            self.percentUnusableDNASeqs.append(throwawayPercent)
            print(f'Evaluate All DNA Sequences in {purple}{read}{resetColor}: '
                  f'{greenLight}{fileName}{resetColor}\n'
                  f'     Total DNA sequences in the file: '
                  f'{red}{totalSeqsDNA:,}{resetColor}\n'
                  f'     Number of extracted Substrates: '
                  f'{red}{extractionCount:,}{resetColor}\n'
                  f'     {yellow}Percent throwaway{resetColor}:'
                  f'{red} {round(throwawayPercent, self.roundVal)} %{resetColor}\n\n')

        # Rank the substrates
        subSequence = dict(sorted(subSequence.items(), key=lambda x: x[1], reverse=True))

        return subSequence



    def extractionEfficiency(self, files):
        print('======================== Substrate Extraction Efficiency '
              '========================')
        for index, file in enumerate(files):
            print(f'{pink}Evaluate file{resetColor}:{yellow} {file}{resetColor}\n'
                  f'     Total DNA sequences in the file: '
                  f'{red}{self.fileSize[index]:,}{resetColor}\n'
                  f'     Number of extracted Substrates: '
                  f'{red}{self.countExtractedSubs[index]:,}{resetColor}\n'
                  f'     {yellow}Percent throwaway{resetColor}: '
                  f'{red}{round(self.percentUnusableDNASeqs[index], self.roundVal)} %'
                  f'{resetColor}\n')
        print('')



    def countResidues(self, substrates, datasetType):
        print('============================= Calculate: AA Counts '
              '==============================')
        print(f'Dataset: {purple}{datasetType}{resetColor}\n'
              f'Unique substrate count:{red} {len(substrates):,}{resetColor}')

        # Determine substrate length
        firstSub, lengthSubstrate = None, None
        countMotif = False
        for substrate in substrates.keys():
            firstSub = substrate
            lengthSubstrate = len(substrate)
            if lengthSubstrate != len(self.xAxisLabels):
                countMotif = True
            break

        # Initialize the count matrix
        if countMotif:
            countedData = pd.DataFrame(0,
                                       index=self.letters,
                                       columns=self.xAxisLabelsMotif,
                                       dtype=int)
        else:
            countedData = pd.DataFrame(0,
                                       index=self.letters,
                                       columns=self.xAxisLabels,
                                       dtype=int)

        # Verify consistent substrate lengths
        countModulus = 100 # Check the Nth substrate length
        for index, substrate in enumerate(substrates.keys()):
            if index % countModulus == 0:
                if len(substrate) != lengthSubstrate:
                    print(f'{orange}ERROR: The substrate lengths do not match\n'
                          f'     {firstSub}: {lengthSubstrate} AA\n'
                          f'     {substrate}: {len(substrate)} AA{resetColor}\n')
                    sys.exit(1)

        # Count the occurrences of each residue
        if countMotif:
            # Count the AAs
            for substrate, counts in substrates.items():
                indicesResidue = [self.letters.index(AA) for AA in substrate]
                for position, indexResidue in enumerate(indicesResidue):
                    countedData.iloc[indexResidue, position] += counts

            # Sum all columns
            columnSums = pd.DataFrame(np.sum(countedData, axis=0),
                                      columns=['Total Counts'],
                                      index=self.xAxisLabelsMotif)
        else:
            # Count the AAs
            for substrate, counts in substrates.items():
                indicesResidue = [self.letters.index(AA) for AA in substrate]
                for position, indexResidue in enumerate(indicesResidue):
                    countedData.iloc[indexResidue, position] += counts


            # Sum all columns
            columnSums = pd.DataFrame(np.sum(countedData, axis=0),
                                      columns=['Total Counts'],
                                      index=self.xAxisLabels)

        # Print: Counts
        columnSumsFormated = columnSums.apply(lambda col: col.map(includeCommas)).copy()
        print(f'{columnSumsFormated}\n')
        countedDataPrint = countedData.apply(lambda col: col.map(includeCommas))
        print(f'Counted data: {purple}{self.enzymeName}{resetColor}\n'
              f'{countedDataPrint}\n\n')


        # Sanity Check: Do the sums of each column match the total number of substrates?
        totalSubs = sum(countedData.iloc[:, 0])
        for indexColumn in countedData.columns:
            columnSum = sum(countedData.loc[:, indexColumn])
            if columnSum != totalSubs:
                print(
                    f'Counted data: {purple}{self.enzymeName}{resetColor}\n'
                    f'{countedData}\n\n'
                    f'{orange}ERROR: The total number of substrates '
                    f'({cyan}{totalSubs:,}{orange}) =/= the sum of column '
                    f'{indexColumn} ({cyan}{columnSum:,}{orange}){resetColor}\n')
                sys.exit(1)

        print(f'Total substrates: {red}{totalSubs:,}{resetColor}\n'
              f'Total Unique Substrates:{red} {len(substrates):,}{resetColor}\n\n')

        return countedData, totalSubs



    def getFilePath(self, datasetTag, motifPath=False, customTag=None):
        # Define: File path
        if motifPath:
            if customTag is None:
                pathSubs = os.path.join(
                    self.pathData,
                    f'fixedMotifSubs - {self.enzymeName} - {self.datasetTagMotif} - '
                    f'FinalSort - MinCounts {self.minSubCount}')
                pathCounts = os.path.join(
                    self.pathData,
                    f'fixedMotifCounts - {self.enzymeName} - {self.datasetTagMotif} - '
                    f'FinalSort - MinCounts {self.minSubCount}')
                pathCountsReleased = os.path.join(
                    self.pathData,
                    f'fixedMotifCountsRel - {self.enzymeName} - {self.datasetTagMotif} - '
                    f'FinalSort - MinCounts {self.minSubCount}')
                paths = [pathSubs, pathCounts, pathCountsReleased]
            else:
                pathSubs = os.path.join(
                    self.pathData,
                    f'fixedMotifSubs - {self.enzymeName} - {customTag} - '
                    f'FinalSort - MinCounts {self.minSubCount}')
                pathCounts = os.path.join(
                    self.pathData,
                    f'fixedMotifCounts - {self.enzymeName} - {customTag} - '
                    f'FinalSort - MinCounts {self.minSubCount}')
                pathCountsReleased = os.path.join(
                    self.pathData,
                    f'fixedMotifCountsRel - {self.enzymeName} - {customTag} - '
                    f'FinalSort - MinCounts {self.minSubCount}')
                paths = [pathSubs, pathCounts, pathCountsReleased]
        else:
            pathSubs = os.path.join(
                self.pathData,
                f'fixedSubs - {self.enzymeName} - {datasetTag} - '
                f'FinalSort - MinCounts {self.minSubCount}')
            pathCounts = os.path.join(
                self.pathData,
                f'counts - {self.enzymeName} - {datasetTag} - '
                f'FinalSort - MinCounts {self.minSubCount}')
            self.pathFilteredSubs = pathSubs
            self.pathFilteredCounts = pathCounts
            paths = [pathSubs, pathCounts]

        return paths



    def loadCounts(self, filter, fileType, datasetTag=None):
        print('================================== Load Counts '
              '==================================')
        if filter:
            labelFile = f'{self.enzymeName} {fileType} - Filter {datasetTag}'
        else:
            labelFile = f'{self.enzymeName} {fileType}'
        print(f'Loading Counts: {purple}{labelFile}{resetColor}\n')
        files = []
        totalCounts = 0

        # Define: File paths
        if filter:
            if self.pathFilteredCounts is None:
                print(f'{orange}ERROR: {cyan}self.pathFilteredCounts {orange}needs '
                      f'to be defined before you can load the counts.{resetColor}\n')
                sys.exit(1)
            files = [self.pathFilteredCounts]
        else:
            if 'initial' in fileType.lower():
                fileNames = self.filesInit
            else:
                fileNames = self.filesFinal

            for fileName in fileNames:
                files.append(os.path.join(self.pathData, f'counts_{fileName}'))

        print(f'Loading data:')
        for filePath in files:
            # Verify if the file exists at its specified path
            if not os.path.exists(filePath):
                print(f'{orange}ERROR: File not found\n'
                      f'     {filePath}\n')
                sys.exit(1)
            print(f'     {greenDark}{filePath}{resetColor}')
        print('\n')


        #  Load: AA counts
        countedData = None
        for index, filePath in enumerate(files):
            fileName = filePath.replace(self.pathData, '')

            # Load: File
            if index == 0:
                data = pd.read_csv(filePath, index_col=0)
                data = data.astype(int)
                countedData = data
            else:
                data = pd.read_csv(filePath, index_col=0)
                data = data.astype(int)
                countedData += data

            # Format values to have commas
            formattedCounts = data.to_string(
                formatters={column: '{:,.0f}'.format for column in
                            data.select_dtypes(include='number').columns})

            # Print: Counts
            substrateCounts = sum(data.iloc[:, 0])
            totalCounts += substrateCounts
            print(f'Counts: {greenLightB}{fileName}{resetColor}\n'
                  f'{formattedCounts}\n'
                  f'Substrate Count: {red}'
                  f'{substrateCounts:,}{resetColor}\n\n')
        # Sum each column
        columnSums = pd.DataFrame(np.sum(countedData, axis=0), columns=['Total Counts'])
        columnSumsFormat = columnSums.apply(lambda x: x.map('{:,}'.format))
        print(f'{columnSumsFormat}')

        # Sanity Check: Do the sums of each column match the total number of substrates?
        for indexColumn, columnSum in enumerate(columnSums.iloc[:, 0]):
            if columnSum != totalCounts:
                columnSums = columnSums.apply(lambda x: x.map('{:,}'.format))
                print(f'{orange}ERROR: The total number of substrates '
                      f'({cyan}{totalCounts:,}{orange}) =/= '
                      f'the sum of column {pink}{columnSums.index[indexColumn]}{orange} '
                      f'({cyan}{columnSum:,}{orange})\n')
                sys.exit(1)
        print('\n')

        return countedData, totalCounts



    def loadSubstrates(self, fileNames, fileType):
        print('============================= Load: Substrate Files '
              '=============================')
        substrates = {}
        substrateTotal = 0

        print(f'Loading dataset: {purple}{self.enzymeName} {fileType}{resetColor}')
        for fileName in fileNames:
            print(f'     {greenLightB}{fileName}{resetColor}')
        print()

        # Function to load each file
        def loadFile(fileName):
            fileLocation = os.path.join(self.pathData, f'substrates_{fileName}')
            print(f'File path:\n     {greenDark}{fileLocation}{resetColor}\n')
            with open(fileLocation, 'rb') as openedFile:  # Open file
                data = pk.load(openedFile) # Access the data
                dataTotalSubs = sum(data.values())
                print(f'     Total substrates in {greenLightB}{fileName}{resetColor}: '
                      f'{red}{dataTotalSubs:,}{resetColor}\n')

                # Combine the loaded dictionary into the main substrates
                nonlocal substrates
                for key, value in data.items():
                    if key in substrates:
                        substrates[key] += value
                    else:
                        substrates[key] = value

                nonlocal substrateTotal
                substrateTotal += dataTotalSubs

        threads = []
        for fileName in fileNames:
            thread = threading.Thread(target=loadFile, args=(fileName,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Sort loaded data
        substrates = dict(sorted(substrates.items(), key=lambda x: x[1], reverse=True))

        # Print: Loaded data substrates
        print(f'Loaded data: {purple}{fileType}{resetColor}')
        iteration = 0
        for substrate, count in substrates.items():
            iteration += 1
            print(f'     {pink}{substrate}{resetColor}, Counts: {red}{count:,}'
                  f'{resetColor}')
            if iteration >= self.printNumber:
                break
        print(f'\nTotal substrates: {purple}{fileType}\n'
              f'     {red} {substrateTotal:,}{resetColor}\n\n')

        return substrates, substrateTotal



    def loadUnfilteredSubs(self, loadInitial=False, loadFinal=False):
        def loadSubsThread(fileNames, fileType, result):
            subsLoaded, totalSubs = self.loadSubstrates(fileNames=fileNames,
                                                        fileType=fileType)
            result[fileType] = (subsLoaded, totalSubs)

        # Initialize result dictionary
        loadedResults = {}
        if loadInitial:
            # Create threads for loading initial and final substrates
            threadInitial = threading.Thread(target=loadSubsThread,
                                             args=(self.filesInit, 'Initial Sort',
                                                   loadedResults))

            # Start the threads
            threadInitial.start()

            # Wait for the threads to complete
            threadInitial.join()

            # Retrieve the loaded substrates
            substratesInitial, totalSubsInitial = loadedResults['Initial Sort']

            return substratesInitial, totalSubsInitial
        elif loadFinal:
            # Create thread for the final substrates
            threadFinal = threading.Thread(target=loadSubsThread,
                                           args=(self.filesFinal, 'Final Sort',
                                                 loadedResults))

            # Start the thread
            threadFinal.start()

            # Wait for the thread to complete
            threadFinal.join()

            # Retrieve the loaded substrates
            substrates, totalSubs = loadedResults['Final Sort']

            return substrates, totalSubs

        else:
            # Create threads for loading initial and final substrates
            threadInitial = threading.Thread(target=loadSubsThread,
                                             args=(self.filesInit, 'Initial Sort',
                                                   loadedResults))
            threadFinal = threading.Thread(target=loadSubsThread,
                                           args=(self.filesFinal, 'Final Sort',
                                                 loadedResults))

            # Start the threads
            threadInitial.start()
            threadFinal.start()

            # Wait for the threads to complete
            threadInitial.join()
            threadFinal.join()

            # Retrieve the loaded substrates
            substratesInitial, totalSubsInitial = loadedResults['Initial Sort']
            substratesFinal, totalSubsFinal = loadedResults['Final Sort']

            return substratesInitial, totalSubsInitial, substratesFinal, totalSubsFinal



    def loadSubstratesFiltered(self):
        print('=========================== Load: Filtered Substrate '
              '============================')
        print(f'Loading substrates: {purple}{self.enzymeName} Fixed {self.datasetTag}\n'
              f'     {greenDark}{self.pathFilteredSubs}{resetColor}\n\n')
        with open(self.pathFilteredSubs, 'rb') as file:
            substrates = pk.load(file)

        iteration = 0
        print(f'Substrates:')
        for substrate, count in substrates.items():
            print(f'     {pink}{substrate}{resetColor}, Count:{red} {count:,}'
                  f'{resetColor}')
            iteration += 1
            if iteration >= self.printNumber:
                print('')
                break

        totalSubsFinal = sum(substrates.values())
        print(f'Total substrates: {red}{totalSubsFinal:,}{resetColor}\n\n')

        return substrates, totalSubsFinal



    def loadMotifCounts(self, motifLabel, motifIndex, returnList=False):
        print('================================ Combine Motifs '
              '=================================')
        initialMotifFrame = self.xAxisLabels[motifIndex[0]:motifIndex[1]]
        print('Combine Matrices: Released counts')
        print(f'Datasets: {purple}{self.datasetTag}{resetColor}\n'
              f'Motif Label: {blue}{", ".join(motifLabel)}{resetColor}\n'
              f'Motif Frame: {blue}{", ".join(initialMotifFrame)}{resetColor}\n\n')

        frameLength = len(motifLabel)
        countsMotifsAll = []
        totalCountsFixedFrame = None

        # Load the counts
        for index, position in enumerate(self.fixedPos):
            # Define: File paths
            motifTag = f'{self.fixedAA[0]}@R{position}'

            # Define: File paths
            _, _, pathFixedMotifRelCounts = self.getFilePath(
                datasetTag=self.datasetTag, motifPath=True, customTag=motifTag)

            # Look for the file
            if os.path.exists(pathFixedMotifRelCounts):
                print(f'Loading: {greenLightB}Released Counts\n{greenDark}'
                      f'     {pathFixedMotifRelCounts}{resetColor}\n')

                # Load file
                countsLoaded = pd.read_csv(pathFixedMotifRelCounts, index_col=0)

                # Define motif positions and extract the data
                startPosition = motifIndex[0]
                startSubPrevious = startPosition
                if index != 0:
                    # Evaluate previous motif index
                    fixedPosDifference = (self.fixedPos[index] -
                                          self.fixedPos[index - 1])
                    startSubPrevious += fixedPosDifference
                    startSub = index + startSubPrevious - 1
                    endSub = startSub + frameLength
                else:
                    startSub = startPosition
                    endSub = motifIndex[-1]
                fixedFramePos = countsLoaded.columns[startSub:endSub]
                countsFixedFrame = countsLoaded.loc[:, fixedFramePos]
                countsFixedFrame.columns = motifLabel
                countsMotifsAll.append(countsFixedFrame.values)

                formattedCounts = countsFixedFrame.to_string(
                    formatters={column: '{:,.0f}'.format for column in
                                countsFixedFrame.select_dtypes(include='number').columns})
                print(f'Selecting Positions: {purple}Fixed Motif {motifTag}{resetColor}\n'
                      f'Counts: {blue}{fixedFramePos}{resetColor}\n'
                      f'{formattedCounts}\n\n')

                # Track totals
                if index == 0:
                    totalCountsFixedFrame = countsFixedFrame
                else:
                    totalCountsFixedFrame += countsFixedFrame
            else:
                print(f'{orange}ERROR: The file was not found\n'
                      f'     {pathFixedMotifRelCounts}\n')
                sys.exit(1)

        # Sum the columns
        posSumsCombinedMotif = []
        for column in motifLabel:
            posSumsCombinedMotif.append(sum(totalCountsFixedFrame.loc[:, column]))

        # Format the DataFrame with commas
        formattedCounts = totalCountsFixedFrame.to_string(
            formatters={column: '{:,.0f}'.format for column in
                        totalCountsFixedFrame.select_dtypes(include='number').columns})

        # Print: Counts
        print(f'{pink}Combined Counts{resetColor}: {purple}{self.enzymeName} - '
              f'{self.datasetTag}{resetColor}\n{formattedCounts}\n')
        print('Total Counts:')
        for index, position in enumerate(motifLabel):
            print(f'     {position}: {red}{posSumsCombinedMotif[index]:,}{resetColor}')
        print('\n')

        if returnList:
            # Convert a list into 3D array
            countsMotifsAll = np.stack(countsMotifsAll, axis=0)

            return countsMotifsAll, totalCountsFixedFrame, posSumsCombinedMotif
        else:
            return totalCountsFixedFrame, posSumsCombinedMotif



    def loadMotifSeqs(self, motifLabel, motifIndex):
        print('============================ Load: Substrate Motifs '
              '=============================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n'
              f'Motif Label: {blue}{", ".join(motifLabel)}{resetColor}\n\n')
        frameLength = len(motifLabel)
        totalMotifs = 0
        motifs = {}
        substrates = {}

        # Assign: Motif parameters
        self.motifIndex = motifIndex
        print(f'Motif Index: {self.motifIndex}\n')

        
        # Load the substrates
        for index, position in enumerate(self.fixedPos):
            # Define: File paths
            motifTag = f'{self.fixedAA[0]}@R{position}'

            # Define: File paths
            pathFixedMotifSubs, _, _ = self.getFilePath(
                datasetTag=self.datasetTag, motifPath=True, customTag=motifTag)

            # Look for the file
            if os.path.exists(pathFixedMotifSubs):
                print(f'Loading: {greenLightB}Filtered Motif\n{greenDark}'
                      f'     {pathFixedMotifSubs}{resetColor}\n')

                # Load file
                with open(pathFixedMotifSubs, 'rb') as file:
                    loadedSubs = pk.load(file)
                    if substrates == {}:
                        substrates = loadedSubs
                    else:
                        for substrate, count in loadedSubs.items():
                            if substrate in substrates.keys():
                                substrates[substrate] += count
                            else:
                                substrates[substrate] = count

                # Define motif positions and extract the sequence
                startPosition = motifIndex[0]
                startSubPrevious = startPosition
                if index != 0:
                    # Evaluate previous motif index
                    fixedPosDifference = (self.fixedPos[index] -
                                          self.fixedPos[index - 1])
                    startSubPrevious += fixedPosDifference
                    startSub = index + startSubPrevious - 1
                    endSub = startSub + frameLength
                else:
                    startSub = startPosition
                    endSub = motifIndex[-1]

                # Print: Loaded data
                iteration = 0
                print(f'Loaded Substrates: {purple}{motifTag}{resetColor}')
                print(f'     Motif Indices: {blue}{self.xAxisLabels[startSub]}-'
                      f'{self.xAxisLabels[endSub - 1]}{resetColor}')
                for substrate, count in loadedSubs.items():
                    print(f'          {pink}{substrate}{resetColor}, '
                          f'{blue}{substrate[startSub:endSub]}{resetColor}, '
                          f'Counts: {red}{count:,}'
                          f'{resetColor}')
                    iteration += 1
                    if iteration >= self.printNumber:
                        break

                # Record motifs
                totalCounts = 0
                for substrate, count in loadedSubs.items():
                    totalMotifs += count
                    totalCounts += count
                    motif = substrate[startSub:endSub]
                    if motif in motifs.keys():
                        motifs[motif] += count
                    else:
                        motifs[motif] = count
                print(f'\n     Total Counts: {red}{totalCounts:,}{resetColor}\n\n')

                # seq = 'ATLQG'
                # print(f'Search ({purple}{motifTag}{resetColor}): '
                #       f'{orange}{seq}{resetColor}')
                # for substrate, count in loadedSubs.items():
                #     if seq in substrate:
                #         print(f'     {pink}{substrate}{resetColor}, '
                #               f'Count: {red}{count:,}{resetColor}')
                # print('\n\n')
            else:
                print(f'{orange}ERROR: The file was not found\n'
                      f'     {pathFixedMotifSubs}\n')
                sys.exit(1)

            self.motifIndexExtracted.append((startSub, endSub))
        # Evaluate: Loaded data
        self.motifLen = len(next(iter(motifs)))

        # Sort the substrate dictionary by counts
        motifs = dict(sorted(motifs.items(), key=lambda x: x[1], reverse=True))
        substrates = dict(sorted(substrates.items(), key=lambda x: x[1], reverse=True))

        iteration = 0
        print(f'Top Motifs:')
        for motif, count, in motifs.items():
            print(f'     {blue}{motif}{resetColor}, Counts: {red}{count:,}{resetColor}')
            iteration += 1
            if iteration >= self.printNumber:
                print(f'\nTotal Motifs: {red}{totalMotifs:,}{resetColor}\n'
                      f'Unique Motifs: {red}{len(motifs.keys()):,}'
                      f'{resetColor}\n\n')
                break

        return motifs, totalMotifs, substrates



    def saveData(self, substrates, counts, saveTag=None, countsReleased=None):
        if not isinstance(substrates, dict):
            print(f'{orange}ERROR: The substrates need to be stored in a dictionary\n'
                  f'     Current data structure: {type(substrates)}')
            sys.exit(1)
        if not isinstance(counts, pd.DataFrame):
            print(f'{orange}ERROR: The counts need to be stored in a Pandas DataFrame\n'
                  f'     Current data structure: {type(counts)}')
            sys.exit(1)

        # Define: Save path
        filePathCountsReleased = None
        if self.expressDNA:
            filePathSubs = os.path.join(self.pathData, f'substrates_{saveTag}')
            filePathCounts = os.path.join(self.pathData, f'counts_{saveTag}')
        else:
            if countsReleased is None:
                (filePathSubs,
                 filePathCounts) = self.getFilePath(datasetTag=self.datasetTag)
            else:
                # Define: File paths
                (filePathSubs,
                 filePathCounts,
                 filePathCountsReleased) = self.getFilePath(datasetTag=self.datasetTag,
                                                            motifPath=True)

        if not os.path.exists(filePathSubs) or not os.path.exists(filePathCounts):
            print('================================= Save The data '
                  '=================================')
            if countsReleased is None:
                print(f'Substrate data saved at:\n'
                      f'     {greenDark}{filePathSubs}\n'
                      f'     {filePathCounts}{resetColor}\n\n')

                # Save the substrates
                with open(filePathSubs, 'wb') as file:
                    pk.dump(substrates, file)

                # Save the counts
                counts.to_csv(filePathCounts)
            else:
                print(f'Substrate data saved at:\n'
                      f'     {greenDark}{filePathSubs}\n'
                      f'     {filePathCounts}\n'
                      f'     {filePathCountsReleased}{resetColor}\n\n')

                # Save the substrates
                with open(filePathSubs, 'wb') as file:
                    pk.dump(substrates, file)

                # Save the counts
                counts.to_csv(filePathCounts)
                countsReleased.to_csv(filePathCountsReleased)



    def saveFigure(self, fig, figType, seqLen, combinedMotifs=False, releasedCounts=False):
        # Define: Save location
        figLabel = ''
        if self.motifFilter and not releasedCounts:
            figLabel = (f'{self.enzymeName} - {figType} '
                        f'{self.saveFigureIteration} - {self.datasetTagMotif} - '
                        f'{seqLen} AA - MinCounts {self.minSubCount}.png')
        elif combinedMotifs and releasedCounts:
            figLabel = (f'{self.enzymeName} - {figType} - '
                        f'Combined Released Counts {self.datasetTag} - '
                        f'{seqLen} AA - MinCounts {self.minSubCount}.png')
        elif combinedMotifs:
                figLabel = (f'{self.enzymeName} - {figType} - '
                            f'Combined {self.datasetTag} - {seqLen} AA - '
                            f'N {self.nSubsFinal} - MinCounts {self.minSubCount}.png')
        elif releasedCounts:
            figLabel = (f'{self.enzymeName} - {figType} Released Counts - '
                        f'{self.datasetTagMotif} - {seqLen} AA - '
                        f'MinCounts {self.minSubCount}.png')
        else:
            figLabel = (f'{self.enzymeName} - {figType} - '
                        f'{self.datasetTag} - '
                        f'N {self.nSubsFinal} - {seqLen} AA - '
                        f'MinCounts {self.minSubCount}.png')
        if combinedMotifs and len(self.motifIndexExtracted) < 2:
            figLabel = figLabel.replace('Combined ', '')
        saveLocation = os.path.join(self.pathSaveFigs, figLabel)


        # Save figure
        if os.path.exists(saveLocation):
            print(f'{yellow}WARNING{resetColor}: '
                  f'{yellow}The figure already exists at the path\n'
                  f'     {saveLocation}\n\n'
                  f'We will not overwrite the figure{resetColor}\n\n')
        else:
            print(f'Saving figure at path:\n'
                  f'     {greenDark}{saveLocation}{resetColor}\n\n')
            fig.savefig(saveLocation, dpi=self.figureResolution)



    def recordSampleSize(self, NInitial, NFinal):
        print('============================== Current Sample Size '
              '==============================')
        if isinstance(NInitial, int) and isinstance(NFinal, int):
            # Update: Sample size
            self.nSubsInitial = NInitial
            self.nSubsFinal = NFinal
            print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n'
                  f'Initial Sort: {red}{self.nSubsInitial:,}{resetColor}\n'
                  f'Final Sort: {red}{self.nSubsFinal:,}{resetColor}\n\n')
        else:
            print(f'Sample size was not recorded.\n'
                  f'N values must be integers:\n'
                  f'     N Initial: {pink}{type(NInitial)}{resetColor}\n'
                  f'     N Final: {pink}{type(NFinal)}{resetColor}\n\n')

        # Set figure titles
        self.titleReleased = (f'{self.enzymeName}\n'
                              f'Combined {self.datasetTagMotif}\n'
                              f'Released Counts')
        if self.showSampleSize:
            self.title = (f'{self.enzymeName}\n'
                          f'N Unsorted = {self.nSubsInitial:,}\n'
                          f'N Sorted = {self.nSubsFinal:,}')
            self.titleCombined = (f'{self.enzymeName}\n'
                                  f'Combined {self.datasetTagMotif}\n'
                                  f'N Unsorted = {self.nSubsInitial:,}\n'
                                  f'N Sorted = {self.nSubsFinal:,}')
            self.titleWeblogo = f'{self.enzymeName}\nN = {self.nSubsFinal:,}'
            self.titleWeblogoCombined = (f'{self.enzymeName}\n'
                                         f'Combined {self.datasetTagMotif}\n'
                                         f'N = {self.nSubsFinal:,}')
        else:
            self.title = f'{self.enzymeName}'
            self.titleWeblogo = f'{self.enzymeName}'
            self.titleWeblogoCombined = (f'{self.enzymeName}\n'
                                         f'Combined {self.datasetTagMotif}')
        if self.filterSubs:
            if self.motifFilter:
                self.titleWords = f'{self.enzymeName}\nMotif {self.motifTag}'
            else:
                self.titleWords = f'{self.enzymeName}\n{self.datasetTagMotif}'
                self.titleWordsCombined = (f'{self.enzymeName}\n'
                                           f'Combined {self.datasetTagMotif}')
        else:
            self.titleWords = f'{self.enzymeName}\nUnfiltered'
        if len(self.motifIndexExtracted) > 1:
            self.titleReleased = self.titleReleased.replace('Combined ', '')
            self.titleCombined = self.titleCombined.replace('Combined ', '')
            self.titleWeblogoCombined =  self.titleWeblogoCombined.replace('Combined ', '')
            self.titleWordsCombined = self.titleWordsCombined.replace('Combined ', '')


    def calculateProbabilitiesCM(self, countsCombinedMotifs):
        print('====================== Calculate: Fixed Motif Probability '
              '=======================')
        # Sum each column
        columnSums = np.sum(countsCombinedMotifs, axis=0)
        columnSums = pd.DataFrame(columnSums, index=countsCombinedMotifs.columns,
                                  columns=['Sum'])

        prob = pd.DataFrame(0.0, index=countsCombinedMotifs.index,
                            columns=countsCombinedMotifs.columns)

        # Calculate: Residue Probability
        for position in countsCombinedMotifs.columns:
            prob.loc[:, position] = (countsCombinedMotifs.loc[:, position] /
                                   columnSums.loc[position, 'Sum'])
        pd.set_option('display.float_format', '{:,.5f}'.format)
        print(f'Dataset: {purple}{self.datasetTag}\n'
              f'{green}{prob}{resetColor}\n\n')
        pd.set_option('display.float_format', '{:,.3f}'.format)

        return prob



    def calculateProbCodon(self, codonSeq):
        print('======================= Calculate: Residue Probabilities '
              '========================')
        print(f'Possible codons for {codonSeq}:')
        nucleotides = ['A', 'C', 'G', 'T']
        S = ['C', 'G']
        K = ['G', 'T']

        # Define what codons are associated with each residue
        codonsAA = {
            'A': ['GCT', 'GCC', 'GCA', 'GCG'],
            'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
            'N': ['AAT', 'AAC'],
            'D': ['GAT', 'GAC'],
            'C': ['TGT', 'TGC'],
            'E': ['GAA', 'GAG'],
            'Q': ['CAA', 'CAG'],
            'G': ['GGT', 'GGC', 'GGA', 'GGG'],
            'H': ['CAT', 'CAC'],
            'I': ['ATT', 'ATC', 'ATA'],
            'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
            'K': ['AAA', 'AAG'],
            'M': ['ATG'],
            'F': ['TTT', 'TTC'],
            'P': ['CCT', 'CCC', 'CCA', 'CCG'],
            'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
            'T': ['ACT', 'ACC', 'ACA', 'ACG'],
            'W': ['TGG'],
            'Y': ['TAT', 'TAC'],
            'V': ['GTT', 'GTC', 'GTA', 'GTG']
        }

        # Initialize a list to store all possible combinations
        codons = []

        # Generate all possible combinations
        for combination in product(nucleotides, repeat=len(codonSeq)):
            # Check if the combination satisfies the conditions
            if all((c == 'N') or (c == 'S' and s in S) or (c == 'K' and s in K)
                   for c, s in zip(codonSeq, combination)):
                codons.append(''.join(combination))

            # Print: All possible codon combinations
            for index, codon in enumerate(codons, 1):
                print(f'Codon {index}: {codon}')
            print('')

        # Count the possible codon combinations for each AA
        codonCounts = pd.DataFrame(index=self.letters, columns=['Counts'], data=0)
        for sequence in codons:
            for residue, codonsResidue in codonsAA.items():
                if sequence in codonsResidue:
                    if residue in codonCounts.index:
                        codonCounts.loc[residue, 'Counts'] += 1
                    break
        codonProbability = pd.DataFrame(index=self.letters, columns=['Probability'], data=0)
        codonProbability['Probability'] = codonCounts['Counts'] / len(codons)

        print('Amino Acid Probabilities:')
        for index, row in codonProbability.iterrows():
            print(f'     {index}    {round(row["Probability"] * 100, 2)} %')
        codonProb = round(sum(codonProbability["Probability"]) * 100, 2)
        print(f'Total probability of AA with {codonSeq}: {codonProb} %')
        print(f'Stop codon probability: {round(100 - codonProb, 2)} %\n\n')

        return codonProbability



    def calculateProbabilities(self, counts, N, fileType, calcAvg=False):
        print('======================== Calculate: Residue Probability '
              '=========================')
        if self.filterSubs and 'initial' not in fileType.lower():
            print(f'Dataset: {purple}{self.enzymeName} {fileType} - Filter '
                  f'{self.datasetTag}{resetColor}\n')
        else:
            print(f'Dataset: {purple}{self.enzymeName} - {fileType}{resetColor}\n')

        # Calculate: Probability
        prob = counts / N
        print(f'{np.round(prob, 4)}\n\n')

        if calcAvg:
            probAvg = np.sum(prob, axis=1) / len(prob.columns)
            prob = pd.DataFrame(probAvg, index=probAvg.index, columns=['Average RF'])
            print(f'{np.round(prob, 4)}\n\n')

        return prob



    def compairRF(self, probInitial, probFinal, selectAA):
        print('======================= Evaluate Specificity: Compair RF '
              '========================')
        if selectAA in self.letters:
            residue = self.residues[self.letters.index(selectAA)][0]
        else:
            print(f'{greyDark}Residue not recognized:{red} {selectAA}{greyDark}\n'
                  f'Please check input:{red} self.fixedAA')
            sys.exit(1)
        print(f'Fixed Residues:{red} {self.datasetTag}{resetColor}\n'
              f'Selected Residue:{red} {residue}{resetColor}\n')

        initial = probInitial[probInitial.index.str.contains(selectAA)]
        final = probFinal[probFinal.index.str.contains(selectAA)]

        print(f'{purple}Initial Sort{resetColor}:\n{initial}\n'
              f'{purple}Final Sort{resetColor}:\n{final}\n\n')

        # Figure parameters
        barWidth = 0.4

        # Determine yMax
        yMin = 0
        yMax = 1

        # Set the positions of the bars on the x-axis
        x = np.arange(len(final.columns))

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=self.figSizeMini)

        # Plotting the bars
        ax.bar(x - barWidth / 2, initial.iloc[0], width=barWidth, label='Initial Sort',
               color='#000000')
        ax.bar(x + barWidth / 2, final.iloc[0], width=barWidth, label='Final Sort',
               color='#BF5700')

        # Adding labels and title
        ax.set_ylabel('Relative Frequency', fontsize=self.labelSizeAxis)
        ax.set_title(f'{residue} RF: {self.enzymeName} Fixed {self.datasetTag}',
                     fontsize=self.labelSizeTitle, fontweight='bold')
        ax.legend()
        plt.subplots_adjust(top=0.898, bottom=0.098, left=0.112, right=0.917)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks)

        # Set x ticks
        ax.set_xticks(x)
        ax.set_xticklabels(self.xAxisLabels)

        ax.set_ylim(yMin, yMax)

        # Set the edge thickness
        for spine in ax.spines.values():
            spine.set_linewidth(self.lineThickness)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        plt.show()



    def boxPlotRF(self, probInitial, probFinal, selectAA):
        print('=============================== Plot: RF Box Plot '
              '===============================')
        if selectAA in self.letters:
            residue = self.residues[self.letters.index(selectAA)][0]
        else:
            print(f'{greyDark}Residue not recognized:{red} {selectAA}{greyDark}\n'
                  f'Please check input:{red} self.fixedAA')
            sys.exit(1)
        print(f'Fixed Residues:{red} {self.datasetTag}{resetColor}\n'
              f'Selected Residue:{red} {residue}{resetColor}\n')

        # Extract data
        initial = probInitial[probInitial.index.str.contains(selectAA)].T
        final = probFinal[probFinal.index.str.contains(selectAA)].T
        print(f'{purple}Initial Sort{resetColor}:\n{initial}\n')
        print(f'{purple}Final Sort{resetColor}:\n{final}\n\n')
        print(f'Pos: {self.fixedPos}')
        final = final.drop(final.index[[int(pos) - 1 for pos in self.fixedPos]])
        print(f'Remove fixed residues: {purple}Final Sort{resetColor}\n{final}\n\n')

        # Set local parameters
        self.tickLength, self.lineThickness = 4, 1
        xLabels = ['Initial Sort', 'Final Sort']

        # Determine yMax
        yMin = 0
        yMax = 1


        # Find outliers in the initial dataset
        outliersInitial = []
        Q1 = initial.quantile(0.25)
        Q3 = initial.quantile(0.75)
        IQR = Q3 - Q1
        outliers = initial[(initial < Q1 - 1.5 * IQR) | (initial > Q3 + 1.5 * IQR)]
        # Iterate over the indices of outliers
        for index, row in outliers.iterrows():
            if not row.isnull().all():
                outliersInitial.append(index)

        # Find outliers in the final dataset
        outliersFinal = []
        Q1 = final.quantile(0.25)
        Q3 = final.quantile(0.75)
        IQR = Q3 - Q1
        outliers = final[(final < Q1 - 1.5 * IQR) | (final > Q3 + 1.5 * IQR)]
        # Iterate over the indices of outliers
        for index, row in outliers.iterrows():
            if not row.isnull().all():
                outliersFinal.append(index)

        # Print: Outliers
        if len(outliersInitial) != 0:
            print(f'Outliers: {purple}Initial Sort{resetColor}')
            for outlierPosition in outliersInitial:
                print(f'     {outlierPosition}')
            print()
        else:
            print(f'There were no{red} {residue}{resetColor} RF outliers in: '
                  f'{purple}Initial Sort{resetColor}\n')
        if len(outliersFinal) != 0:
            print(f'Outliers: {purple}Final Sort{resetColor}')
            for outlierPosition in outliersFinal:
                print(f'     {outlierPosition}')
            print('\n')
        else:
            print(f'There were no{red} {residue}{resetColor} RF outliers in: '
                  f'{purple}Final Sort{resetColor} '
                  f'fixed{red} {self.datasetTag}{resetColor} \n\n')

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=self.figSizeMini)

        # Plot the data
        initial.boxplot(
            ax=ax, positions=[0], widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='black'), whiskerprops=dict(color='black'),
            medianprops=dict(color='#F7971F', linewidth=0.5),
            flierprops=dict(marker='o', markerfacecolor='#F7971F', markersize=10))
        final.boxplot(
            ax=ax, positions=[1], widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='#BF5700'), whiskerprops=dict(color='black'),
            medianprops=dict(color='#F7971F', linewidth=0.5),
            flierprops=dict(marker='o', markerfacecolor='#F7971F', markersize=10))
        plt.subplots_adjust(top=0.898, bottom=0.098, left=0.112, right=0.917)

        # Add labels and title
        ax.set_title(f'{self.enzymeName} - {residue} RF: Fixed {self.datasetTag}',
                     fontsize=self.labelSizeTitle, fontweight='bold')
        ax.set_ylabel('Relative Frequency', fontsize=self.labelSizeAxis)


        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks)

        # Set x & y-axis tick labels
        ax.set_xticks(range(len(xLabels)))
        ax.set_xticklabels(xLabels, fontsize=self.labelSizeAxis)
        ax.set_ylim(yMin, yMax)

        # Set the edge thickness
        for spine in ax.spines.values():
            spine.set_linewidth(self.lineThickness)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        plt.show()



    def getDatasetTag(self, combinedMotifs=False):
        if combinedMotifs:
            if len(self.fixedPos) == 1:
                self.datasetTag = f'Reading Frame {self.fixedAA[0]}@R{self.fixedPos[0]}'
            else:
                fixedPos = sorted(self.fixedPos)
                continuous = True
                for index in range(len(fixedPos) - 1):
                    pos1, pos2 = fixedPos[index], fixedPos[index + 1]
                    if pos1 == pos2 - 1 or pos1 == pos2 + 1:
                        continue
                    else:
                        continuous = False
                        break
                if continuous:
                    self.datasetTag = (f'Reading Frames {self.fixedAA[0]}@R{fixedPos[0]}-'
                                  f'R{fixedPos[-1]}')
                else:
                    self.datasetTag = (f'Reading Frames {self.fixedAA[0]}@R{fixedPos[0]}-'
                                  f'R{fixedPos[1]}, R{fixedPos[-1]}')
        else:
            if self.filterSubs:
                fixResidueList = []
                if self.excludeAAs:
                    # Exclude residues
                    for index, removedAA in enumerate(self.excludeAA):
                        if index == 0:
                            fixResidueList.append(
                                f'Excl-{removedAA}@R{self.excludePosition[index]}'.replace(
                                    ' ', ''))
                        else:
                            fixResidueList.append(
                                f'{removedAA}@R{self.excludePosition[index]}'.replace(
                                    ' ', ''))
    
                    # Fix residues
                    for index in range(len(self.fixedAA)):
                        if index == 0:
                            fixResidueList.append(
                                f'Fixed-{self.fixedAA[index]}@R'
                                f'{self.fixedPos[index]}'.replace(' ', ''))
                        else:
                            fixResidueList.append(
                                f'{self.fixedAA[index]}@R'
                                f'{self.fixedPos[index]}'.replace(' ', ''))
    
                    self.datasetTag = '_'.join(fixResidueList)
                    self.datasetTag = self.datasetTag.replace("_Fixed",
                                                                ' Fixed')
                else:
                    # Fix residues
                    for index in range(len(self.fixedAA)):
                        fixResidueList.append(
                            f'{self.fixedAA[index]}@R'
                            f'{self.fixedPos[index]}'.replace(' ', ''))
    
                    self.datasetTag = '_'.join(fixResidueList)
    
                # Condense the string
                if "'" in self.datasetTag:
                    self.datasetTag = self.datasetTag.replace("'", '')
    
                # Clean up fixed sequence tag
                if self.substrateLength == 9:
                    removeTag = 'Excl-Y@R1_Y@R2_Y@R3_Y@R4_Y@R6_Y@R7_Y@R8_Y@R9'
                    if removeTag in self.datasetTag:
                        # This should be reserved for simplifying dataset tags
                        self.datasetTag = self.datasetTag.replace(removeTag, '')
                        self.datasetTag = f'Excl-Y{self.datasetTag}'
            else:
                self.datasetTag = 'Unfiltered'
    
        # Define: Dataset Label
        if self.initialize:
            self.datasetTagMotif = self.datasetTag
            self.initialize = False

        return self.datasetTag



    def fixResidue(self, substrates, fixedString, printRankedSubs, sortType):
        print('=============================== Filter Substrates '
              '===============================')
        fixedSubs = {}
        fixedSubsTotal = 0
        print(f'Selecting {purple}{sortType} {resetColor}substrates with: '
              f'{red}{fixedString}{resetColor}\n')

        # Sort the substrate dictionary by counts
        substrates = dict(sorted(substrates.items(), key=lambda x: x[1], reverse=True))


        # Select substrates that contain selected AA at a specified position in the substrate
        if self.excludeAAs:
            # Verify if the substrates contain the residue(s) you wish to remove
            for substrate, count in substrates.items():
                # Inspect substrate count
                if count < self.minSubCount:
                    break

                keepSub = True
                for indexExclude, AAExclude in enumerate(self.excludeAA):
                    if len(AAExclude) == 1:
                        indexRemoveAA = self.excludePosition[indexExclude] - 1

                        # Is the AA acceptable?
                        if substrate[indexRemoveAA] == AAExclude:
                            keepSub = False
                            continue
                    else:
                        # Remove Multiple AA at a specific position
                        for AAExcludeMulti in AAExclude:
                            indexRemoveAA = self.excludePosition[indexExclude] - 1
                            for AAExclude in AAExcludeMulti:

                                # Is the AA acceptable?
                                if substrate[indexRemoveAA] == AAExclude:
                                    keepSub = False
                                    continue

                # If the substrate has not been blacklisted, look for the desired AA
                if keepSub:
                    if len(self.fixedAA) == 1 and len(self.fixedAA[0]) == 1:
                        # Fix only one AA
                        if substrate[self.fixedPos[0] - 1] != self.fixedAA[0]:
                            keepSub = False
                    else:
                        for indexFixed, fixedAA in enumerate(self.fixedAA):
                            indexFixAA = self.fixedPos[indexFixed] - 1

                            if len(fixedAA) == 1:
                                # Fix one AA at a given position
                                if substrate[indexFixAA] != fixedAA:
                                    keepSub = False
                                    break
                            else:
                                # Fix multiple AAs at a given position
                                if substrate[indexFixAA] not in fixedAA:
                                    keepSub = False
                                    break
                # Extract the substrate
                if keepSub:
                    fixedSubs[substrate] = count
                    fixedSubsTotal += count
        else:
            # Fix AAs, and dont exclude any AAs
            if len(self.fixedAA) == 1 and len(self.fixedAA[0]) == 1:
                for substrate, count in substrates.items():
                    # Inspect substrate count
                    if count < self.minSubCount:
                        break

                    subAA = substrate[self.fixedPos[0] - 1]
                    if subAA in self.fixedAA[0]:
                        fixedSubs[substrate] = count
                        fixedSubsTotal += count
                        continue
            else:
                for substrate, count in substrates.items():
                    # Inspect substrate count
                    if count < self.minSubCount:
                        break

                    keepSub = []
                    for index in range(len(self.fixedAA)):
                        fixIndex = self.fixedPos[index] - 1
                        subAA = substrate[fixIndex]
                        selectAA = self.fixedAA[index]

                        if subAA in selectAA:
                            keepSub.append(True)
                        else:
                            keepSub.append(False)

                    if False not in keepSub:
                        fixedSubs[substrate] = count
                        fixedSubsTotal += count


        # Rank fixed substrates
        rankedFixedSubstrates = dict(sorted(fixedSubs.items(),
                                            key=lambda x: x[1], reverse=True))

        # Print: Fixed substrates
        if printRankedSubs:
            iteration = 0
            fixedUniqueSubsTotal = len(rankedFixedSubstrates)
            print('Ranked Fixed Substrates:')
            if fixedUniqueSubsTotal == 0:
                print('')
                print(f'{orange}ERROR:\n'
                      f'     No substrates in {purple}{sortType}{orange} contained: '
                      f'{red}{fixedString}{resetColor}\n')
                sys.exit(1)
            else:
                for substrate, count in rankedFixedSubstrates.items():
                    print(f'     {pink}{substrate}{resetColor}, Counts: {red}{count:,}'
                          f'{resetColor}')
                    iteration += 1
                    if iteration >= self.printNumber:
                        break


        print(f'\nNumber of substrates with fixed {red}{fixedString}{resetColor}: '
              f'{red}{fixedSubsTotal:,}{resetColor}\n\n')

        return rankedFixedSubstrates, fixedSubsTotal



    def identifyMotif(self, fixFullFrame):
        print('================================ Identify Motif '
              '=================================')
        if fixFullFrame:
            print(f'Selecting continuous motif')
        else:
            print(f'Selecting non-continuous motif')
        print(f'Minimum ΔS Value:{red} {self.minEntropy} Bits{resetColor}\n\n'
              f'Positional Entropy:\n'
              f'{self.entropy}{resetColor}\n')
        subFrame = self.entropy.copy()
        lastPosition = len(self.entropy) - 1

        # Determine Substrate Frame
        if fixFullFrame:
            for indexPos, position in enumerate(self.entropy.index):
                if indexPos == 0 or indexPos == lastPosition:
                    if self.entropy.loc[position, 'ΔS'] < self.minEntropy:
                        subFrame.drop(position, inplace=True)
                else:
                    if self.entropy.loc[position, 'ΔS'] < self.minEntropy:
                        if (self.entropy.iloc[indexPos - 1, 0] > self.minEntropy and
                                self.entropy.iloc[indexPos + 1, 0] > self.minEntropy):
                            pass
                        else:
                            subFrame.drop(position, inplace=True)
        else:
            for indexPos, position in enumerate(self.entropy.index):
                if self.entropy.loc[position, 'ΔS'] < self.minEntropy:
                    subFrame.drop(position, inplace=True)

        # Sort the frame
        self.subFrame = subFrame.sort_values(by='ΔS', ascending=False).copy()
        print(f'Reading Frame :\n'
              f'{blue}{subFrame}{resetColor}\n\n'
              f'Ranked Motif Frame:\n'
              f'{blue}{self.subFrame}{resetColor}\n\n')

        # Define motif label
        indexSubFrameList = list(self.entropy.index)
        self.motifIndex = [indexSubFrameList.index(idx) for idx in self.subFrame.index]
        self.xAxisLabelsMotif = self.xAxisLabels[
                                min(self.motifIndex):max(self.motifIndex)+1]

        return self.subFrame



    def getMotif(self, substrates):
        print('================================= Extract Motif '
              '=================================')
        motifs = {}
        indexStart = min(self.motifIndex)
        indexEnd = max(self.motifIndex) + 1
        print(f'Reading Frame Indices: {purple}{self.datasetTag}{resetColor}\n')

        # Print: data
        iteration = 0
        print(f'Substrates:')
        for substrate, count in substrates.items():
            print(f'     {pink}{substrate}{resetColor}, '
                  f'{blue}{substrate[indexStart:indexEnd]}{resetColor}, '
                  f'Count:{red} {count:,}{resetColor}')
            iteration += 1
            if iteration >= self.printNumber:
                break
        print(f'\nUnique Substrates:{red} {len(substrates):,}{resetColor}\n\n')

        # Extract motif
        countTotalSubstrates = 0
        for substrate, count in substrates.items():
            countTotalSubstrates += count
            motif = substrate[indexStart:indexEnd]
            if motif in motifs.keys():
                motifs[motif] += count
            else:
                motifs[motif] = count
        self.motifLen = len(next(iter(motifs)))
        motifs = dict(sorted(motifs.items(), key=lambda x: x[1], reverse=True))

        # Print: data
        iteration = 0
        print(f'Motifs:')
        for motif, count in motifs.items():
            print(f'     {blue}{motif}{resetColor}, Count:{red} {count:,}{resetColor}')
            iteration += 1
            if iteration >= self.printNumber:
                print()
                break

        print(f'Unique Motifs:{red} {len(motifs):,}{resetColor}\n\n')

        return motifs



    def calculateEnrichment(self, probInitial, probFinal,
                            releasedCounts=False, combinedMotifs=False):
        print('========================== Calculate: Enrichment Score '
              '==========================')
        print(f'Enrichment Scores:\n'
              f'     {magenta}log₂(prob FinalAA / prob InitialAA){resetColor}\n\n')
        # Calculate: Enrichment scores
        if len(probInitial.columns) == 1:
            matrix = pd.DataFrame(0.0, index=probFinal.index,
                                  columns=probFinal.columns)
            for position in probFinal.columns:
                matrix.loc[:, position] = np.log2(probFinal.loc[:, position] /
                                                  probInitial.iloc[:, 0])
        else:
            probInitial.columns = probFinal.columns
            matrix  = np.log2(probFinal / probInitial)
        print(f'Enrichment Score: {purple}{self.datasetTag}{resetColor}\n\n'
              f'{matrix.round(self.roundVal)}\n\n')


        print('====================== Calculate: Scaled Enrichment Score '
              '=======================')
        print(f'Scale Enrichment Scores:\n'
              f'     {magenta}Enrichment Scores * ΔS{resetColor}\n')

        # Calculate: Letter heights
        heights = pd.DataFrame(0, index=matrix.index,
                               columns=matrix.columns, dtype=float)
        for indexColumn in heights.columns:
            heights.loc[:, indexColumn] = (matrix.loc[:, indexColumn] *
                                           self.entropy.loc[indexColumn, 'ΔS'])

        # Record values
        if releasedCounts:
            self.eMapReleased = matrix
            self.eMapReleasedScaled = heights.copy()
        else:
            self.eMap = matrix
            self.eMapScaled = heights.copy()

        heights.replace([np.inf, -np.inf], 0, inplace=True)
        self.heights = heights
        print(f'Residue Heights: {purple}{self.datasetTag}{resetColor}\n'
              f'{heights}\n\n')

        # Plot: Enrichment Map
        if self.plotFigEM:
            self.plotEnrichmentScores(dataType='Enrichment',
                                      releasedCounts=releasedCounts,
                                      combinedMotifs=combinedMotifs)
        if self.plotFigEMScaled:
            self.plotEnrichmentScores(dataType='Scaled Enrichment',
                                      releasedCounts=releasedCounts,
                                      combinedMotifs=combinedMotifs)

        # Plot: Enrichment Logo
        if self.plotFigLogo:
            self.plotEnrichmentLogo(releasedCounts=releasedCounts,
                                    combinedMotifs=combinedMotifs)

        # Calculate & Plot: Weblogo
        if self.plotFigWebLogo:
            self.calculateWeblogo(probability=probFinal, releasedCounts=releasedCounts,
                                  combinedMotifs=combinedMotifs)

        return self.eMap



    def plotEnrichmentScores(self, dataType, combinedMotifs=False, releasedCounts=False):
        print('============================ Plot: Enrichment Score '
              '=============================')
        # Select: Dataset
        if 'scaled' in dataType.lower():
            if releasedCounts:
                scores = self.eMapReleasedScaled
            else:
                scores = self.eMapScaled
        else:
            if releasedCounts:
                scores = self.eMapReleased
            else:
                scores = self.eMap

        # Define: Figure title
        if releasedCounts:
            title = self.titleReleased
        elif combinedMotifs and len(self.motifIndexExtracted) > 1:
            title = self.titleCombined
        elif combinedMotifs:
            title = self.titleCombined
            title = title.replace('Combined ', '')
        else:
            title = self.title


        if self.motifFilter:
            print(f'Figure Number: '
                  f'{magenta}{self.saveFigureIteration}{resetColor}')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n\n'
              f'{scores}\n\n')


        # Create heatmap
        cMapCustom = self.createCustomColorMap(colorType='EM')

        # Define the yLabel
        if self.residueLabelType == 0:
            scores.index = [residue[0] for residue in self.residues]
        elif self.residueLabelType == 1:
            scores.index = [residue[1] for residue in self.residues]
        elif self.residueLabelType == 2:
            scores.index = [residue[2] for residue in self.residues]

        # Define color bar limits
        if np.max(scores) >= np.min(scores):
            cBarMax = np.max(scores)
            cBarMin = -1 * cBarMax
        else:
            cBarMin = np.min(scores)
            cBarMax = -1 * cBarMin

        # Plot the heatmap with numbers centered inside the squares
        fig, ax = plt.subplots(figsize=self.figSizeEM)
        if self.figEMSquares:
            heatmap = sns.heatmap(scores, annot=False, cmap=cMapCustom, cbar=True,
                                  linewidths=self.lineThickness - 1, linecolor='black',
                                  square=self.figEMSquares, center=None,
                                  vmax=cBarMax, vmin=cBarMin)
        else:
            heatmap = sns.heatmap(scores, annot=True, fmt='.3f', cmap=cMapCustom,
                                  cbar=True, linewidths=self.lineThickness - 1,
                                  linecolor='black', square=self.figEMSquares,
                                  center=None, vmax=cBarMax, vmin=cBarMin,
                                  annot_kws={'fontweight': 'bold'})
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        ax.set_xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        ax.set_ylabel('Residue', fontsize=self.labelSizeAxis)
        if self.figEMSquares:
            figBorders = [0.852, 0.075, 0, 0.895]
        else:
            figBorders = [0.852, 0.075, 0.117, 1]  # Top, bottom, left, right
        plt.subplots_adjust(top=figBorders[0], bottom=figBorders[1],
                            left=figBorders[2], right=figBorders[3])

        # Set the thickness of the figure border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', rotation=0, length=self.tickLength,
                       labelsize=self.labelSizeTicks, width=self.lineThickness)

        # Set x-ticks
        xTicks = np.arange(len(scores.columns)) + 0.5
        ax.set_xticks(xTicks)
        ax.set_xticklabels(scores.columns)

        # Set y-ticks
        yTicks = np.arange(len(scores.index)) + 0.5
        ax.set_yticks(yTicks)
        ax.set_yticklabels(scores.index)

        # Set invalid values to grey
        cmap = plt.cm.get_cmap(cMapCustom)
        cmap.set_bad(color='lightgrey')

        # Modify the colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(axis='y', which='major', labelsize=self.labelSizeTicks,
                            length=self.tickLength, width=self.lineThickness)
        cbar.outline.set_linewidth(self.lineThickness)
        cbar.outline.set_edgecolor('black')

        fig.canvas.mpl_connect('key_press_event', pressKey)
        if self.setFigureTimer:
            plt.ion()
            plt.show()
            plt.pause(self.figureTimerDuration)
            plt.close(fig)
            plt.ioff()
        else:
            plt.show()


        # Save the figure
        if self.saveFigures:
            if 'Scaled' in dataType:
                datasetType = 'EM Scaled'
            elif 'Enrichment' in dataType:
                datasetType = 'EM'
            else:
                print(f'{orange}ERROR: What do I do with this dataset type -'
                      f'{cyan} {dataType}{resetColor}\n')
                sys.exit(1)
            self.saveFigure(fig=fig, figType=datasetType, seqLen=len(xTicks),
                            combinedMotifs=combinedMotifs, releasedCounts=releasedCounts)



    def plotEnrichmentLogo(self, combinedMotifs=False, releasedCounts=False):
        print('============================= Plot: Enrichment Logo '
              '=============================')
        # Define: Figure title
        if releasedCounts:
            title = self.titleReleased
        elif combinedMotifs and len(self.motifIndexExtracted) > 1:
            title = self.titleCombined
        elif combinedMotifs:
            title = self.titleCombined
            title = title.replace('Combined ', '')
        else:
            title = self.title

        # Print: data
        if self.motifFilter:
            print(f'Figure Number: '
                  f'{magenta}{self.saveFigureIteration}{resetColor}')
        print(f'Residue heights:\n'
              f'{self.heights}\n')


        # Calculate: Max and min
        columnTotals = [[], []]
        for indexColumn in self.heights.columns:
            totalPos = 0
            totalNeg = 0
            for value in self.heights.loc[:, indexColumn]:
                if value > 0:
                    totalPos += value
                elif value < 0:
                    totalNeg += value
            columnTotals[0].append(totalPos)
            columnTotals[1].append(totalNeg)
        yMax = max(columnTotals[0])
        yMin = min(columnTotals[1])
        print(f'y Max: {red}{np.round(yMax, 4)}{resetColor}\n'
              f'y Min: {red}{np.round(yMin, 4)}{resetColor}\n\n')

        # Rename columns for logomaker script
        data = self.heights.copy()
        data.columns = range(len(data.columns))

        # Set local parameters
        if self.bigAAonTop:
            stackOrder = 'big_on_top'
        else:
            stackOrder = 'small_on_top'

        # Set: Figure borders
        if self.showSampleSize:
            figBorders = [0.852, 0.075, 0.164, 0.938]
        else:
            figBorders = [0.852, 0.075, 0.164, 0.938]

        # Plot the sequence motif
        fig, ax = plt.subplots(figsize=self.figSize)
        motif = logomaker.Logo(data.transpose(), ax=ax, color_scheme=self.colorsAA,
                               width=0.95, stack_order=stackOrder)
        motif.ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        plt.subplots_adjust(top=figBorders[0], bottom=figBorders[1],
                            left=figBorders[2], right=figBorders[3])

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks)

        # Set borders
        motif.style_spines(visible=False)
        motif.style_spines(spines=['left', 'bottom'], visible=True)
        for spine in motif.ax.spines.values():
            spine.set_linewidth(self.lineThickness)

        # Set x-ticks
        motif.ax.set_xticks([pos for pos in range(len(self.heights.columns))])
        motif.ax.set_xticklabels(self.heights.columns, fontsize=self.labelSizeTicks,
                                 rotation=0, ha='center')

        # Set y-ticks
        yTicks = [yMin, 0, yMax]
        yTickLabels = [f'{tick:.2f}' if tick != 0 else f'{int(tick)}' for tick in yTicks]
        motif.ax.set_yticks(yTicks)
        motif.ax.set_yticklabels(yTickLabels, fontsize=self.labelSizeTicks)
        motif.ax.set_ylim(yMin, yMax)

        # Set tick width
        for tick in motif.ax.xaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness)
        for tick in motif.ax.yaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness)

        # Label the axes
        motif.ax.set_xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        motif.ax.set_ylabel('Scaled Enrichment', fontsize=self.labelSizeAxis)

        # Set horizontal line
        motif.ax.axhline(y=0, color='black', linestyle='-', linewidth=self.lineThickness)

        # Evaluate dataset for fixed residues
        spacer = np.diff(motif.ax.get_xticks()) # Find the space between each tick
        spacer = spacer[0] / 2

        # Use the spacer to set a gray background to fixed residues
        for index, position in enumerate(self.xAxisLabels):
            if position in self.fixedPos:
                # Plot gray boxes on each side of the xtick
                motif.ax.axvspan(index - spacer, index + spacer,
                                 facecolor='darkgrey', alpha=0.2)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        if self.setFigureTimer:
            plt.ion()
            plt.show()
            plt.pause(self.figureTimerDuration)
            plt.close(fig)
            plt.ioff()
        else:
            plt.show()

        # Save the figure
        if self.saveFigures:
            datasetType = 'Logo'
            self.saveFigure(fig=fig, figType=datasetType,  seqLen=len(data.columns),
                            combinedMotifs=combinedMotifs, releasedCounts=releasedCounts)



    def calculateWeblogo(self, probability, combinedMotifs=False, releasedCounts=False):
        print('============================= Calculate: Weblogo '
              '================================')
        print(f'Probability: {self.datasetTag}\n{probability}\n\n'
              f'Max Entropy: {red}{self.entropyMax.round(6)}{resetColor}\n\n')

        # Calculate: Weblogo
        self.weblogo = pd.DataFrame(0, index=probability.index,
                                    columns=probability.columns, dtype=float)
        for indexColumn in self.weblogo.columns:
            self.weblogo.loc[:, indexColumn] = (probability.loc[:, indexColumn] *
                                                self.entropy.loc[indexColumn, 'ΔS'])

        if self.plotFigWebLogo:
            self.plotWeblogo(combinedMotifs=combinedMotifs, releasedCounts=releasedCounts)

        return self.weblogo



    def plotWeblogo(self, combinedMotifs=False, releasedCounts=False):
        print('================================= Plot: Weblogo '
              '=================================')
        if self.motifFilter:
            print(f'Figure Number: '
                  f'{magenta}{self.saveFigureIteration}{resetColor}')
        print(f'Letter Heights: {purple}{self.datasetTag}{resetColor}\n'
              f'{self.weblogo}\n\n')

        # Define: Figure title
        if releasedCounts:
            title = self.titleReleased
        elif combinedMotifs and len(self.motifIndexExtracted) > 1:
            title = self.titleWeblogoCombined
        elif combinedMotifs:
            title = self.titleWeblogoCombined
            title = title.replace(f'Combined ', '')
        else:
            title = self.titleWeblogo

        # Set: Figure borders
        figBorders = [0.852, 0.075, 0.112, 0.938]

        # Set local parameters
        if self.bigAAonTop:
            stackOrder = 'big_on_top'
        else:
            stackOrder = 'small_on_top'

        # Rename column headers
        dataColumnsReceived = self.weblogo.columns
        self.weblogo.columns = range(len(self.weblogo.columns))

        # Set -inf to zero
        if self.weblogo.isin([np.inf, -np.inf]).any().any():
            self.weblogo.replace([np.inf, -np.inf], 0, inplace=True)

        # Plot the sequence motif
        fig, ax = plt.subplots(figsize=self.figSize)
        motif = logomaker.Logo(self.weblogo.transpose(), ax=ax,
                               color_scheme=self.colorsAA, width=0.95,
                               stack_order=stackOrder)
        motif.ax.set_title(title, fontsize=self.labelSizeTitle,
                           fontweight='bold')
        plt.subplots_adjust(top=figBorders[0], bottom=figBorders[1],
                            left=figBorders[2], right=figBorders[3])

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks)

        # Set borders
        motif.style_spines(visible=False)
        motif.style_spines(spines=['left', 'bottom'], visible=True)
        for spine in motif.ax.spines.values():
            spine.set_linewidth(self.lineThickness)

        # Set x-ticks
        if len(dataColumnsReceived) == len(self.xAxisLabels):
            motif.ax.set_xticks([pos for pos in range(len(self.xAxisLabels))])
            motif.ax.set_xticklabels(self.xAxisLabels, fontsize=self.labelSizeTicks,
                                     rotation=0, ha='center')
        else:
            motif.ax.set_xticks([pos for pos in range(len(dataColumnsReceived))])
            motif.ax.set_xticklabels(dataColumnsReceived, fontsize=self.labelSizeTicks,
                                     rotation=0, ha='center')

        for tick in motif.ax.xaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness)  # Set tick width

        # Set y-ticks
        yTicks = range(0, 5)
        yTickLabels = [f'{tick:.0f}' if tick != 0 else f'{int(tick)}'
                       for tick in yTicks]
        motif.ax.set_yticks(yTicks)
        motif.ax.set_yticklabels(yTickLabels, fontsize=self.labelSizeTicks)
        motif.ax.set_ylim(0, self.entropyMax)
        for tick in motif.ax.yaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness) # Set tick width

        # Label the axes
        motif.ax.set_xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        motif.ax.set_ylabel('Bits', fontsize=self.labelSizeAxis)

        # Set horizontal line
        motif.ax.axhline(y=0, color='black', linestyle='-', linewidth=self.lineThickness)

        # Evaluate dataset for fixed residues
        spacer = np.diff(motif.ax.get_xticks()) # Find the space between each tick
        spacer = spacer[0] / 2

        # Use the spacer to set a gray background to fixed residues
        for index, position in enumerate(self.xAxisLabels):
            if position in self.fixedPos:
                # Plot gray boxes on each side of the xtick
                motif.ax.axvspan(index - spacer, index + spacer,
                                 facecolor='darkgrey', alpha=0.2)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        if self.setFigureTimer:
            plt.ion()
            plt.show()
            plt.pause(self.figureTimerDuration)
            plt.close(fig)
            plt.ioff()
        else:
            plt.show()


        # Save the figure
        if self.saveFigures:
            datasetType = 'Weblogo'
            self.saveFigure(
                fig=fig, figType=datasetType, seqLen=len(self.weblogo.columns),
                combinedMotifs=combinedMotifs, releasedCounts=releasedCounts)



    def fixedMotifStats(self, countsList, initialProb, motifFrame, datasetTag):
        print('================== Statistical Evaluation: Fixed Motif Counts '
              '===================')
        print(f'Evaluate: {purple}{datasetTag}{resetColor}\n')
        countsFrameTotal = pd.DataFrame(0, index=range(0, len(self.fixedPos)),
                                        columns=motifFrame)
        frameProb = pd.DataFrame(0.0, index=initialProb.index,
                                 columns=motifFrame)
        frameES = frameProb.copy()
        frameESList = []

        for index, countsFrame in enumerate(countsList):
            countsFrame = pd.DataFrame(countsFrame, index=initialProb.index,
                                       columns=motifFrame)

            # Format values to have commas
            formattedCounts = countsFrame.to_string(
                formatters={column: '{:,.0f}'.format for column in
                            countsFrame.select_dtypes(include='number').columns})
            print(f'Counts: {purple}Fixed Motif '
                  f'{self.fixedAA[0]}@R{self.fixedPos[index]}{resetColor}\n'
                  f'{formattedCounts}\n')

            for column in countsFrame.columns:
                countsFrameTotal.loc[index, column] = countsFrame[column].sum()

                # Calculate: RF
                frameProb.loc[:, column] = (countsFrame.loc[:, column] /
                                            countsFrameTotal.loc[index, column])

                # Calculate: ES
                frameES.loc[:, column] = np.log2(
                    frameProb.loc[:, column] / initialProb['Average RF'])

            print(f'Enrichment Score:'
                  f'{purple}Fixed Motif {self.fixedAA[0]}@R{self.fixedPos[index]}\n'
                  f'{greenLight}{frameES}{resetColor}\n\n')
            frameESList.append(frameES.copy())

        # Combine the ES DFs
        frameESCombined = pd.concat(frameESList, axis=0, keys=range(len(frameESList)))

        def calcAverage(x):
            # Function to calcglate standard deviation ignoring -inf values

            # Remove -inf values
            xFiltered = x.replace([-np.inf, np.inf], np.nan).dropna()

            return np.average(xFiltered)

        def calcStDev(x):
            # Function to calculate standard deviation ignoring -inf values

            # Remove -inf values
            xFiltered = x.replace([-np.inf, np.inf], np.nan).dropna()

            return np.std(xFiltered)

        # Calculate standard deviation for each value across the corresponding positions
        frameESAvg = frameESCombined.groupby(level=1, sort=False).agg(calcAverage)
        frameESStDev = frameESCombined.groupby(level=1, sort=False).agg(calcStDev)

        print(f'Average: {purple}Enrichment Score\n'
              f'{greyDark}{frameESAvg}{resetColor}\n\n'
              f'Standard Deviation: {purple}Enrichment Score\n'
              f'{greyDark}{frameESStDev}{resetColor}\n\n')


        # Plot: Standard deviation
        self.plotStats(countedData=frameESStDev, totalCounts=None, dataType='StDev')



    def processSubstrates(self, subsInit, subsFinal, motifs, subLabel,
                          combinedMotifs=False, predActivity=False, predModel=False):
        if predActivity:
            # Calculate: Motif enrichment
            for predType, predictions in motifs.items():
                print(f'Evaluating Predictions: {purple}{predType}{resetColor}\n'
                      f'Predictions: {type(predictions)}')
                motifES = self.motifEnrichment(
                    subsInit=subsInit, subsFinal=subsFinal, motifs=predictions,
                    predActivity=predActivity, predModel=predModel, predType=predType)

                # Plot: Work cloud
                self.plotWordCloud(
                    substrates=motifES, combinedMotifs=combinedMotifs,
                    predActivity=predActivity, predModel=predModel)

            return None
        else:
            # Calculate: Motif enrichment
            motifES = self.motifEnrichment(
                subsInit=subsInit, subsFinal=subsFinal, motifs=motifs,
                predActivity=predActivity, predModel=predModel)

        # Plot: Work cloud
        if self.plotFigWords:
            self.plotWordCloud(substrates=motifES, combinedMotifs=combinedMotifs)


        predMotif = False
        if predMotif:
            # Predict: Motif enrichment
            self.predMotifEnrichment(motifES=motifES)

        # Plot: Bar graphs
        if self.plotFigBars:
            self.plotBarGraph(substrates=motifs, dataType='Counts',
                              combinedMotifs=combinedMotifs, subsInit=subsInit)
            self.plotBarGraph(substrates=motifs, dataType='Probability',
                              combinedMotifs=combinedMotifs)

        # PCA
        if self.plotFigPCA:
            # Convert substrate data to numerical
            tokensESM, subsESM, subCountsESM = self.ESM(
                substrates=motifES, subLabel=subLabel)

            # Cluster substrates
            subPopulations = self.plotPCA(substrates=motifES, data=tokensESM,
                                          indices=subsESM, N=subCountsESM,
                                          combinedMotifs=combinedMotifs)
            for NCluster, motifCluster in enumerate(subPopulations):
                NClusterAdj = NCluster + 1

                # Plot: Motif enrichment for this selected cluster
                # if self.plotFigMotifEnrich:
                self.plotMotifEnrichment(motifs=motifCluster,
                                         combinedMotifs=combinedMotifs,
                                         clusterNumPCA=NClusterAdj)
                # if self.plotFigMotifEnrichSelect:
                self.plotMotifEnrichment(motifs=motifCluster,
                                         combinedMotifs=combinedMotifs,
                                         clusterNumPCA=NClusterAdj,
                                         limitNBars=True)

                # Plot: Work cloud
                self.plotWordCloud(substrates=motifCluster, clusterNumPCA=NClusterAdj,
                                   combinedMotifs=combinedMotifs)

        # Suffix tree
        if self.plotSuffixTree:
            print(f'ADD: Suffix tree')
            sys.exit()

        return motifES



    def plotMotifEnrichment(self, motifs, barColor='#CC5500', barWidth=0.65,
                            clusterNumPCA=None, combinedMotifs=False, limitNBars=False,
                            predActivity=False, predModel=None, predType=None):
        NSubs = len(motifs.keys())
        if predActivity:
            if predType.lower() == 'custom':
                # Collect all datapoints
                x, y = [], []
                for motif, count in motifs.items():
                    x.append(str(motif))
                    y.append(count)
            else:
                # Collect top datapoints
                iteration = 0
                x, y = [], []
                for motif, count in motifs.items():
                    x.append(str(motif))
                    y.append(count)
                    iteration += 1
                    if iteration == self.NSubBars:
                        break
        else:
            if limitNBars:
                # Collect top datapoints
                iteration = 0
                x, y = [], []
                for motif, count in motifs.items():
                    x.append(str(motif))
                    y.append(count)
                    iteration += 1
                    if iteration == self.NSubBars:
                        break
            else:
                # Collect all datapoints
                x, y = [], []
                for motif, count in motifs.items():
                    x.append(str(motif))
                    y.append(count)
        plotNSubs = len(x)
        motifLen = len(x[0])

        # Evaluate: Y axis
        maxValue = math.ceil(max(y))
        magnitude = math.floor(math.log10(maxValue))
        unit = 10 ** (magnitude - 1)
        yMax = math.ceil(maxValue / unit) * unit
        yMax += 3 * unit # Increase yMax
        yMin = 0


        # # Calculate: Decay constant
        # k = ngs.decayRate(y=y)


        # Plot the data
        fig, ax = plt.subplots(figsize=self.figSize)
        if limitNBars:
            bars = plt.bar(x, y, color=barColor, width=barWidth)

            # Determine x values
            xTicks = np.arange(0, len(x))
            ax.set_xticks(xTicks)
            ax.set_xticklabels(x, rotation=90, ha='center')
            ax.set_xlim(left=xTicks[0] - barWidth, right=xTicks[-1] + barWidth)

            # Set the edge color
            for bar in bars:
                bar.set_edgecolor('black')
        else:
            bars = plt.bar(x, y, color=barColor, width=barWidth)

            # Determine x values
            magnitude = np.floor(np.log10(NSubs))
            div = 10 ** (magnitude - 1)
            xMax = int(np.ceil(NSubs))
            step = int(div * 10)  # int(xMax / 10)
            xTicks = np.arange(0, xMax + step, step)
            ax.set_xticks(xTicks)
            ax.set_xticklabels(xTicks)
            ax.set_xlim(left=-NSubs/30, right=xTicks[-1])

            # Set the edge color
            for bar in bars:
                bar.set_edgecolor(barColor)

        # Define: Figure title
        yLabel = 'Enrichment Factor'
        if predActivity:
            title = (f'{self.enzymeName}\n{predModel}\n'
                         f'{NSubs:,} {predType} Sequences')
            yLabel = 'Predicted Activity'
        else:
            if clusterNumPCA is not None:
                title = (f'{self.enzymeName}\n{self.datasetTag}\n'
                         f'{NSubs:,} Unique Motifs - PCA Cluster #{clusterNumPCA}')
            else:
                title = (f'{self.enzymeName}\n{self.datasetTag}\n'
                         f'{NSubs:,} Unique Motifs')
            if combinedMotifs and len(self.motifIndexExtracted) > 1:
                title = title.replace(self.datasetTag,
                                      f'Combined {self.datasetTag}')
        if limitNBars and predType.lower() != 'chosen':
            title = title.replace(f'{NSubs:,}', f'Top {plotNSubs:,}')
        plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        plt.ylabel(yLabel, fontsize=self.labelSizeAxis)
        plt.axhline(y=0, color='black', linewidth=self.lineThickness)
        plt.ylim(yMin, yMax)

        # Set: y ticks
        step = yMax / 10
        yTicks = np.arange(0, yMax + step, step)
        plt.xticks(rotation=90, ha='center')
        plt.yticks(yTicks)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks, width=self.lineThickness)

        # Set the thickness of the figure border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        fig.tight_layout()
        plt.show()

        # Save the figure
        if self.saveFigures:
            # Define: Save location
            if predActivity:
                if predType.lower() == 'chosen':
                    figLabel = (f'{self.enzymeName} - Predicted Activity - '
                                f'{self.datasetTag} - {predType} Subs - {predModel} - '
                                f'{motifLen} AA - Plot N {plotNSubs}.png')
                else:
                    figLabel = (f'{self.enzymeName} - Predicted Activity - '
                                f'{self.datasetTag} - {predType} Subs - {predModel} - '
                                f'{motifLen} AA - Select N {self.NSubBars} '
                                f'Plot N {plotNSubs}.png')
            else:
                if limitNBars:
                    figLabel = (f'{self.enzymeName} - Motif Enrichment - '
                                f'{self.datasetTag} - {motifLen} AA - '
                                f'Select N {self.NSubBars} Plot N {plotNSubs} - '
                                f'MinCounts {self.minSubCount}.png')
                else:
                    figLabel = (f'{self.enzymeName} - Motif Enrichment - '
                                f'{self.datasetTag} - {motifLen} AA - '
                                f'N {plotNSubs} - MinCounts {self.minSubCount}.png')
                if combinedMotifs and len(self.motifIndexExtracted) > 1:
                    figLabel = figLabel.replace(self.datasetTag,
                                                f'Combined {self.datasetTag}')
                if clusterNumPCA is not None:
                    figLabel = figLabel.replace(
                        'Motif Enrichment',
                        f'Motif Enrichment - PCA {clusterNumPCA}')

            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                print(f'{yellow}The figure was not saved\n\n'
                      f'File was already found at path:\n'
                      f'     {saveLocation}{resetColor}\n\n')
            else:
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)



    def predMotifEnrichment(self, motifES, scaledMatrix=False):
        print('=========================== Predict: Motif Enrichment '
              '===========================')
        if scaledMatrix:
            matrixType = 'Released Count Scaled Enrichment Scores'
            matrix = self.eMapReleasedScaled
        else:
            matrixType = 'Released Count Enrichment Scores'
            matrix = self.eMapReleased
        print(f'Prediction Matrix: {magenta}{matrixType}{resetColor}\n'
              f'{self.eMap}\n\n')

        predScores = {}
        maxScore, minScore = 0, 0
        normalizeValues = True

        for motif in motifES:
            score = 0
            for index, AA in enumerate(motif):
                position = self.xAxisLabelsMotif[index]
                score += matrix.loc[AA, position]
            # score = np.log2(score)
            predScores[motif] = score
            if score > maxScore:
                maxScore = score
            if score < minScore:
                minScore = score

        if normalizeValues:
            # Set minimum predicted score = 0
            if minScore < 0:
                for substrate, score in predScores.items():
                    newScore = score - minScore
                    predScores[substrate] = newScore

                    # Update max score
                    if newScore > maxScore:
                        maxScore = newScore
                minScore -= minScore
            else:
                for substrate, score in predScores.items():
                    newScore = score + minScore
                    predScores[substrate] = newScore
                maxScore += minScore

            # Normalize Values
            for substrate, score in predScores.items():
                predScores[substrate] = score / maxScore

        iteration = 0
        print(f'Predicted Scores:')
        for substrate, score in predScores.items():
            print(f'     {pink}{substrate}{resetColor}, '
                  f'Score: {red}{np.round(score, 2)}{resetColor}')
            iteration += 1
            if iteration >= self.printNumber:
                print('\n')
                break

        self.plotScatter(valuesExp=motifES, valuesPred=predScores, matrixType=matrixType)



    def plotScatter(self, valuesExp, valuesPred, matrixType, color='#CC5500'):
        x, y = valuesExp.values(), valuesPred.values()

        # Normalize predicted values
        xMax, xMin, xNorm = max(x), min(x), []
        xAdj = xMax - xMin
        for value in x:
            xNorm.append((value - xMin) / xAdj)
        print(f'Experimental Values:\n'
              f'     Max: {red}{max(x)}{resetColor}'
              f'     Min: {red}{min(x)}{resetColor}\n'
              f'Normalized Experimental Values:\n'
              f'     Max: {red}{max(xNorm)}{resetColor}'
              f'     Min: {red}{min(xNorm)}{resetColor}\n\n'
              f'Predicted Values:\n'
              f'     Max: {red}{max(y)}{resetColor}'
              f'     Min: {red}{min(y)}{resetColor}\n\n')


        accuracy = 1.0
        label = f'R² = {np.round(accuracy, 3)}'
        title = (f'{self.enzymeName}\n'
                 f'Combined {self.datasetTag}\n'
                 f'{matrixType}')

        # Create scatter plot
        fig, ax = plt.subplots(figsize=self.figSize)
        ax.scatter(xNorm, y, color=color)
        plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        plt.xlabel('Motif Enrichment', fontsize=self.labelSizeAxis)
        plt.ylabel('Predicted Scores', fontsize=self.labelSizeAxis)
        plt.subplots_adjust(top=0.852, bottom=0.076, left=0.145, right=0.938)

        # Define: Axis ticks
        ticks = np.linspace(0, 1, 5)
        tickLabels = ['0', '0.25', '0.5', '0.75', '1']

        # Set x-ticks
        ax.set_xticks(ticks)
        ax.set_xticklabels(tickLabels)
        ax.set_xlim(-0.03, 1.03)

        # Set y-ticks
        ax.set_yticks(ticks)
        ax.set_yticklabels(tickLabels)
        ax.set_ylim(-0.03, 1.03)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks, width=self.lineThickness)

        # Set the thickness of the figure border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        plt.show()



    def motifEnrichment(self, subsInit, subsFinal, motifs,
                        predActivity=False, predModel=False, predType=False):
        print('=============================== Motif Enrichment '
              '================================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n\n'
              f'Normalizing substrate counts in the '
              f'{purple}Final Sort{resetColor} library:\n'
              f'     Enrichment Ratio (ER) = {magenta}Counts Final{resetColor} / '
              f'{magenta}Counts Initial{resetColor}\n')
        combinedMotifs = False
        if len(self.motifIndexExtracted) > 1:
            combinedMotifs = True

        if predActivity:
            motifEnrichment = motifs
        else:
            k = None
            motifEnrichment = {}
            ratios = {}
            totalCountsInit = 0
            totalCountsInitAdj = 0
            totalCountsFinal = 0
            totalUniqueSubsFinal = len(subsFinal)


            # ============================================================================

            # Calc: ER
                # Use: CountFinal / CountInit

                # Or Use: (CountFinal / NFinalSubs) / (CountInit / NInitSubs)

                # Use?: log2(ratio)

            # ============================================================================

            # Sort input sequences
            subsFinal = dict(sorted(subsFinal.items(),
                                          key=lambda x: x[1], reverse=True))

            # Print: Substrates
            iteration = 0
            print(f'Substrates: {magenta}Final Sort{resetColor}')
            for substrate, count in subsFinal.items():
                print(f'     {pink}{substrate}{resetColor}, '
                      f'Count:{red} {count:,}{resetColor}')
                iteration += 1
                if iteration >= self.printNumber:
                    print('')
                    break


            # Evaluate: Motif enrichment
            for substrate, count in subsFinal.items():
                totalCountsFinal += count
                if substrate in subsInit.keys():
                    # Limit subInit length to match frame extraction zones
                    countInit = subsInit[substrate]
                else:
                    countInit = 1
                    totalCountsFinal += 1
                totalCountsInit += countInit
                if countInit == 1:
                    totalCountsInitAdj += countInit
                ratios[substrate] = count / countInit

            print(f'Motif Indices: {self.motifIndexExtracted}\n')
            # sys.exit()

            # Sort collected substrates and add to the list
            ratios = dict(sorted(ratios.items(), key=lambda x: x[1], reverse=True))

            iteration = 0
            print('Enrichment Ratios:')
            for substrate, ratio in ratios.items():
                # if int(ratio) != subsFinal[substrate]:
                iteration += 1
                print(f'     {pink}{substrate}{resetColor}, '
                      f'ER: {red}{round(ratio, 1):,}{resetColor}')
                if iteration >= self.printNumber:
                    print('')
                    break

            print(f'Unique Substrates: {red}{totalUniqueSubsFinal:,}{resetColor}\n'
                  f'Counts +1: {red}{totalCountsInitAdj:,}{resetColor}\n'
                  f'Percent +1: {yellow}'
                  f'{round(100*(totalCountsInitAdj / 
                                totalUniqueSubsFinal), self.roundVal)} %'
                  f'{resetColor}')


            # Evaluate: Motifs
            for motif in motifs.keys():
                for substrate, ratio in ratios.items():
                    if motif in substrate:
                        if motif in motifEnrichment.keys():
                            motifEnrichment[motif] += ratio
                        else:
                            motifEnrichment[motif] = ratio
            totalMotifs = len(motifs.keys())
            print(f'Unique Motifs: {red}{totalMotifs:,}{resetColor}\n')

            # Sort collected substrates and add to the list
            motifEnrichment = dict(sorted(motifEnrichment.items(),
                                          key=lambda x: x[1], reverse=True))

        iteration = 0
        if predActivity:
            print(f'Predicted Activity: {pink}Top {predType} Sequences{resetColor}')
            print(f'Model: {purple}{predModel}{resetColor}')
            for substrate, value in motifEnrichment.items():
                value = float(value)
                print(f'     {pink}{substrate}{resetColor}: {red}{round(value, 3):,}'
                      f'{resetColor}')
                iteration += 1
                if iteration >= self.printNumber:
                    break
            print('\n')
        else:
            print(f'Enrichment Motifs: {pink}Top Sequences{resetColor}')
            for motif, value in motifEnrichment.items():
                value = float(value)
                print(f'     {blue}{motif}{resetColor}, '
                      f'ER: {red}{round(value, 3):,}{resetColor}')
                iteration += 1
                if iteration >= self.printNumber:
                    break
            print('\n')
    

        # Plot: Motif enrichment
        if predActivity:
            self.plotMotifEnrichment(
                motifs=motifEnrichment, combinedMotifs=combinedMotifs,
                limitNBars=True, predActivity=predActivity, predModel=predModel,
                predType=predType)
        else:
            if self.plotFigMotifEnrich:
                self.plotMotifEnrichment(
                    motifs=motifEnrichment, combinedMotifs=combinedMotifs,
                    predActivity=predActivity, predModel=predModel, predType=predType)
            if self.plotFigMotifEnrichSelect:
                self.plotMotifEnrichment(
                    motifs=motifEnrichment, combinedMotifs=combinedMotifs,
                    limitNBars=True, predActivity=predActivity, predModel=predModel,
                    predType=predType)

        return motifEnrichment



    def decayRate(self, y):
        print('========================= Calculate: Exponential Decay '
              '==========================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n')
        
        # Set: xValues 
        x = np.arange(0, len(y))
        smoothx = np.linspace(x[0], x[-1], 20)

        guessA, guessB, guessC = 4000, -0.005, 100
        guess = [guessA, guessB, guessC]

        exp_decay = lambda x, A, t, y0: A * np.exp(x * t) + y0

        params, cov = curve_fit(exp_decay, x, y, p0=guess)

        A, t, y0 = params

        print("A = %s\nt = %s\ny0 = %s\n" % (A, t, y0))



    def plotBarGraph(self, substrates, dataType, barColor='#CC5500', barWidth=0.75,
                     combinedMotifs=False, subsInit=False):
        print('================================ Plot: Bar Graph '
              '================================')
        print(f'Collecting the top {red}{self.NSubBars}{resetColor} substrates\n')
        xValues = []
        yValues = []

        print(f'Substrates:')
        iteration = 0
        for substrate, count in substrates.items():
            print(f'     {blue}{substrate}{resetColor}, Counts: {red}{count}{resetColor}')
            iteration += 1
            if iteration >= self.printNumber:
                print()
                break

        # Collect substrates
        iteration = 0
        countsTotal = 0
        for count in substrates.values():
            countsTotal += count
        print(f'Total Substrates: {red}{countsTotal:,}{resetColor}')

        if 'counts' in dataType.lower():
            # Evaluate: Substrates
            for substrate, count in substrates.items():
                xValues.append(str(substrate))
                yValues.append(count)
                iteration += 1
                if iteration == self.NSubBars:
                    break

            # Evaluate: Y axis
            maxValue = math.ceil(max(yValues))
            magnitude = math.floor(math.log10(maxValue))
            unit = 10**(magnitude-1)
            yMax = math.ceil(maxValue / unit) * unit
            if yMax < max(yValues):
                increaseValue = unit / 2
                while yMax < max(yValues):
                    print(f'Increase yMax by:{yellow} {increaseValue}{resetColor}')
                    yMax += increaseValue
                print('\n')
            yMin = 0 # math.floor(min(yValues) / unit) * unit - spacer
        elif 'probability' in dataType.lower():
            # Evaluate: Substrates
            for substrate, count in substrates.items():
                xValues.append(str(substrate))
                yValues.append(count / countsTotal)
                iteration += 1
                if iteration == self.NSubBars:
                    break

            # Evaluate: Y axis
            maxValue = max(yValues)
            magnitude = math.floor(math.log10(maxValue))
            adjustedMax = maxValue * 10**abs(magnitude)
            yMax = math.ceil(adjustedMax) * 10**magnitude
            adjVal = 5 * 10**(magnitude-1)
            yMaxAdjusted = yMax - adjVal
            if yMaxAdjusted > maxValue:
                yMax = yMaxAdjusted
            yMin = 0
        else:
            # Evaluate: Substrates
            for substrate, count in substrates.items():
                xValues.append(str(substrate))
                yValues.append(count)
                iteration += 1
                if iteration == self.NSubBars:
                    break

            # Evaluate: Y axis
            spacer = 0.2
            yMax = math.ceil(max(yValues)) + spacer
            yMin = math.floor(min(yValues))
        NSubs = len(xValues)
        print(f'Number of plotted sequences: {red}{NSubs}{resetColor}\n\n')

        # Define: Figure title
        if combinedMotifs and len(self.motifIndexExtracted) > 1:
            title = (f'{self.enzymeName}\n Combined {self.datasetTag}\n'
                     f'Top {NSubs} Substrates')
        else:
            title = (f'{self.enzymeName}\n{self.datasetTag}\n'
                     f'Top {NSubs} Substrates')

        # Plot the data
        fig, ax = plt.subplots(figsize=self.figSize)
        bars = plt.bar(xValues, yValues, color=barColor, width=barWidth)
        plt.ylabel(dataType, fontsize=self.labelSizeAxis)
        plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        plt.axhline(y=0, color='black', linewidth=self.lineThickness)
        plt.ylim(yMin, yMax)
        # plt.subplots_adjust(top=0.873, bottom=0.12, left=0.101, right=0.979)

        # Set the edge color
        for bar in bars:
            bar.set_edgecolor('#CC5500')

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks, width=self.lineThickness)
        plt.xticks(rotation=90, ha='center')

        # Set the thickness of the figure border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        fig.tight_layout()
        plt.show()

        # Save the figure
        if self.saveFigures:
            # Define: Save location
            if combinedMotifs:
                figLabel = (f'{self.enzymeName} - Bars - {dataType} - N {NSubs} - '
                            f'Combined {self.datasetTag} - MinCounts {self.minSubCount}.'
                            f'png')
            else:
                figLabel = (f'{self.enzymeName} - Bars - {dataType} - N {NSubs} - '
                            f'{self.datasetTag} - MinCounts {self.minSubCount}.png')
            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                print(f'{yellow}The figure was not saved\n\n'
                      f'File was already found at path:\n'
                      f'     {saveLocation}{resetColor}\n\n')
            else:
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)



    def ESM(self, substrates, subLabel, useSubCounts=True):
        print('=========================== Convert To Numerical: ESM '
              '===========================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n\n'
              f'Collecting up to {red}{self.NSubsPCA:,}{resetColor} substrates\n'
              f'Total unique substrates: {red}{len(substrates):,}{resetColor}')

        # Extract: Datapoints
        iteration = 0
        collectedTotalValues = 0
        evaluateSubs = {}
        for substrate, value in substrates.items():
            evaluateSubs[str(substrate)] = value
            iteration += 1
            collectedTotalValues += value
            if iteration >= self.NSubsPCA:
                break
        sampleSize = len(evaluateSubs)
        print(f'Collected substrates:{red} {sampleSize:,}{resetColor}')
        if isinstance(collectedTotalValues, float):
              print(f'Total Values:{red} {round(collectedTotalValues, 1):,}'
                    f'{resetColor}\n\n')
        else:
            print(f'Total Values:{red} {collectedTotalValues:,}{resetColor}\n\n')

        # Step 1: Convert substrates to ESM model format and generate embeddings
        subs = []
        counts = []
        if useSubCounts:
            for index, (seq, count) in enumerate(evaluateSubs.items()):
                subs.append((f'Sub{index}', seq))
                counts.append(count)
        else:
            for index, seq in enumerate(evaluateSubs.keys()):
                subs.append((f'Sub{index}', seq))


        # Step 2: Load the ESM model and batch converter
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        batch_converter = alphabet.get_batch_converter()


        # Step 3: Convert substrates to ESM model format and generate embeddings
        try:
            batchLabels, batchSubs, batchTokens = batch_converter(subs)
        except Exception as exc:
            print(f'{orange}ERROR: The ESM has failed to evaluate your substrates\n\n'
                  f'Exception:\n{exc}\n\n'
                  f'Suggestion:'
                  f'     Try replacing: {cyan}esm.pretrained.esm2_t36_3B_UR50D()'
                  f'{orange}\n'
                  f'     With: {cyan}esm.pretrained.esm2_t33_650M_UR50D()'
                  f'{resetColor}\n')
            sys.exit(1)

        print(f'Batch Tokens:{greenLight} {batchTokens.shape}{resetColor}\n'
              f'{greenLight}{batchTokens}{resetColor}\n\n')
        slicedTokens = pd.DataFrame(batchTokens[:, 1:-1],
                                    index=batchSubs,
                                    columns=subLabel)
        if useSubCounts:
            slicedTokens['Counts'] = counts
        print(f'Sliced Tokens:\n'
              f'{greenLight}{slicedTokens}{resetColor}\n\n')

        return slicedTokens, batchSubs, sampleSize



    def plotPCA(self, substrates, data, indices, N, combinedMotifs=False):
        print('====================================== PCA '
              '======================================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n')

        # Initialize lists for the clustered substrates
        self.selectedSubstrates = []
        self.selectedDatapoints = []
        rectangles = []

        # Define component labels
        pcaHeaders = []
        for componentNumber in range(1, self.NPCs + 1):
            pcaHeaders.append(f'PC{componentNumber}')
        headerCombinations = list(combinations(pcaHeaders, 2))

        # # Cluster the datapoints
        # Step 1: Apply PCA on the standardized data
        pca = PCA(n_components=self.NPCs) # Adjust the number of components as needed
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        dataPCA = pca.fit_transform(data)
        # loadings = pca.components_.T

        # Step 2: Create a DataFrame for PCA results
        dataPCA = pd.DataFrame(dataPCA, columns=pcaHeaders, index=indices)
        print(f'PCA data:{red} # of components = {self.NPCs}\n'
              f'{greenLight}{dataPCA}{resetColor}\n\n')

        # Step 3: Print explained variance ratio
        varRatio = pca.explained_variance_ratio_ * 100
        print(f'Explained Variance Ratio: '
              f'{red}{" ".join([f"{x:.3f}" for x in varRatio])}{resetColor} %\n\n')


        # Define: Figure title
        if self.filterSubs:
            # Modify dataset tag
            if 'Excl' in self.datasetTag:
                title = (f'\n{self.enzymeName}\n'
                         f'{self.datasetTag.replace('Excl', 'Exclude')}\n'
                         f'{N:,} Unique Substrates')
            else:
                title = (f'\n{self.enzymeName}\n'
                         f'{self.datasetTag}\n'
                         f'{N:,} Unique Substrates')
        else:
            title = (f'\n{self.enzymeName}\n'
                     f'Unfiltered'
                     f'{N:,} Unique Substrates')
        if combinedMotifs  and len(self.motifIndexExtracted) > 1:
            title = title.replace('Motifs', 'Combined Motifs')


        # Plot the data
        for components in headerCombinations:
            fig, ax = plt.subplots(figsize=self.figSize)

            def selectDatapoints(eClick, eRelease):
                # # Function to update selection with a rectangle

                nonlocal ax, rectangles

                # Define x, y coordinates
                x1, y1 = eClick.xdata, eClick.ydata  # Start of the rectangle
                x2, y2 = eRelease.xdata, eRelease.ydata  # End of the rectangle

                # Collect selected datapoints
                selection = []
                selectedSubs = []
                for index, (x, y) in enumerate(zip(dataPCA.loc[:, 'PC1'],
                                                   dataPCA.loc[:, 'PC2'])):
                    if (min(x1, x2) <= x <= max(x1, x2) and
                            min(y1, y2) <= y <= max(y1, y2)):
                        selection.append((x, y))
                        selectedSubs.append(dataPCA.index[index])
                if selection:
                    self.selectedDatapoints.append(selection)
                    self.selectedSubstrates.append(selectedSubs)

                # Draw the boxes
                if self.selectedDatapoints:
                    for index, box in enumerate(self.selectedDatapoints):
                        # Calculate the bounding box for the selected points
                        padding = 0.05
                        xMinBox = min(x for x, y in box) - padding
                        xMaxBox = max(x for x, y in box) + padding
                        yMinBox = min(y for x, y in box) - padding
                        yMaxBox = max(y for x, y in box) + padding

                        # Draw a single rectangle around the bounding box
                        boundingRect = plt.Rectangle((xMinBox, yMinBox),
                                                     width=xMaxBox - xMinBox,
                                                     height=yMaxBox - yMinBox,
                                                     linewidth=2,
                                                     edgecolor='black',
                                                     facecolor='none')
                        ax.add_patch(boundingRect)
                        self.rectangles.append(boundingRect)

                        # Add text only if there are multiple boxes
                        if len(self.selectedDatapoints) > 1:
                            # Calculate the center of the rectangle for text positioning
                            centerX = (xMinBox + xMaxBox) / 2
                            centerY = (yMinBox + yMaxBox) / 2

                            # Number the boxes
                            text = ax.text(centerX, centerY, f'{index + 1}',
                                           horizontalalignment='center',
                                           verticalalignment='center',
                                           fontsize=25,
                                           color='#F79620',
                                           fontweight='bold')
                            text.set_path_effects(
                                [path_effects.Stroke(linewidth=2, foreground='black'),
                                 path_effects.Normal()])
                plt.draw()
            plt.scatter(dataPCA[components[0]], dataPCA[components[1]],
                        c='#CC5500', edgecolor='black')
            plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
            plt.xlabel(f'Principal Component {components[0][-1]} '
                       f'({np.round(varRatio[0], self.roundVal)} %)',
                       fontsize=self.labelSizeAxis)
            plt.ylabel(f'Principal Component {components[1][-1]} '
                       f'({np.round(varRatio[1], self.roundVal)} %)',
                       fontsize=self.labelSizeAxis)
            plt.subplots_adjust(top=0.852, bottom=0.075, left=0.13, right=0.938)


            # Set tick parameters
            ax.tick_params(axis='both', which='major', length=self.tickLength,
                           labelsize=self.labelSizeTicks, width=self.lineThickness)

            # Set the thickness of the figure border
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(self.lineThickness)


            # Create a RectangleSelector
            selector = RectangleSelector(ax,
                                         selectDatapoints,
                                         useblit=True,
                                         minspanx=5,
                                         minspany=5,
                                         spancoords='pixels',
                                         interactive=True)

            # Change rubber band color
            selector.set_props(facecolor='none', edgecolor='green', linewidth=3)

            fig.canvas.mpl_connect('key_press_event', pressKey)
            # fig.tight_layout()
            plt.show()


        # Save the Figure
        if self.saveFigures:
            # Define: Save location
            figLabel = (f'{self.enzymeName} - PCA - {self.datasetTag} - '
                        f'{N} - MinCounts {self.minSubCount}.png')
            if combinedMotifs and len(self.motifIndexExtracted) > 1:
                figLabel = figLabel.replace(self.datasetTag,
                                            f'Combined {self.datasetTag}')
            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                print(f'{yellow}The figure was not saved\n\n'
                      f'File was already found at path:\n'
                      f'     {saveLocation}{resetColor}\n\n')
            else:
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)


        # Create a list of collected substrate dictionaries
        if self.selectedSubstrates:
            collectedSubs = []
            for index, substrateSet in enumerate(self.selectedSubstrates):
                print(f'PCA Cluster: {greenLightB}{index + 1}{resetColor}')
                iteration = 0
                collectionSet = {}
                for substrate in substrateSet:
                    collectionSet[substrate] = substrates[substrate]

                # Sort collected substrates and add to the list
                collectionSet = dict(sorted(collectionSet.items(),
                                            key=lambda x: x[1], reverse=True))
                collectedSubs.append(collectionSet)

                # Print: Collected substrates
                for substrate, score in collectionSet.items():
                    if isinstance(score, float):
                        print(f'     {blue}{substrate}{resetColor}, '
                              f'Counts: {red}{round(score, 1):,}{resetColor}')
                    else:
                        print(f'     {blue}{substrate}{resetColor}, '
                              f'Counts: {red}{score:,}{resetColor}')
                    iteration += 1
                    if iteration >= self.printNumber:
                        print('\n')
                        break

            return collectedSubs
        else:
            return None



    def KLDivergence(self, P, Q, printProb, scaler):
        print('================================= KL Divergence '
              '=================================')
        P.columns = Q.columns
        if printProb:
            print(f'Baseline Probability Distribution:\n{Q}\n\n\n'
                  f'Probability Distribution:\n{P}\n\n')

        divergence = pd.DataFrame(0, columns=Q.columns,
                                  index=[self.datasetTag], dtype=float)
        divergenceMatrix = pd.DataFrame(0, columns=Q.columns,
                                        index=Q.index, dtype=float)

        for position in Q.columns:
            p = P.loc[:, position]
            q = Q.loc[:, position]
            divergence.loc[self.datasetTag, position] = (
                np.sum(np.where(p != 0, p * np.log2(p / q), 0)))

            for residue in Q.index:
                initial = Q.loc[residue, position]
                final = P.loc[residue, position]
                if initial == 0 or final == 0:
                    divergenceMatrix.loc[residue, position] = 0
                else:
                    divergenceMatrix.loc[residue, position] = (final *
                                                               np.log2(final / initial))

        # Scale the values
        if scaler is not None:
            for position in Q.columns:
                divergenceMatrix.loc[:, position] = (divergenceMatrix.loc[:, position] *
                                                     scaler.loc[position, 'ΔS'])

        print(f'{greyDark}KL Divergence:'
              f'{pink} Fixed Final Sort - {self.datasetTag}{resetColor}\n'
              f'{divergence}\n\n\n{greyDark}Divergency Matrix:'
              f'{pink} Fixed Final Sort - {self.datasetTag}'
              f'{resetColor}\n{divergenceMatrix.round(4)}\n\n')

        return divergenceMatrix, divergence



    def optimalWord(self, matrix, matrixType, maxResidues, dropPos,
                    printOptimalAA, normalizeValues):
        print('========================= Synthesize Optimal Substrates '
              '=========================')

        # Determine the OS
        combinations = 1
        optimalAA = []
        substratesOS = {}

        # Drop irrelevant positions
        if dropPos:
            dropColumns = []
            for indexDropPosition in dropPos:
                dropColumns.append(matrix.columns[indexDropPosition])
            # print(f'Dropping positions: {dropColumns}\n')
            matrix = matrix.drop(columns=dropColumns)
        # print(f'Values used to synthesize optimal substrates:\n{matrix}\n\n')



        for indexColumn, column in enumerate(matrix.columns):
            # Find the best residues at this position
            optimalAAPos = matrix[column].nlargest(maxResidues)

            # Filter the data
            for rank, (AA, score) in (enumerate(
                    zip(optimalAAPos.index, optimalAAPos.values), start=1)):
                if score <= 0:
                    optimalAAPos = optimalAAPos.drop(index=AA)
            optimalAA.append(optimalAAPos)


        if printOptimalAA:
            print(f'Optimal Residues: {purple}{self.enzymeName} - '
                  f'Fixed {self.datasetTag}{resetColor}')
            for index, data in enumerate(optimalAA, start=1):
                # Determine the number of variable residues at this position
                numberAA = len(data)
                combinations *= numberAA

                # Define substrate position
                positionSub = self.xAxisLabels[index-1]
                print(f'Position: {purple}{positionSub}{resetColor}')

                for AA, datapoint in data.items():
                    print(f'     {AA}:{red} {datapoint:.6f}{resetColor}')
                print('\n')
        else:
            for index, data in enumerate(optimalAA, start=1):
                # Determine the number of variable residues at this position
                combinations *= len(data)

        print(f'Possible Substrate Combinations:{pink} {combinations:,}{resetColor}\n')

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

        # Create additional substrates
        for indexColumn, column in enumerate(matrix.columns):

            # Collect new substrates to add after the iteration
            newSubstratesList = []

            for substrate, ESMax in list(substratesOS.items()):
                AAOS = substrate[indexColumn]
                scoreSubstrate = optimalAA[indexColumn][AAOS]

                # Access the correct filtered data for the column
                optimalAAPos = optimalAA[indexColumn]
                # print(f'optimalAA:\n{optimalAA}\n\n'
                #       f'     optimalAAPos:\n{optimalAAPos}\n\n')

                for AA, score in optimalAAPos.items():
                    # print(f'AA: {AA}\nScore: {score}\n\n')
                    if AA != AAOS:

                        # Replace AAOS with AA
                        newSubstrate = (substrate[:indexColumn] + AA +
                                        substrate[indexColumn + 1:])
                        newES = ESMax + (score - scoreSubstrate)
                        # print(f'{greyDark}New Substrate{resetColor}:'
                        #       f'{greenLightB} {newSubstrate}{resetColor}, '
                        #       f'ES:{red} {newES}{resetColor}\n'
                        #       f'     Residue Score New:{red} {score}{resetColor}\n'
                        #       f'     Residue Score Old:{red} {scoreSubstrate}'
                        #       f'{resetColor}\n\n')

                        # Collect new substrate and ES to add later
                        newSubstratesList.append((newSubstrate, newES))
            # Update substratesOS with new substrates after the iteration
            for newSubstrate, newES in newSubstratesList:
                substratesOS[newSubstrate] = newES

        substratesOS = dict(sorted(substratesOS.items(),
                                   key=lambda x: x[1], reverse=True))
        if normalizeValues:
            _, topScore = substratesOS.popitem(last=False) # Top score
            print(f'Top Score:{red} {topScore}{resetColor}\n')

            # Normalize the values
            substratesOS = {key: value / topScore for key, value in substratesOS}
            # substratesOS = sorted(substratesOS.items(), key=lambda x: x[1], reverse=True)

        print(f'Top {self.printNumber} Optimal Substrates:{greyDark} '
              f'{matrixType}{resetColor}')
        iteration = 0
        for substrate, ES in substratesOS.items():
            print(f'     Substrate:{red} {substrate}{resetColor}, '
                  f'ES:{red} {ES:.6f}{resetColor}')
            iteration += 1
            if iteration == self.printNumber:
                break

        print(f'\nNumber of synthesized substrates:'
              f'{pink} {len(substratesOS):,}{resetColor}\n\n')

        return substratesOS



    def substrateEnrichment(self, initialSubs, finalSubs, NSubs, saveData):
        print('========================= Evaluate Substrate Enrichment '
              '=========================')
        if self.datasetTag == None:
            datasetType = 'NNS'
        else:
            datasetType = f'{self.datasetTag}'

        # Define headers
        headersInitial = ['Initial Subs', 'Counts']
        headersFinal = ['Final Subs', 'Counts']
        headersEnriched = ['Enriched Subs', 'log₂(probFinal / probInitial)']
        # headerWidth = {1: 14.6,
        #                4: 14.6,
        #                7: 14.6,
        #                8: 24} # Adjust column width at these indices for an excel sheet


        iteration = 0
        totalSubstratesInitial = 0
        totalUniqueSubstratesInitial = len(initialSubs)
        totalsubstrates = 0
        totalUniquesubstrates = len(finalSubs)

        # Evaluate the substrates
        if self.datasetTag == None:
            # Process: initial sort
            print(f'Ranked Substrates: {purple}Initial Sort{resetColor} -'
                  f'{red} NNS{resetColor}')
            if totalUniqueSubstratesInitial >= self.printNumber:
                for substrate, count in initialSubs.items():
                    if iteration <= self.printNumber:
                        print(f'     {pink}{substrate}{resetColor}, '
                              f'Counts: {red}{count:,}{resetColor}')
                        iteration += 1
                        totalSubstratesInitial += count
                    else:
                        totalSubstratesInitial += count
            else:
                print(f'{orange}The number of unique substrates '
                      f'({red}{totalUniqueSubstratesInitial}'
                      f'{orange}) is less than the number you requested to be see '
                      f'({red}{self.printNumber}{orange}){resetColor}')
                for substrate, count in initialSubs.items():
                    print(f'     {pink}{substrate}{resetColor}, Counts: {red}{count:,}'
                          f'{resetColor}')
                    totalSubstratesInitial += count
            iteration = 0
            # Print: Dataset totals
            print(f'\n     Total substrates {purple}Initial Sort{resetColor}:'
                  f'{red} {totalSubstratesInitial:,}{resetColor}\n'
                  f'     Unique substrates {purple}Initial Sort{resetColor}:'
                  f'{red} {totalUniqueSubstratesInitial:,}{resetColor}\n\n')


            # Process: final sort
            print(f'Ranked Substrates: {purple}Final Sort{resetColor} -'
                  f'{red} NNS{resetColor}')
            if totalUniquesubstrates >= self.printNumber:
                for substrate, count in finalSubs.items():
                    if iteration <= self.printNumber:
                        print(f'     {pink}{substrate}{resetColor}, '
                              f'Counts: {red}{count:,}{resetColor}')
                        iteration += 1
                        totalsubstrates += count
                    else:
                        totalsubstrates += count
            else:
                print(f'{orange}The number of unique substrates '
                      f'({red}{totalUniquesubstrates}'
                      f'{orange}) is less than the number you requested to be see '
                      f'({red}{self.printNumber}{orange}){resetColor}\n')
                for substrate, count in finalSubs.items():
                    print(f'     {pink}{substrate}{resetColor}, '
                          f'Counts: {red}{count:,}{resetColor}')
                    totalsubstrates += count
            iteration = 0
            # Print: Dataset totals
            print(f'\n     Total substrates {purple}Final Sort{resetColor}:'
                  f'{red} {totalsubstrates:,}{resetColor}\n'
                  f'     Unique substrates {purple}Final Sort{resetColor}:'
                  f'{red} {totalUniquesubstrates:,}{resetColor}\n\n')
        else:
            fixedSort = True

            # Print: Initial sort
            print(f'Ranked Substrates: {purple}Initial Sort{resetColor}{resetColor}')
            if totalUniqueSubstratesInitial >= self.printNumber:
                for substrate, count in initialSubs.items():
                    if iteration < self.printNumber:
                        print(f'     {pink}{substrate}{resetColor}, '
                              f'Counts: {red}{count:,}{resetColor}')
                        iteration += 1
                        totalSubstratesInitial += count
                    else:
                        totalSubstratesInitial += count
            else:
                print(f'{orange}The number of unique substrates '
                      f'({red}{totalUniqueSubstratesInitial}'
                      f'{orange}) is less than the number you requested to be see '
                      f'({red}{self.printNumber}{orange}){resetColor}')
                for substrate, count in initialSubs.items():
                    print(f'     {pink}{substrate}{resetColor}, '
                          f'Counts: {red}{count:,}{resetColor}')
                    totalSubstratesInitial += count
            print(f'\n     Total substrates {purple}Initial Sort{resetColor}:'
                  f'{red} {totalSubstratesInitial:,}{resetColor}\n'
                  f'     Unique substrates {purple}Initial Sort{resetColor}:'
                  f'{red} {totalUniqueSubstratesInitial:,}{resetColor}\n\n')
            iteration = 0


            # Process: Final sort
            print(f'Ranked Substrates: {purple}Final Sort{resetColor} -'
                  f'{red} Fixed {self.datasetTag}{resetColor}')
            if totalUniquesubstrates >= self.printNumber:
                for substrate, count in finalSubs.items():
                    if iteration < self.printNumber:
                        print(f'     {pink}{substrate}{resetColor}, '
                              f'Counts: {red}{count:,}{resetColor}')
                        iteration += 1
                        totalsubstrates += count
                    else:
                        totalsubstrates += count
            else:
                print(f'{orange}The number of unique substrates '
                      f'({red}{totalUniquesubstrates}'
                      f'{orange}) is less than the number you requested to be see '
                      f'({red}{self.printNumber}{orange}){resetColor}')
                for substrate, count in finalSubs.items():
                    print(f'     {pink}{substrate}{resetColor}, '
                              f'Counts: {red}{count:,}{resetColor}')
                    totalsubstrates += count
            print(f'\n     Total substrates {purple}Final Sort{resetColor}:'
                  f'{red} {totalsubstrates:,}{resetColor}\n'
                  f'     Unique substrates {purple}Final Sort{resetColor}:'
                  f'{red} {totalUniquesubstrates:,}{resetColor}\n\n')
            iteration = 0


        # Calculate: Substrate enrichment
        enrichedSubs = {}
        setMinCountFinal = False
        if setMinCountFinal:
            print(f'Mininum Substrate Count:{red} {self.minSubCount}{resetColor}')
            for substrate, count in finalSubs.items():
                if count < self.minSubCount:
                    continue

                if substrate in initialSubs.keys():
                    countInitial = initialSubs[substrate]
                else:
                    countInitial = 1
                probFinal = count / totalsubstrates
                probInitial = countInitial / totalSubstratesInitial
                enrichment = np.log2(probFinal/probInitial)
                enrichedSubs[substrate] = enrichment
        else:
            for substrate, count in finalSubs.items():
                if substrate in initialSubs.keys():
                    countInitial = initialSubs[substrate]
                else:
                    countInitial = 1
                probFinal = count / totalsubstrates
                probInitial = countInitial / totalSubstratesInitial
                enrichment = np.log2(probFinal/probInitial)
                enrichedSubs[substrate] = enrichment
        # sys.exit(1)

        # Sort enrichment dictionary
        enrichedSubs = dict(sorted(enrichedSubs.items(),
                                   key=lambda x: x[1], reverse=True))

        # Print: top enriched substrates
        print(f'{purple}Enriched substrates{resetColor}:')
        for substrate, score in enrichedSubs.items():
            if iteration < self.printNumber:
                print(f'     {magenta}{substrate}{resetColor}, '
                      f'ES: {red}{score:.3f}{resetColor}')
                iteration += 1
            else:
                break
        print('\n')


        if saveData:
            # Define file path
            filePathCSV = os.path.join(
                self.pathData,
                f'{self.enzymeName} - Enriched Subs - {self.datasetTag} - '
                f'MinCounts {self.minSubCount}.csv')

            # Convert dictionaries to a data frame and save as an Excel file
            clipDataset = False
            if totalSubstratesInitial > NSubs:
                clipDataset = True

                # Did you ask for too many substrates?
                if NSubs == 10**6:
                    NSubs -= 1
                elif NSubs > 10**6:
                    print(f'     The list of substrates in the  {purple}Initial Sort'
                          f'{resetColor} that you attempted to save '
                          f'({red}N = {NSubs:,}{resetColor}) '
                          f'is to large to save in an Excel file.\n'
                          f'          Extracting the first{red} {10**6:,}{resetColor} '
                          f'substrates and discarding the rest.\n')
                    NSubs = 10**6

                initialSubsDF = pd.DataFrame.from_dict(dict(
                    list(initialSubs.items())[:NSubs]).items())
                initialSubsDF.columns = headersInitial
            else:
                initialSubsDF = pd.DataFrame.from_dict(initialSubs.items())
                initialSubsDF.columns = headersInitial
            if totalsubstrates > NSubs:
                clipDataset = True

                # Did you ask for too many substrates?
                if NSubs > 10**6:
                    print(f'     The list of substrates in the  {purple}Final Sort'
                          f'{resetColor} that you attempted to save '
                          f'({red}N = {NSubs:,}{resetColor}) '
                          f'is to large to save in an Excel file.\n'
                          f'          Extracting the first{red} {10**6:,}{resetColor} '
                          f'substrates and discarding the rest.\n')
                    NSubs = 10**6

                finalSubsDF = pd.DataFrame.from_dict(dict(
                    list(finalSubs.items())[:NSubs]).items())
                finalSubsDF.columns = headersFinal

                enrichedSubsDF = pd.DataFrame.from_dict(dict(
                    list(enrichedSubs.items())[:NSubs]).items())
                enrichedSubsDF.columns = headersEnriched
            else:
                finalSubsDF = pd.DataFrame.from_dict(finalSubs.items())
                finalSubsDF.columns = headersFinal

                enrichedSubsDF = pd.DataFrame.from_dict(enrichedSubs.items())
                enrichedSubsDF.columns = headersEnriched
            if clipDataset:
                print(f'Saving dataset with a maximum of{red} {NSubs:,}'
                      f'{resetColor} substrates.\n\n'
                      f'Any data returned from this function that will be used for further '
                      f'analysis\nwill include the complete dataset\n\n')

            # Print: Sample sizes
            print(f'{greyDark}Dataset Size{resetColor}:\n'
                  f'     Initial Substrates:{red} {len(initialSubsDF):,}{resetColor}\n'
                  f'     Final Substrates:{red} {len(finalSubsDF):,}{resetColor}\n'
                  f'     Enriched Substrates:{red} {len(enrichedSubsDF):,}{resetColor}\n')

            if os.path.exists(filePathCSV):
                print(f'{orange}The{red} {datasetType}{orange} dataset at was '
                      f'found at the path:'
                      f'\n     {resetColor}{filePathCSV}\n\n'
                      f'{orange}The file was not overwritten{resetColor}\n\n')
            else:
                print(f'Saving the{red} {datasetType}{resetColor} dataset at the path:'
                      f'\n     {filePathCSV}{resetColor}\n\n')

                # Combine the data frames
                if datasetType == 'NNS':
                    enrichedSubsCSV = pd.concat([initialSubsDF,
                                                 finalSubsDF,
                                                 enrichedSubsDF],
                                                axis=1)
                    enrichedSubsCSV.to_csv(filePathCSV, index=True)
                else:
                    enrichedSubsCSV = pd.concat([finalSubsDF, enrichedSubsDF],
                                                axis=1)
                    enrichedSubsCSV.to_csv(filePathCSV, index=True)

        return enrichedSubs


    # ====================================================================================


    def plotSubstratePopulations(self, substrates, clusterIndex, numClusters,
                                 datasetTag, saveTag):
        print('=============================== Plot PCA Clusters '
              '===============================')
        print(f'Dataset: {purple}{datasetTag}{resetColor}\n'
              f'Save Tag: {purple}{saveTag}{resetColor}\n\n'
              f'Total Clusters:{red} {numClusters}{resetColor}\n'
              f'     Cluster Number:{red} {clusterIndex + 1}{resetColor}\n\n')

        # Define figure titles
        if numClusters == 1:
            figureTitleEM = (f'\n{inTitleEnrichmentMap}: PCA Population\n'
                             f'{self.datasetTag}')
            figureTitleMotif = (f'{inTitleMotif}: PCA Population\n'
                                f'{self.datasetTag}')
            figureTitleWordCloud = (f'{inTitleEnrichmentMap}: '
                                    f'PCA Population\n{self.datasetTag}')
            datasetTag = f'PCA Pop - {self.datasetTag}'
        else:
            figureTitleEM = (
                f'\n{inTitleEnrichmentMap}: PCA Population {clusterIndex + 1}\n'
                f'{self.datasetTag}')
            figureTitleMotif = (f'{inTitleMotif}: PCA Population {clusterIndex + 1}\n'
                                f'{self.datasetTag}')
            figureTitleWordCloud = (f'{inTitleEnrichmentMap}: '
                                    f'PCA Population {clusterIndex + 1}\n{self.datasetTag}')
            datasetTag = f'PCA Pop {clusterIndex + 1} - {self.datasetTag}'

        # Count fixed substrates
        countFullSubstrate = False
        if countFullSubstrate:
            countsFinal, countsTotalFinal = self.countResidues(substrates=substrates,
                                                              datasetType='')
        else:
            countsFinal, countsTotalFinal = self.countResidues(substrates=substrates,
                                                              datasetType='')
        self.sampleSizeUpdate(NSubs=countsTotalFinal, sortType='Final Sort',
                             datasetTag=self.datasetTag)

        # Adjust the zero counts at nonfixed positions
        countsFinalAdjusted = countsFinal.copy()
        if inAdjustZeroCounts:
            for indexColumn in countsFinalAdjusted.columns:
                for AA in countsFinalAdjusted.index:
                    if countsFinalAdjusted.loc[AA, indexColumn] == 0:
                        countsFinalAdjusted.loc[AA, indexColumn] = 1
            print(f'Adjusted Final Counts:{pink} {self.enzymeName}\n'
                  f'{red}{countsFinalAdjusted}{resetColor}\n\n')

        # Calculate: RF
        probFinal = self.calculateAAProb(counts=countsFinal, N=countsTotalFinal,
                                  fileType='Final Sort')
        probFinalAdjusted = self.calculateAAProb(counts=countsFinalAdjusted, N=countsTotalFinal,
                                          fileType='Final Sort')

        if inPlotEntropyPCAPopulations:
            # Plot: Positional entropy
            self.plotEntropy(entropy=self.entropy)

        # Calculate: Enrichment scores
        fixedFramePopES = self.enrichmentMatrix(initialSortRF=probInitialAvg,
                                                  finalSortRF=probFinal)
        fixedFramePopESAdjusted = self.enrichmentMatrix(initialSortRF=probInitialAvg,
                                                          finalSortRF=probFinalAdjusted)

        # Calculate: Enrichment scores scaled
        fixedFramePCAESScaled = pd.DataFrame(0.0, index=fixedFramePopES.index,
                                             columns=fixedFramePopES.columns)
        fixedFramePCAESScaledAdjusted = pd.DataFrame(0.0, index=fixedFramePopES.index,
                                                     columns=fixedFramePopES.columns)

        # Scale enrichment scores with Shannon Entropy
        for positon in fixedFramePopES.columns:
            fixedFramePCAESScaled.loc[:, positon] = (fixedFramePopES.loc[:, positon] *
                                                     self.entropy.loc[
                                                         positon, 'ΔS'])
            fixedFramePCAESScaledAdjusted.loc[:, positon] = (
                    fixedFramePopESAdjusted.loc[:, positon] *
                    self.entropy.loc[positon, 'ΔS'])
        print(f'Motif:{greenLight} Scaled{resetColor}\n{fixedFramePCAESScaled}\n\n')
        yMax = max(fixedFramePCAESScaled[fixedFramePopES > 0].sum())
        yMin = min(fixedFramePCAESScaled[fixedFramePopES < 0].sum())

        # # # Plot data
        # # Plot: Enrichment Map
        # self.plotEnrichmentScores(scores=fixedFramePopESAdjusted, dataType='Enrichment',
        #                           motifFilter=False, duplicateFigure=False,
        #                           saveTag=datasetTag)
        #
        # # Plot: Enrichment Map Scaled
        # self.plotEnrichmentScores(scores=fixedFramePCAESScaledAdjusted,
        #                           dataType='Scaled Enrichment')
        #
        # # Plot: Sequence Motif
        # self.plotMotif(data=fixedFramePCAESScaled.copy(), dataType='Scaled Enrichment',
        #                bigLettersOnTop=inBigLettersOnTop, yMax=yMax, yMin=yMin,
        #                showYTicks=False, addHorizontalLines=inAddHorizontalLines,
        #                motifFilter=False, duplicateFigure=False, saveTag=datasetTag)

        # Plot: Work cloud
        self.plotWordCloud(substrates=substrates)


    # ====================================================================================


    def suffixTree(self, substrates, N, entropySubFrame, indexSubFrame, entropyMin,
                   datasetTag, dataType):
        print('================================== Suffix Tree '
              '==================================')
        if datasetTag is None:
            print(f'Dataset: {purple}{self.enzymeName} - Unfixed{resetColor}\n')
        else:
            print(f'Dataset: {purple}{self.enzymeName} {dataType} {datasetTag}'
                  f'{resetColor}\n')


        from Trie import Trie


        trie = Trie() # Initialize Trie
        motifs = {}
        indexStart = min(indexSubFrame)
        indexEnd = max(indexSubFrame)

        # Print: Substrates
        iteration = 0
        substrates = dict(sorted(substrates.items(), key=lambda item: item[1],
                                 reverse=True))
        for substrate, count in substrates.items():
            iteration += 1
            print(f'     {pink}{substrate}{resetColor}, '
                  f'Counts: {red}{count:,}{resetColor}')
            if iteration >= self.printNumber:
                break
        print('\n')

        # Find motif positions based on the entropy threshold
        indexPos = []
        for index in entropySubFrame.index:
            posEntropy = entropySubFrame.loc[index, 'ΔS']
            if posEntropy >= entropyMin:
                indexPos.append(int(index.replace('R', '')) - 1)
        print(f'Index Pos: {indexPos}')

        motifTrie = {}
        countsMotif = 0
        def addMotif(motif, count):
            # Extract important AAs from the motif
            motif = ''.join(motif[index] for index in indexPos)

            # Add motif to the trie
            if motif in motifTrie.keys():
                motifTrie[motif] += count
            else:
                motifTrie[motif] = count
                trie.insert(motif)


        # Extract the motifs
        motifCount = 0
        for substrate, count in substrates.items():
            motif = substrate[indexStart:indexEnd + 1]
            if motif in motifs:
                motifs[motif] += count
            else:
                motifs[motif] = count
                motifCount += 1

            # Add the motif to the tree
            addMotif(motif, count)
            countsMotif = len(motifTrie.keys())
            if countsMotif >= N:
                break
        motifs = dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))
        motifTrie = dict(sorted(motifTrie.items(), key=lambda item: item[1],
                                reverse=True))

        # Print: Motifs
        print(f'Extracted Motifs:')
        for index, (motif, count) in enumerate(motifs.items()):
            print(f'{index+1}:{blue} {motif}{resetColor} '
                  f'Count:{red} {count:,}{resetColor}')
        print('\n')

        # Print: Trie
        print(f'Extracted Trie:')
        for index, (seq, count) in enumerate(motifTrie.items()):
            print(f'{index + 1}:{pink} {seq}{resetColor} '
                  f'Count:{red} {count:,}{resetColor}')
        print('\n')

        # Calculate: RF
        motifTable = self.evaluateSubtrees(trie=trie, motifTrie=motifTrie)

        # Plot the Trie
        self.plotTrie(trie=trie, motifTable=motifTable, countsMotif=countsMotif,
                      datasetTag=datasetTag)



    def plotCounts(self, countedData, totalCounts, fileName):
        # Remove commas from string values and convert to float
        countedData = countedData.applymap(lambda x:
                                           float(x.replace(',', ''))
                                           if isinstance(x, str) else x)

        # Create heatmap
        cMapCustom = self.createCustomColorMap(colorType='Counts')

        # Convert the counts to a data frame for Seaborn heatmap
        if self.residueLabelType == 0:
            countedData.index = [residue[0] for residue in self.residues]
        elif self.residueLabelType == 1:
            countedData.index = [residue[1] for residue in self.residues]
        elif self.residueLabelType == 2:
            countedData.index = [residue[2] for residue in self.residues]

        # Set figure title
        title = f'\n\n{self.enzymeName}\n{fileName}\nN={totalCounts:,}'


        # Plot the heatmap with numbers centered inside the squares
        fig, ax = plt.subplots(figsize=self.figSize)
        heatmap = sns.heatmap(countedData, annot=True, fmt=',d', cmap=cMapCustom,
                              cbar=True, linewidths=self.lineThickness-1,
                              linecolor='black', square=False, center=None,
                              annot_kws={'fontweight': 'bold'})
        ax.set_xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        ax.set_ylabel('Residue', fontsize=self.labelSizeAxis)
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        figBorders = [0.852, 0.075, 0.117, 1]
        plt.subplots_adjust(top=figBorders[0], bottom=figBorders[1],
                            left=figBorders[2], right=figBorders[3])

        # Set the thickness of the figure border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks, width=self.lineThickness)
        ax.tick_params(axis='y', labelrotation=0)

        # Set x-ticks
        xTicks = np.arange(len(countedData.columns)) + 0.5
        ax.set_xticks(xTicks)
        ax.set_xticklabels(countedData.columns)

        # Set y-ticks
        yTicks = np.arange(len(countedData.index)) + 0.5
        ax.set_yticks(yTicks)
        ax.set_yticklabels(countedData.index)


        for _, spine in ax.spines.items():
            spine.set_visible(True)

        # Modify the colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(axis='y', which='major', labelsize=self.labelSizeTicks,
                            length=self.tickLength, width=self.lineThickness)
        cbar.outline.set_linewidth(self.lineThickness)
        cbar.outline.set_edgecolor('black')

        fig.canvas.mpl_connect('key_press_event', pressKey)
        plt.show()



    def calculateEntropy(self, probability, fixFullFrame=None, combinedMotifs=False,
                         releasedCounts=False):
        print('============================== Calculate: Entropy '
              '===============================')
        self.entropy = pd.DataFrame(0.0, index=probability.columns, columns=['ΔS'])
        self.entropyMax = np.log2(len(probability.index))
        for indexColumn in probability.columns:
            S = 0
            for indexRow, probRatio in probability.iterrows():
                prob = probRatio[indexColumn]
                if prob == 0:
                    continue
                else:
                    S += -prob * np.log2(prob)
            self.entropy.loc[indexColumn, 'ΔS'] = self.entropyMax - S
        print(f'{self.entropy}\n\nMax Entropy: {self.entropyMax.round(6)}\n\n')

        # Identify motif frame
        if fixFullFrame is not None:
            self.identifyMotif(fixFullFrame=fixFullFrame)

        if self.plotFigEntropy:
            self.plotEntropy(entropy=self.entropy, combinedMotifs=combinedMotifs,
                             releasedCounts=releasedCounts)

        return self.entropy



    def plotEntropy(self, entropy, combinedMotifs=False, releasedCounts=False):
        if self.filterSubs:
            title = f'\n\n{self.enzymeName}\n{self.datasetTag}'
        else:
            title = f'\n\n\n{self.enzymeName}'
        if combinedMotifs and len(self.motifIndexExtracted) > 1:
            title = title.replace(self.datasetTag, f'Combined {self.datasetTag}')
        if releasedCounts:
            title = title.replace(self.datasetTag, f'Released {self.datasetTag}')


        # Figure parameters
        yMax = self.entropyMax + 0.2

        # Map entropy values to colors using the colormap
        colors = [(0, 'navy'),
                  (0.3/self.entropyMax, 'navy'),
                  (0.7/self.entropyMax, 'dodgerblue'),
                  (0.97/self.entropyMax, 'white'),
                  (0.98/self.entropyMax, 'white'),
                  (1.0/self.entropyMax, 'white'),
                  (1.65/self.entropyMax, 'red'),
                  (3/self.entropyMax, 'firebrick'),
                  (1, 'darkred')]
        colorBar = LinearSegmentedColormap.from_list('custom_colormap', colors)


        # Map entropy values to colors using the colormap
        normalize = Normalize(vmin=0, vmax=yMax) # Normalize the entropy values
        cMap = [colorBar(normalize(value)) for value in entropy['ΔS'].astype(float)]

        # Plotting the entropy values as a bar graph
        fig, ax = plt.subplots(figsize=self.figSize)
        plt.bar(entropy.index, entropy['ΔS'], color=cMap,
                edgecolor='black', linewidth=self.lineThickness, width=0.8)
        plt.xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        plt.ylabel('ΔS', fontsize=self.labelSizeAxis, rotation=0, labelpad=15)
        plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        plt.subplots_adjust(top=0.852, bottom=0.075, left=0.12, right=0.9)
        # self.figSizeMini
        # plt.subplots_adjust(top=0.898, bottom=0.098, left=0.121, right=0.917)


        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks)

        # Set x-ticks
        xTicks = np.arange(0, len(entropy.iloc[:, 0]), 1)
        ax.set_xticks(xTicks)
        ax.set_xticklabels(entropy.index, rotation=0, ha='center')
        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness) # Set tick width

        # Set y-ticks
        yTicks = range(0, 5)
        yTickLabels = [f'{tick:.0f}' if tick != 0 else f'{int(tick)}' for tick in yTicks]
        ax.set_yticks(yTicks)
        ax.set_yticklabels(yTickLabels)
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness) # Set tick width

        # Set the edge thickness
        for spine in ax.spines.values():
            spine.set_linewidth(self.lineThickness)

        # Set axis limits
        ax.set_ylim(0, yMax)

        # Add color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalize, cmap=colorBar), cax=cax)
        cbar.ax.tick_params(axis='y', which='major', labelsize=self.labelSizeTicks,
                            length=self.tickLength, width=self.lineThickness)
        for tick in cbar.ax.yaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness) # Set tick width
        cbar.outline.set_linewidth(self.lineThickness)

        fig.canvas.mpl_connect('key_press_event', pressKey)
        if self.setFigureTimer:
            plt.ion()
            plt.show()
            plt.pause(self.figureTimerDuration)
            plt.close(fig)
            plt.ioff()
        else:
            plt.show()


        # Save the figure
        if self.saveFigures:
            # Define: Save location
            if self.filterSubs:
                figLabel = (f'{self.enzymeName} - Entropy - '
                            f'{self.datasetTag} - {len(xTicks)} AA - '
                            f'MinCounts {self.minSubCount}.png')
            else:
                figLabel = (f'{self.enzymeName} - Positional Entropy - '
                            f'Unfiltered - {len(xTicks)} AA - '
                            f'MinCounts {self.minSubCount}.png')
            if combinedMotifs and len(self.motifIndexExtracted) > 1:
                figLabel = figLabel.replace(
                    self.datasetTag, f'Combined {self.datasetTag}')
            if releasedCounts:
                figLabel = figLabel.replace(self.datasetTag,
                                            f'Released {self.datasetTag}')
            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                print(f'{yellow}The figure was not saved\n\n'
                      f'File was already found at path:\n'
                      f'     {saveLocation}{resetColor}\n\n')
            else:
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)



    def plotLibraryProbDist(self, probInitial, probFinal, codonType, datasetTag,
                            skipInitial=False):
        # Inspect data
        if probInitial is None and probFinal is None:
            print(f'{orange}ERROR: both of the inputs for probInitial and '
                  f'probFinal cannot be None.{resetColor}\n')
            sys.exit(1)

        # Initialize parameters
        plotInitial, plotFinal = False, False
        maxInitial = 0
        maxInitialAdj = 0
        maxFinal = 0
        maxFinalAdj = 0

        # Determine yMax
        if probInitial is not None:
            plotInitial = True
            numPos = probInitial.shape[1]
            numAA = probInitial.shape[0]
            maxInitial = probInitial.values.max()
            maxInitialAdj = np.floor(maxInitial * 10) / 10
        if probFinal is not None:
            plotFinal = True
            numPos = probFinal.shape[1]
            numAA = probFinal.shape[0]
            maxFinal = probFinal.values.max()
            maxFinalAdj= np.floor(maxFinal * 10) / 10
        if maxFinalAdj > maxInitialAdj:
            yMax = maxFinalAdj
            maxY = maxFinal
        else:
            yMax = maxInitialAdj
            maxY = maxInitial
        if yMax > 0.6:
            tickStepSize = 0.1
        else:
            tickStepSize = 0.05
        if yMax != 1.0:
            if yMax < maxY:
                while yMax < maxY:
                    yMax += tickStepSize
        yMax = round(yMax, 1)


        def plotFig(probability, sortType):
            print('======================= Plot: AA Probability Distribution '
                  '=======================')
            if codonType == datasetTag:
                print(f'Plotting Probability Distribution:'
                      f' {purple}{codonType} codon{resetColor}')
            else:
                print(f'Plotting Probability Distribution:'
                      f' {purple}{self.enzymeName} {sortType}{resetColor}')
            print(f'{probability}\n')

            if sortType == 'Initial Sort':
                title = f'Unsorted {self.enzymeName} Library'
            else:
                if datasetTag is None or datasetTag == 'Unfiltered':
                    title = f'Sorted {self.enzymeName} Library'
                elif codonType == datasetTag:
                    title = f'{codonType} Codon'
                else:
                    title = f'Sorted {self.enzymeName} Library - {datasetTag}'

            fig, ax = plt.subplots(figsize=self.figSize)
            plt.ylabel('Probability', fontsize=self.labelSizeAxis)
            plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
            plt.subplots_adjust(top=0.926, bottom=0.068, left=0.102, right=0.979)

            # Set tick parameters
            ax.tick_params(axis='both', which='major', length=self.tickLength,
                           width=self.lineThickness)

            # Set x-ticks
            if codonType == datasetTag:
                widthBar = 9
            else:
                widthBar = 2
            spacing = widthBar * numPos
            widthCluster = spacing + 5
            indices = np.arange(numAA) * widthCluster
            midPoint = (numPos - 1) / 2 * widthBar
            xTicks = indices + midPoint
            ax.set_xticks(xTicks)
            ax.set_xticklabels(probability.index, rotation=0, ha='center',
                               fontsize=self.labelSizeTicks)

            # Set y-ticks
            yTicks = np.arange(0, yMax + tickStepSize, tickStepSize)
            yTickLabels = [f'{tick:.0f}' if tick == 0 or tick == 1
                           else f'{tick:.2f}' for tick in yTicks]
            ax.set_yticks(yTicks)
            ax.set_yticklabels(yTickLabels, fontsize=self.labelSizeTicks)
            for tick in ax.yaxis.get_major_ticks():
                tick.tick1line.set_markeredgewidth(self.lineThickness)

            # Set the edge color
            for index, AA in enumerate(probability.index):
                xPos = indices[index] + np.arange(numPos) * widthBar
                if AA == 'F' or AA == 'W' or AA == 'Y': # AA == 'N' or AA == 'Q' or
                    ax.bar(xPos, probability.loc[AA], widthBar, label=AA,
                           color=self.colorsAA[AA], edgecolor='dimgray')
                else:
                    ax.bar(xPos, probability.loc[AA], widthBar, label=AA,
                           color=self.colorsAA[AA], edgecolor='black')

            # Set the edge thickness
            for spine in ax.spines.values():
                spine.set_linewidth(self.lineThickness)

            # Set axis limits
            plt.ylim(0, yMax)

            fig.canvas.mpl_connect('key_press_event', pressKey)
            plt.show()

            if self.saveFigures:
                # Define: Save location
                if datasetTag is None:
                    figLabel = (f'AA Distribution - {self.enzymeName} - '
                                f'{sortType} - Y Max {yMax} - {codonType} - '
                                f'MinCounts {self.minSubCount}.png')
                else:
                    if codonType == datasetTag:
                        figLabel = (f'AA Distribution - {codonType} Codon - '
                                    f'Y Max {yMax}.png')
                    else:
                        figLabel = (f'AA Distribution - {self.enzymeName} - {datasetTag} - '
                                    f'{sortType} - Y Max {yMax} - {codonType} - '
                                    f'MinCounts {self.minSubCount}.png')
                saveLocation = os.path.join(self.pathSaveFigs, figLabel)

                # Save figure
                if os.path.exists(saveLocation):
                    print(f'{yellow}The figure was not saved\n\n'
                          f'File was already found at path:\n'
                          f'     {saveLocation}{resetColor}\n\n')
                else:
                    print(f'Saving figure at path:\n'
                          f'     {greenDark}{saveLocation}{resetColor}\n\n')
                    fig.savefig(saveLocation, dpi=self.figureResolution)

        # Plot the data
        if plotInitial and not skipInitial:
            plotFig(probability=probInitial, sortType='Initial Sort')
        if plotFinal:
            if codonType == datasetTag:
                plotFig(probability=probFinal, sortType=datasetTag)
            else:
                plotFig(probability=probFinal, sortType='Final Sort')



    def plotPositionalProbDist(self, probability, entropyScores, sortType, datasetTag):
        print('======================== Plot: Probability Distribution '
              '=========================')
        for position in entropyScores.index:
            yMax = 1

            # Extract values for plotting
            probabilities = list(probability.loc[:, position])

            # Calculate positional entropy
            shannonS = 0
            notNumber = False
            for AA in self.letters:
                prob = probability.loc[AA, position]
                shannonS += -prob * np.log2(prob)
            if math.isnan(shannonS):
                shannonS = 0
                notNumber = True

            # Plot the data
            setEdgeColor = True
            fig, ax = plt.subplots(figsize=self.figSizeMini)
            if setEdgeColor:
                widthBar = 0.8
                xPos = np.arange(len(probability.index))

                for index, AA in enumerate(probability.index):
                    edgeColor = 'dimgray' if AA in ['F', 'W', 'Y'] else 'black'
                    ax.bar(xPos[index], probability.loc[AA, position], widthBar, label=AA,
                           color=self.colorsAA[AA], edgecolor=edgeColor)
            else:
                plt.bar(self.letters, probabilities,
                        color=[self.colorsAA[AA] for AA in self.letters])
            # plt.xlabel('Amino Acids', fontsize=self.labelSizeAxis)
            plt.ylabel('Probability', fontsize=self.labelSizeAxis)
            if notNumber:
                plt.title(f'{self.enzymeName}: '
                          f'Amino Acid Distribution at {position}\n'
                          f'ΔS = {entropyScores.loc[position, "ΔS"]:.3f}, '
                          f'Shannon Entropy = {shannonS:.0f}',
                          fontsize=self.labelSizeTitle, fontweight='bold')
            else:
                plt.title(f'{self.enzymeName}: '
                          f'Amino Acid Distribution at {position}\n'
                          f'ΔS = {entropyScores.loc[position, "ΔS"]:.3f}, '
                          f'Shannon Entropy = {shannonS:.3f}',
                          fontsize=self.labelSizeTitle, fontweight='bold')
            plt.subplots_adjust(top=0.898, bottom=0.1, left=0.129, right=0.936)
            plt.ylim(0, yMax)

            # Set tick parameters
            ax.tick_params(axis='both', which='major', length=self.tickLength,
                           labelsize=self.labelSizeTicks)

            # Set x-ticks
            xTicks = np.arange(0, len(self.letters), 1)
            ax.set_xticks(xTicks) # Set the positions of the ticks
            ax.set_xticklabels(self.letters, rotation=0, ha='center',
                               fontsize=self.labelSizeTicks)
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_markeredgewidth(self.lineThickness) # Set tick width

            # Set y-ticks
            tickStepSize = 0.2
            yTicks = np.arange(0, yMax + tickStepSize, tickStepSize)
            yTickLabels = [f'{tick:.0f}' if tick == 0 or tick == 1 else f'{tick:.1f}'
                           for tick in yTicks]
            ax.set_yticks(yTicks) # Set the positions of the ticks
            ax.set_yticklabels(yTickLabels)
            for tick in ax.yaxis.get_major_ticks():
                tick.tick1line.set_markeredgewidth(self.lineThickness)

            # Set the edge thickness
            for spine in ax.spines.values():
                spine.set_linewidth(self.lineThickness)

            fig.canvas.mpl_connect('key_press_event', pressKey)
            plt.show()

            # Save the figure
            if self.saveFigures:
                # Define: Save location
                if datasetTag is None:
                    figLabel = (f'AA Distribution - {position} - {self.enzymeName} - '
                                f'{sortType} - Unfiltered - '
                                f'MinCounts {self.minSubCount}.png')
                else:
                    figLabel = (f'AA Distribution - {position} - {self.enzymeName} - '
                                f'{sortType} - {datasetTag} - '
                                f'MinCounts {self.minSubCount}.png')
                saveLocation = os.path.join(self.pathSaveFigs, figLabel)

                # Save figure
                if os.path.exists(saveLocation):
                    print(f'{yellow}The figure was not saved\n\n'
                          f'File was already found at path:\n'
                          f'     {saveLocation}{resetColor}\n\n')
                else:
                    print(f'Saving figure at path:\n'
                          f'     {greenDark}{saveLocation}{resetColor}\n\n')
                    fig.savefig(saveLocation, dpi=self.figureResolution)



    def plotStats(self, countedData, totalCounts, dataType,
                  combinedMotifs=False, releasedCounts=False):
        print('========================= Plot: Statistical Evaluation '
              '==========================')
        # Set figure title
        if self.showSampleSize:
            title = f'\n{self.enzymeName}\nAverage ES\nN={totalCounts:,}'
        else:
            title = f'\n\n{self.enzymeName}\nAverage ES'


        # Create heatmap
        cMapCustom = self.createCustomColorMap(colorType=dataType)

        # Convert the counts to a data frame for Seaborn heatmap
        if self.residueLabelType == 0:
            countedData.index = [residue[0] for residue in self.residues]
        elif self.residueLabelType == 1:
            countedData.index = [residue[1] for residue in self.residues]
        elif self.residueLabelType == 2:
            countedData.index = [residue[2] for residue in self.residues]

        # Plot the heatmap with numbers centered inside the squares
        fig, ax = plt.subplots(figsize=self.figSizeEM)
        heatmap = sns.heatmap(countedData, annot=True, fmt='.3f', cmap=cMapCustom,
                              cbar=True, linewidths=self.lineThickness-1,
                              linecolor='black', square=False, center=None,
                              annot_kws={'fontweight': 'bold'})
        ax.set_xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        ax.set_ylabel('Residue', fontsize=self.labelSizeAxis)
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        figBorders = [0.852, 0.075, 0.117, 1]
        plt.subplots_adjust(top=figBorders[0], bottom=figBorders[1],
                            left=figBorders[2], right=figBorders[3])


        # Set the thickness of the figure border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks, width=self.lineThickness)
        ax.tick_params(axis='y', labelrotation=0)

        # Set x-ticks
        xTicks = np.arange(len(countedData.columns)) + 0.5
        ax.set_xticks(xTicks)
        ax.set_xticklabels(countedData.columns)

        # Set y-ticks
        yTicks = np.arange(len(countedData.index)) + 0.5
        ax.set_yticks(yTicks)
        ax.set_yticklabels(countedData.index)

        # Set the edge thickness
        for _, spine in ax.spines.items():
            spine.set_visible(True)

        # Modify the colorbar
        cbar = heatmap.collections[0].colorbar
        tickLabels = cbar.ax.get_yticklabels()
        cbarLabels = []
        for label in tickLabels:
            labelText = label.get_text() # Get the text of the label
            try:
                labelValue = float(labelText) # Convert to a float
                if labelValue.is_integer():  # Check if it's an integer
                    cbarLabels.append(int(labelValue)) # Append as an integer
            except ValueError:
                print(f'{orange}ERROR: Unable to plot the{cyan} {dataType}{orange} '
                      f'label{cyan} {label}\n')
                sys.exit(1)
        cbar.set_ticks(cbarLabels) # Set the positions of the ticks
        cbar.set_ticklabels(cbarLabels)
        cbar.ax.tick_params(axis='y', which='major', labelsize=self.labelSizeTicks,
                            length=self.tickLength, width=self.lineThickness)
        cbar.outline.set_linewidth(self.lineThickness)
        cbar.outline.set_edgecolor('black')

        fig.canvas.mpl_connect('key_press_event', pressKey)
        if self.setFigureTimer:
            plt.ion()
            plt.show()
            plt.pause(self.figureTimerDuration)
            plt.close(fig)
            plt.ioff()
        else:
            plt.show()

        # Save the figure
        if self.saveFigures:
            self.saveFigure(fig=fig, figType=dataType, seqLen=len(xTicks),
                            combinedMotifs=combinedMotifs, releasedCounts=releasedCounts)



    def plotWordCloud(self, substrates, clusterNumPCA=None, combinedMotifs=False,
                      predActivity=False, predModel=False):
        print('=============================== Plot: Word Cloud '
              '================================')
        if clusterNumPCA is not None:
            print(f'Selecting PCA Population:{red} {clusterNumPCA}{resetColor}')
        else:
            print(f'Substrates: {purple}{self.datasetTag}{resetColor}')
        iteration = 0
        for substrate, count in substrates.items():
            print(f'     {blue}{substrate}{resetColor}, '
                  f'Count:{red} {round(count, 1):,}{resetColor}')
            iteration += 1
            if iteration == self.printNumber:
                break
        print('')

        # Define: Figure title
        if predActivity:
            title = f'{self.enzymeName}\n{predModel}'
        else:
            if combinedMotifs and len(self.motifIndexExtracted) > 1:
                title = self.titleWordsCombined
            elif combinedMotifs:
                title = self.titleWordsCombined
                title = title.replace('Combined ', '')
            else:
                title = self.titleWords


        # Limit the number of words
        if self.wordsLimit:
            print(f'Selecting: {red}{self.wordsTotal}{resetColor} words')
            subs = {}
            iteration = 0
            for substrate, count in substrates.items():
                subs[substrate] = count
                iteration += 1
                if iteration >= self.wordsTotal:
                    break
            substrates = subs
        totalWords = len(substrates)
        print(f'Plotting: {red}{totalWords:,}{resetColor} words\n\n')

        # Create word cloud
        cmap = self.createCustomColorMap(colorType='Word Cloud')
        wordcloud = (WordCloud(
            width=950,
            height=800,
            background_color='white',
            min_font_size=10, # Minimum font size
            max_font_size=120, # Maximum font size
            scale=5,  # Increase scale for larger words
            colormap=cmap
        ).generate_from_frequencies(substrates))


        # Display the word cloud
        fig = plt.figure(figsize=self.figSize, facecolor='white')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        plt.axis('off')
        fig.canvas.mpl_connect('key_press_event', pressKey)
        fig.tight_layout()
        if self.setFigureTimer:
            plt.ion()
            plt.show()
            plt.pause(self.figureTimerDuration)
            plt.close(fig)
            plt.ioff()
        else:
            plt.show()


        # Save the Figure
        if self.saveFigures:
            if self.motifLen == None:
                seqLength = len(self.xAxisLabels)
            else:
                seqLength = self.motifLen

            # Define: Save location
            figLabel = (f'{self.enzymeName} - Words - {self.datasetTag} - '
                        f'{seqLength} AA - Plot {totalWords} - '
                        f'MinCounts {self.minSubCount}.png')
            if self.wordsLimit:
                figLabel = figLabel.replace(
                    f'Plot {totalWords}',
                    f'Select {self.wordsTotal} Plot {totalWords}')
            if len(self.motifIndexExtracted) > 1:
                figLabel = figLabel.replace('Reading Frame',
                                            'Combined Reading Frame')
            if clusterNumPCA is not None:
                figLabel = figLabel.replace('Words',
                                            f'Words - PCA {clusterNumPCA}')
            if predActivity:
                figLabel = figLabel.replace(
                    self.datasetTag,
                    f'{self.datasetTag} - Predictions - {predModel}')
            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                print(f'{yellow}The figure was not saved\n\n'
                      f'File was already found at path:\n'
                      f'     {saveLocation}{resetColor}\n\n')
            else:
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)



    def extractMotif(self, substrates, motifFrame, frameIndicies, datasetTag):
        print('================================= Extract Motif '
              '=================================')
        print(f'Binning Substrates: {purple}{datasetTag}{resetColor}\n'
              f'Start Position:{greenLightB} {motifFrame[frameIndicies[0]]}'
              f'{resetColor}\n'
              f'   Start Index:{greenLightB} {frameIndicies[0]}{resetColor}\n'
              f'End Position:{greenLightB} {motifFrame[frameIndicies[-1]]}'
              f'{resetColor}\n'
              f'   End Index:{greenLightB} {frameIndicies[-1]}{resetColor}\n\n')
        frameLength = len(motifFrame)
        sys.exit(1)


        # Bin substrates
        motifs = {}
        countTotalSubstrates = 0
        countUniqueSubstrates = 0
        for index, subsFixedFrame in enumerate(substrates):
            # Define fixed frame positions & extract the data
            startPosition = frameIndicies[0]
            startSubPrevious = startPosition
            if index != 0:
                # Evaluate previous fixed frame index
                fixedPosDifference = (self.fixedPos[index] -
                                      self.fixedPos[index - 1])
                startSubPrevious += fixedPosDifference
                # print(f'Pos Curr: {purple}{self.fixedPos[index]}{resetColor}\n'
                #       f'Pos Prev: {purple}{self.fixedPos[index - 1]}{resetColor}')
                # print(f'     Start Diff:{greenLightB} {fixedPosDifference}{resetColor}\n'
                #       f'     Start Prev:{greenLightB} {startSubPrevious}{resetColor}')
                startSub = index + startSubPrevious - 1
                endSub = startSub + frameLength
            else:
                startSub = startPosition
                endSub = frameIndicies[-1] + 1
            # print(f'Start:{red} {startSub}{resetColor}\n'
            #       f'Stop:{red} {endSub}{resetColor}\n\n')

            # Print: Substrates
            print(f'Fixed Motif: {purple}{self.fixedAA[0]}@{self.fixedPos[index]}'
                  f'{resetColor}')
            iteration = 0
            for substrate, count in subsFixedFrame.items():
                print(f'Substrate:{pink} {substrate}{resetColor}\n'
                      f'    Frame:{blue} {substrate[startSub:endSub]}{resetColor}\n'
                      f'    Count:{red} {count:,}{resetColor}')
                iteration += 1
                if iteration == self.printNumber:
                    print('\n')
                    break

                # Add substrate frame to the substrate dictionary
                for substrate, count in subsFixedFrame.items():
                    countTotalSubstrates += count
                    sub = substrate[startSub:endSub]

                    if sub in motifs:
                        motifs[sub] += count
                    else:
                        countUniqueSubstrates += 1
                        motifs[sub] = count
        # Sort the dictionary
        motifs = dict(sorted(motifs.items(), key=lambda item: item[1], reverse=True))

        # Print: Binned substrates
        iteration = 0
        print(f'Binned Substrates{resetColor}: {purple}{datasetTag}{resetColor}')
        for substrate, count in motifs.items():
            print(f'     {pink} {substrate}{resetColor}, '
                  f'Count:{red} {count:,}{resetColor}')
            iteration += 1
            if iteration == self.printNumber:
                print('\n')
                break
        print(f'Total Substrates:{red} {countTotalSubstrates:,}{resetColor}\n'
              f'Unique Substrates:{red} {countUniqueSubstrates:,}{resetColor}\n\n')

        return motifs, countTotalSubstrates



    def evaluateSubtrees(self, trie, motifTrie):
        print('============================= Evaluate Suffix Tree '
              '==============================')
        print(f'Datapoints: {len(motifTrie.keys())}')

        def subtreeTable(subtreeFreq):
            # Sort motifs by length
            sortedMotifs = sorted(subtreeFreq.keys(), key=len)

            # Organize motifs by their length and sort by frequency (highest first)
            motifGroups = {}
            for motif in sortedMotifs:
                length = len(motif)
                if length not in motifGroups:
                    motifGroups[length] = []
                motifGroups[length].append((motif, subtreeFreq[motif]))

            # Sort motifs in each length group by frequency (descending)
            for length in motifGroups:
                motifGroups[length].sort(key=lambda x: x[1], reverse=True)

            # Convert motifs back to formatted strings
            for length in motifGroups:
                motifGroups[length] = [f"{motif}: {round(freq, 5)}"
                                       for motif, freq in motifGroups[length]]

            # Find the max number of motifs in any length group
            maxRows = max(len(motifs) for motifs in motifGroups.values())

            # Construct the table row by row
            tableData = []
            for i in range(maxRows):
                row = []
                for length in sorted(motifGroups.keys()):
                    motifs = motifGroups[length]
                    row.append(motifs[i] if i < len(
                        motifs) else "") # Fill missing values with empty strings
                tableData.append(row)

            # Convert to DataFrame
            motifTable = pd.DataFrame(tableData,
                                      index=range(1, len(motifTrie.keys()) + 1),
                                      columns=[str(length)
                                               for length in sorted(motifGroups.keys())])
            print(f'{motifTable}\n\n')

            return motifTable


        def printTrie(node, level=0, path=""):
            # Recursively print the Trie structure
            if node is None:
                return

            # Print: Current node's path and level
            print("  " * level + f"Level {level}: {path}")

            # Recursively print all children of the current node
            for char, nodeChild in node.children.items():
                printTrie(nodeChild, level + 1, path + char)


        motifsTotal = 0
        for motif, count in motifTrie.items():
            motifsTotal += count
        print(f'Total Motifs: {motifsTotal:,}\n')

        # Evaluate: Partial sequence counts
        subtreeCount = {}
        motifLength = len(next(iter(motifTrie)))
        for index in range(motifLength):
            for motif, count in motifTrie.items():
                subSeq = motif[0:index+1]
                if subSeq in subtreeCount.keys():
                    subtreeCount[subSeq] += count
                else:
                    subtreeCount[subSeq] = count

        # Evaluate: Partial sequence frequency
        subtreeFreq = {}
        for subSeq, count in subtreeCount.items():
            subtreeFreq[subSeq] = count / motifsTotal
        prevSeqLen = 1
        for subSeq, count in subtreeFreq.items():
            if len(subSeq) != prevSeqLen:
                prevSeqLen = len(subSeq)
        motifTable = subtreeTable(subtreeFreq)


        # Plot the trie
        printTrie(trie.root)
        print('\n')

        return motifTable



    def plotTrie(self, trie, motifTable, countsMotif, datasetTag):
        print('=============================== Plot: Suffix Tree '
              '===============================')
        import networkx as nx

        inOffset = 2000
        inNodeSizeMax = 800
        inNodeSizeMin = 100
        inFontSize = 10
        inScaleX = 2
        inScaleY = 1

        # Calculate: Node size
        nodeSizes = pd.DataFrame('',
                                 index=motifTable.index,
                                 columns=motifTable.columns)
        for col in motifTable.columns:
            for index, entry in enumerate(motifTable[col].dropna()):
                if ": " in entry:
                    motif, rf = entry.split(": ")
                    nodeSize = inNodeSizeMax - (inNodeSizeMax * (1 - float(rf)))
                    if nodeSize < 100:
                        nodeSize = inNodeSizeMin
                    if len(motif) > 2:
                        motif = motif[-2:]
                    nodeSizes.loc[index+1, col] = f'{motif}: {nodeSize:.2f}'
        print(f'Node Size:\n{nodeSizes}\n')


        def addNodesToGraph(node, graph, scaleX, scaleY, offset=inOffset,
                            nodeSizesDF=None):
            pos = {}
            nodeSizes = {}
            nodeCountLevel = {}

            # Track node index separately
            queue = [(node, None, '', '', 0,
                      1)]  # (currentNode, parentID, char, fullMotif, level, index)

            while queue:
                nodeCurrent, parent, char, motifSoFar, level, index = queue.pop(0)
                nodeID = f"{char}-{level}-{id(nodeCurrent)}"

                if level not in nodeCountLevel:
                    nodeCountLevel[level] = []
                nodeCountLevel[level].append(
                    (nodeCurrent, parent, char, nodeID, motifSoFar, index))

                fullMotif = motifSoFar + char  # Build full motif sequence

                # Assign node size from nodeSizesDF if available
                nodeSize = inNodeSizeMin  # Default size
                if nodeSizesDF is not None and level in nodeSizesDF.columns:
                    entry = nodeSizesDF.iloc[index - 1, level]  # Use the tracked index
                    if isinstance(entry, str) and ": " in entry:
                        _, size = entry.split(": ")
                        nodeSize = float(size)

                graph.add_node(nodeID, label=char, size=nodeSize)
                nodeSizes[nodeID] = nodeSize

                if parent is not None:
                    graph.add_edge(parent, nodeID, arrowstyle='->')

                # Track child nodes with incremented index
                childIndex = 1
                for child_char, nodeChild in nodeCurrent.children.items():
                    queue.append(
                        (nodeChild, nodeID, child_char, fullMotif, level + 1, childIndex))
                    childIndex += 1  # Ensure a unique index for each child

            return pos, nodeSizes


        # Build the graph
        graph = nx.DiGraph()
        pos, nodeSizes = addNodesToGraph(trie.root, graph, scaleX=inScaleX,
                                         scaleY=inScaleY, offset=inOffset,
                                         nodeSizesDF=nodeSizes)
        finalNodeSizes = [graph.nodes[node]["size"] for node in graph.nodes]

        # Get node labels
        labels = {node: data['label'] for node, data in graph.nodes(data=True)}

        # Print: Dataset tag
        if datasetTag is None:
            figLabel = f'Suffix Tree - {self.enzymeName} - {countsMotif} - Unfixed'

        else:
            figLabel = f'Suffix Tree - {self.enzymeName} - {countsMotif} - {datasetTag}'


        # Plot the data
        fig, ax = plt.subplots(figsize=self.figSize)
        fig.canvas.mpl_connect('key_press_event', pressKey)

        # Draw graph
        nx.draw(graph, pos, with_labels=True, labels=labels, node_size=finalNodeSizes,
                node_color="#F18837", font_size=inFontSize, font_weight="bold",
                edge_color="#101010", ax=ax, arrows=False)
        plt.title(f'{self.enzymeName}: {datasetTag}\nTop {countsMotif:,} Motifs',
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Save the figure
        if self.saveFigures:
            # Define: Save location
            figLabel += '.png'
            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                print(f'{yellow}The figure was not saved\n\n'
                      f'File was already found at path:\n'
                      f'     {saveLocation}{resetColor}\n\n')
            else:
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)



    def generateSubstrates(self, df, eMap, minES, dataType, subsReq={}, filter={}):
        print('============================== Generate Substrates '
              '==============================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n'
              f'DataFrame Type: {purple}{dataType}{resetColor}\n'
              f'{df}\n')

        # Select favorable AAs
        print(f'Preferred Residues:')
        preferredAAs = []
        addedAAsTag = ''
        for index, pos in enumerate(df.columns):
            if pos in filter.keys():
                nonzeroAAs = filter[pos]
            else:
                nonzeroAAs = df[df[pos] != 0].index.tolist()

                # Filter out AAs with low ES
                ES = eMap.loc[:, pos]
                nonzeroAAs = [aa for aa in nonzeroAAs if ES[aa] >= minES]

            # Add AAs for inSubsPred
            if subsReq != {}:
                addedAAs = []
                for key, substrates in subsReq.items():
                    for substrate in substrates:
                        AA = substrate[index]
                        if AA not in addedAAs and AA not in nonzeroAAs:
                            addedAAs.append(AA)
                            nonzeroAAs.append(AA)

                    # Record added AAs
                    if addedAAs:
                        if len(addedAAs) == 1:
                            label = f'{addedAAs[0]}@{pos}'
                        else:
                            label = f'[{", ".join(addedAAs)}]@{pos}'
                        if index == len(df.columns) - 1:
                            addedAAsTag += f'{label}'
                        else:
                            addedAAsTag += f'{label}_'
            print(f'     Position {greenLight}{pos}{resetColor}: '
                  f'{pink}{", ".join(nonzeroAAs)}{resetColor}')
            preferredAAs.append(nonzeroAAs)
        print()

        if addedAAsTag != '':
            print(f'Added AAs: {pink}{addedAAsTag}{resetColor}')
        print(f'Minimum ES: {red}{minES}{resetColor}\n')
        # Generate all possible substrate combinations
        allCombos = list(product(*preferredAAs))
        # print(f'\n\nCombos:\n{allCombos}\n\n')

        # Convert tuples to strings (AA sequences)
        genSubstrates = [''.join(combo) for combo in allCombos]
        NSubs = len(genSubstrates)
        print(f'Generated substrates: {red}N = {NSubs:,}{resetColor}')
        for iteration in range(0, self.printNumber):
            index = random.randint(0, NSubs)
            print(f'     {pink}{genSubstrates[index]}{resetColor}')
        print('\n')

        return genSubstrates, addedAAsTag



# =================================== Machine Learning ===================================
class CNN:
    def __init__(self, substrates):
        self.substrates = substrates



class RandomForestRegressorXGB:
    def __init__(self, dfTrain, dfPred, subsPredChosen, minES, pathModel, modelTag,
                 NTrees, device, printNumber):
        print('============================ Random Forest Regressor '
              '============================')
        from xgboost import XGBRFRegressor

        print(f'Module: {purple}XGBoost{resetColor}\n'
              f'Model: {purple}{modelTag}{resetColor}\n')
        self.device = device
        self.NTrees = NTrees
        self.predictions = {}
        subsPred = list(dfPred.index)


        # Get Model: Gradient Boosting
        if os.path.exists(pathModel):
            # Load the model
            self.loadModel(model=XGBRFRegressor(), pathModel=pathModel)
        else:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            # Process dataframe
            testSize = 0.2
            x = dfTrain.drop(columns='activity').values
            y = np.log1p(dfTrain['activity'].values)
            xtrain, xTest, yTrain, yTest = train_test_split(
                x, y, test_size=testSize, random_state=19)

            # Train the model
            print(f'Training Model: {blue}Split Training Set '
                  f'{round((1 - testSize) * 100, 0)}{pink}:'
                  f'{blue}{round((testSize) * 100, 0)}{resetColor}\n'
                f'     Train: {red}{xtrain.shape}{resetColor}\n'
                f'     Test: {red}{xTest.shape}{resetColor}')
            start = time.time()
            model = XGBRFRegressor(device=self.device, tree_method="hist",
                                   n_estimators=100, random_state=42)
            model.fit(xtrain, yTrain)
            end = time.time()
            runtime = (end - start) / 60
            print(f'      Training Time: {red}{round(runtime, 3):,} min{resetColor}\n')
            print(f'Evaluate Model Accuracy:')
            yPred = model.predict(xTest)
            MAE = mean_absolute_error(yPred, yTest)
            MSE = mean_squared_error(yPred, yTest)
            R2 = r2_score(yTest, yPred)
            print(f'     MAE: {red}{round(MAE, 3)}{resetColor}\n'
                  f'     MSE: {red}{round(MSE, 3)}{resetColor}\n'
                  f'     R2: {red}{round(R2, 3)}{resetColor}\n\n')

            # Train the model
            print(f'Training the model: {purple}Full Training Set{resetColor}')
            start = time.time()
            model = XGBRFRegressor(device=self.device, tree_method="hist",
                                   n_estimators=100, random_state=42)
            model.fit(x, y)
            end = time.time()
            runtime = (end - start) / 60
            print(f'      Training time: {red}{round(runtime, 3):,} min{resetColor}\n')
            print(f'Saving Trained ESM Model:\n'
                  f'     {greenDark}{pathModel}{resetColor}\n\n')
            model.save_model(pathModel)

            self.loadModel(model=XGBRFRegressor(), pathModel=pathModel)


        # Predict substrate activity
        print(f'Predicting Substrate Activity:\n'
              f'     Total Substrates: {red}{len(dfPred.index):,}{resetColor}')
        start = time.time()
        activityPred = self.model.predict(dfPred)
        activityPred = np.expm1(activityPred)  # Reverse log1p transform
        end = time.time()
        runtime = (end - start) * 1000
        print(f'     Runtime: {red}{round(runtime, 3):,} ms{resetColor}\n')

        # Rank predictions
        activityPredRandom = {
            substrate: float(score)
            for substrate, score in zip(subsPred, activityPred)
        }
        activityPredRandom = dict(sorted(
            activityPredRandom.items(), key=lambda x: x[1], reverse=True))
        self.predictions['Random'] = activityPredRandom
        print(f'Predicted Activity: {purple}Random Substrates - Min ES: {minES}'
              f'{resetColor}')
        for iteration, (substrate, value) in enumerate(activityPredRandom.items()):
            if iteration >= printNumber:
                break
            print(f'     {substrate}: {red}{round(value, 3):,}{resetColor}')
        print('\n')

        # Get: Chosen substrate predictions
        if subsPredChosen != {}:
            print(f'Predicted Activity: {purple}Chosen Substrates{resetColor}')
            for key, substrates in subsPredChosen.items():
                activityPredChosen = {}
                for substrate in substrates:
                    activityPredChosen[substrate] = (
                        activityPredRandom)[substrate]
                activityPredChosen = dict(sorted(activityPredChosen.items(),
                                                 key=lambda x: x[1], reverse=True))
                self.predictions[key] = activityPredChosen
                print(f'Substrate Set: {purple}{key}{resetColor}')
                for iteration, (substrate, activity) in (
                        enumerate(activityPredChosen.items())):
                    activity = float(activity)
                    print(f'     {pink}{substrate}{resetColor}: '
                          f'{red}{round(activity, 3):,}{resetColor}')
                print('\n')


    def loadModel(self, model, pathModel):
        print(f'Loading Trained ESM Model:\n'
              f'     {greenDark}{pathModel}{resetColor}\n')
        self.model = model
        self.model.load_model(pathModel)



class RandomForestRegressor:
    def __init__(self, dfTrain, dfPred, subsPredChosen, minES, pathModel, modelTag,
                 NTrees, device, printNumber):
        print('============================ Random Forest Regressor '
              '============================')
        from sklearn.ensemble import RandomForestRegressor

        print(f'Module: {purple}Scikit-Learn{resetColor}\n'
              f'Model: {purple}{modelTag}{resetColor}\n')
        self.device = device
        self.NTrees = NTrees
        self.predictions = {}
        subsPred = list(dfPred.index)


        # # Get Model: Random Forrest Regressor
        if os.path.exists(pathModel):
            self.loadModel(model=RandomForestRegressor(), pathModel=pathModel)
        else:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            # Process dataframe
            testSize = 0.2
            x = dfTrain.drop(columns='activity').values
            y = np.log1p(dfTrain['activity'].values)
            xtrain, xTest, yTrain, yTest = train_test_split(
                x, y, test_size=testSize, random_state=19)

            # Train the model
            print(f'Training Model: {blue}Split Training Set '
                  f'{round((1 - testSize) * 100, 0)}{pink}:'
                  f'{blue}{round((testSize) * 100, 0)}{resetColor}\n'
                  f'     Train: {red}{xtrain.shape}{resetColor}\n'
                  f'     Test: {red}{xTest.shape}{resetColor}')
            start = time.time()
            model = RandomForestRegressor(n_estimators=self.NTrees, random_state=42)
            model.fit(xtrain, yTrain)
            end = time.time()
            runtime = (end - start) / 60
            print(f'      Training Time: {red}{round(runtime, 3):,} min{resetColor}\n')

            # Evaluate the model
            print(f'Evaluate Model Accuracy:')
            yPred = model.predict(xTest)
            MAE = mean_absolute_error(yPred, yTest)
            MSE = mean_squared_error(yPred, yTest)
            R2 = r2_score(yTest, yPred)
            print(f'     MAE: {red}{round(MAE, 3)}{resetColor}\n'
                  f'     MSE: {red}{round(MSE, 3)}{resetColor}\n'
                  f'     R2: {red}{round(R2, 3)}{resetColor}\n')


            # Train the model
            print(f'Training the model: {purple}Full Training Set{resetColor}')
            start = time.time()
            self.model = RandomForestRegressor(n_estimators=self.NTrees, random_state=42)
            self.model.fit(x, y)
            end = time.time()
            runtime = (end - start) / 60
            print(f'Training time: {red}{round(runtime, 3):,} min{resetColor}\n')
            print(f'Saving Trained ESM Model:\n'
                  f'     {greenDark}{pathModel}{resetColor}\n\n')
            joblib.dump(self.model, pathModel)

            self.loadModel(model=RandomForestRegressor(), pathModel=pathModel)


        # Predict substrate activity
        print(f'Predicting Substrate Activity:\n'
              f'     Total Substrates: {red}{len(dfPred.index):,}{resetColor}')
        start = time.time()
        activityPred = self.model.predict(dfPred.values)
        activityPred = np.expm1(activityPred) # Reverse log1p transform
        end = time.time()
        runtime = (end - start) * 1000
        print(f'     Runtime: {red}{round(runtime, 3):,} ms{resetColor}\n')

        # Rank predictions
        activityPredRandom = {
            substrate: float(score)
            for substrate, score in zip(subsPred, activityPred)
        }
        activityPredRandom = dict(sorted(
            activityPredRandom.items(), key=lambda x: x[1], reverse=True))
        self.predictions['Random'] = activityPredRandom
        print(f'Predicted Activity: {purple}Random Substrates - Min ES: {minES}'
              f'{resetColor}')
        for iteration, (substrate, value) in enumerate(activityPredRandom.items()):
            if iteration >= printNumber:
                break
            print(f'     {substrate}: {red}{round(value, 3):,}{resetColor}')
        print('\n')

        # Get: Chosen substrate predictions
        if subsPredChosen != {}:
            print(f'Predicted Activity: {purple}Chosen Substrates{resetColor}')
            for key, substrates in subsPredChosen.items():
                activityPredChosen = {}
                for substrate in substrates:
                    activityPredChosen[substrate] = (
                        activityPredRandom)[substrate]
                activityPredChosen = dict(sorted(activityPredChosen.items(),
                                                      key=lambda x: x[1], reverse=True))
                self.predictions[key] = activityPredChosen
                print(f'Substrate Set: {purple}{key}{resetColor}')
                for iteration, (substrate, activity) in (
                        enumerate(activityPredChosen.items())):
                    activity = float(activity)
                    print(f'     {pink}{substrate}{resetColor}: '
                          f'{red}{round(activity, 3):,}{resetColor}')
                print('\n')

    def loadModel(self, model, pathModel):
        print(f'Loading Trained ESM Model:\n'
              f'     {greenDark}{pathModel}{resetColor}\n')
        self.model = model
        self.model = joblib.load(pathModel)



class PredictActivity:
    def __init__(self, enzymeName, datasetTag, folderPath, subsTrain, subsPred,
                 subsPredChosen, tagChosenSubs, batchSize, minES, labelsXAxis,
                 printNumber):
        # Parameters: Files
        self.pathFolder = folderPath
        self.pathData = os.path.join(self.pathFolder, 'Data')
        self.pathEmbeddings = os.path.join(self.pathFolder, 'Embeddings')
        self.pathModels = os.path.join(self.pathFolder, 'Models')
        self.pathSaveFigs = os.path.join(self.pathFolder, 'Figures')
        os.makedirs(self.pathData, exist_ok=True)
        os.makedirs(self.pathEmbeddings, exist_ok=True)
        os.makedirs(self.pathModels, exist_ok=True)

        # Parameters: Dataset
        self.enzymeName = enzymeName
        self.datasetTag = datasetTag
        self.subsPredChosen = subsPredChosen
        self.tagChosenSubs = tagChosenSubs
        self.minES = minES
        self.labelsXAxis = labelsXAxis
        self.printNumber = printNumber
        self.predictions = {}

        # Parameters: Model
        self.batchSize = batchSize
        self.NTrees = 100
        self.device = self.getDevice()
        self.embeddingsNameESM = ''
        self.subsInitial = None
        self.subsTrain = subsTrain
        self.subsTrainN = len(self.subsTrain.keys())
        self.subsPred = subsPred
        self.subsPredN = len(subsPred)

        # Parameters: ESM
        modelPrams = 2
        if modelPrams == 0: # Choose: ESM PLM model
            self.sizeESM = '15B Params'
        elif modelPrams == 1:
            self.sizeESM = '3B Params'
        else:
            self.sizeESM = '650M Params'
        self.tagEmbeddingsTrain = (
            f'Embeddings - ESM {self.sizeESM} - {self.enzymeName} - {self.datasetTag} - '
            f'Batch {self.batchSize} - N {self.subsTrainN} - {len(self.labelsXAxis)} AA')
        self.tagEmbeddingsPred = (
            f'Embeddings - ESM {self.sizeESM} - {self.enzymeName} - Predictions - '
            f'Min ES {self.minES} - Batch {self.batchSize} - N {self.subsPredN} - '
            f'{len(self.labelsXAxis)} AA')
        if self.tagChosenSubs != '':
            self.tagEmbeddingsPred = self.tagEmbeddingsPred.replace(
                f'Min ES {self.minES}',
                f'Min ES {self.minES} - Added {self.tagChosenSubs}')


        # Define: Model paths
        modelTagScikit = (f'Random Forrest - Scikit - N Trees {self.NTrees} - '
                          f'{self.tagEmbeddingsTrain}')
        modelTagXGBoost = (f'Random Forrest - XGBoost - N Trees {self.NTrees} - '
                          f'{self.tagEmbeddingsTrain}')
        pathModelScikit = os.path.join(self.pathModels, f'{modelTagScikit}.ubj')
        pathModelXGBoost = os.path.join(self.pathModels, f'{modelTagXGBoost}.ubj')
        # ubj: Binary JSON file

        # Generate: Embeddings
        if not os.path.exists(pathModelScikit) or not os.path.exists(pathModelXGBoost):
            self.embeddingsSubsTrain = self.ESM(
                substrates=self.subsTrain, embeddings=self.tagEmbeddingsTrain,
                trainingSet=True)
        self.embeddingsSubsPred = self.ESM(
            substrates=self.subsPred, embeddings=self.tagEmbeddingsPred)


        # # Predict: Substrate activity
        # Model: Scikit-Learn Random Forest Regressor
        randomForestRegressor = RandomForestRegressor(
            dfTrain=self.embeddingsSubsTrain, dfPred=self.embeddingsSubsPred,
            subsPredChosen=self.subsPredChosen, minES=self.minES,
            pathModel=pathModelScikit, modelTag=modelTagScikit, NTrees=self.NTrees,
            device=self.device, printNumber=self.printNumber)
        self.predictions['Scikit-Learn: Random Forest Regressor'] = (
            randomForestRegressor.predictions)

        # Model: XGBoost Random Forest
        randomForestRegressorXGB = RandomForestRegressorXGB(
            dfTrain=self.embeddingsSubsTrain, dfPred=self.embeddingsSubsPred,
            subsPredChosen=self.subsPredChosen, minES=self.minES,
            pathModels=pathModelXGBoost, modelTag=modelTagScikit,
            device=self.device, NTrees=self.NTrees, printNumber=self.printNumber)
        self.predictions['XGBoost: Random Forest Regressor'] = (
            randomForestRegressorXGB.predictions)



    def getDevice(self):
        # Set device
        print('============================== Set Training Device '
              '==============================')
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f'Train with Device:{magenta} {device}{resetColor}\n'
                  f'Device Name:{magenta} {torch.cuda.get_device_name(device)}'
                  f'{resetColor}\n\n')
        else:
            import platform
            device = 'cpu'
            print(f'Train with Device:{magenta} {device}{resetColor}\n'
                  f'Device Name:{magenta} {platform.processor()}{resetColor}\n\n')

        return device



    def ESM(self, substrates, embeddings, trainingSet=False):
        print('=========================== Generate Embeddings: ESM '
              '============================')
        # Choose: ESM PLM model
        modelPrams = 2
        if modelPrams == 0:
            sizeESM = '15B Params'
        elif modelPrams == 1:
            sizeESM = '3B Params'
        else:
            sizeESM = '650M Params'

        # Inspect: Data type
        predictions = True
        if trainingSet:
            predictions = False
        print(f'Dataset: {purple}{embeddings}{resetColor}\n'
              f'Total unique substrates: {red}{len(substrates):,}{resetColor}')

        # Load: ESM Embeddings
        pathEmbeddings = os.path.join(self.pathEmbeddings, f'{embeddings}.csv')
        if os.path.exists(pathEmbeddings):
            print(f'\nLoading: ESM Embeddings\n'
                  f'     {greenDark}{pathEmbeddings}{resetColor}\n')
            subEmbeddings = pd.read_csv(pathEmbeddings, index_col=0)
            print(f'Substrate Embeddings shape: '
                  f'{pink}{subEmbeddings.shape}{resetColor}\n\n')

            return subEmbeddings


        # # Generate Embeddings
        # Step 1: Convert substrates to ESM model format and generate Embeddings
        totalSubActivity = 0
        subs = []
        values = []
        if trainingSet:
            # Randomize substrates
            items = list(substrates.items())
            random.shuffle(items)
            substrates = dict(items)

            for index, (substrate, value) in enumerate(substrates.items()):
                totalSubActivity += value
                subs.append((f'Sub{index}', substrate))
                values.append(value)
        else:
            for index, substrate in enumerate(substrates):
                subs.append((f'Sub{index}', substrate))
        sampleSize = len(substrates)
        print(f'Collected substrates:{red} {sampleSize:,}{resetColor}')
        if totalSubActivity != 0:
            if isinstance(totalSubActivity, float):
                print(f'Total Values:{red} {round(totalSubActivity, 1):,}'
                      f'{resetColor}')
            else:
                print(f'Total Values:{red} {totalSubActivity:,}{resetColor}')
        print()


        # Step 2: Load the ESM model and batch converter
        if sizeESM == '15B Params':
            model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            numLayersESM = 48
        elif sizeESM == '3B Params':
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            numLayersESM = 36
        else:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            numLayersESM = 33
            # esm2_t36_3B_UR50D has 36 layers
            # esm2_t33_650M_UR50D has 33 layers
            # esm2_t12_35M_UR50D has 12 layers
        model = model.to(self.device)

        # Get: batch tensor
        batchConverter = alphabet.get_batch_converter()


        # Step 3: Convert substrates to ESM model format and generate Embeddings
        try:
            batchLabels, batchSubs, batchTokens = batchConverter(subs)
            batchTokensCPU = batchTokens
            batchTokens = batchTokens.to(self.device)

        except Exception as exc:
            print(f'{orange}ERROR: The ESM has failed to evaluate your substrates\n\n'
                  f'Exception:\n{exc}\n\n'
                  f'Suggestion:'
                  f'     Try replacing: {cyan}esm.pretrained.esm2_t36_3B_UR50D()'
                  f'{orange}\n'
                  f'     With: {cyan}esm.pretrained.esm2_t33_650M_UR50D()'
                  f'{resetColor}\n')
            sys.exit(1)
        print(f'Batch Tokens:{greenLight} {batchTokens.shape}{resetColor}\n'
              f'{greenLight}{batchTokens}{resetColor}\n')

        # Record tokens
        slicedTokens = pd.DataFrame(batchTokensCPU[:, 1:-1],
                                    index=batchSubs,
                                    columns=self.labelsXAxis)
        if totalSubActivity != 0:
            slicedTokens['Values'] = values
        print(f'\nSliced Tokens:\n'
              f'{greenLight}{slicedTokens}{resetColor}\n')

        # Generate embeddings
        batchTotal = len(batchTokens)
        allEmbeddings = []
        allValues = []
        startInit = time.time()
        print('Generating ESM Embeddings:')
        with torch.no_grad():
            for i in range(0, len(batchTokens), self.batchSize):
                start = time.time()
                batch = batchTokens[i:i + self.batchSize].to(self.device)
                result = model(batch, repr_layers=[numLayersESM], return_contacts=False)
                tokenReps = result["representations"][numLayersESM]
                seqEmbed = tokenReps[:, 0, :].cpu().numpy()
                allEmbeddings.append(seqEmbed)
                end = time.time()
                runtime = end - start
                runtimeTotal = (end - startInit) / 60
                percentCompletion = round((i / batchTotal)* 100, 1)
                print(f'ESM Progress: {red}{i:,}{resetColor} / {red}{batchTotal:,}'
                      f'{resetColor} ({red}{percentCompletion} %{resetColor})\n'
                      f'     Batch Shape: {greenLight}{batch.shape}{resetColor}\n'
                      f'     Runtime: {red}{round(runtime, 3):,} s'
                      f'{resetColor}\n'
                      f'     Total Time: {red}{round(runtimeTotal, 3):,} min'
                      f'{resetColor}\n')
                if trainingSet:
                    allValues.extend(values[i:i + self.batchSize])

                # Clear data to help free memory
                del tokenReps, batch
                torch.cuda.empty_cache()
        end = time.time()
        runtime = end - start
        runtimeTotal = (end - startInit) / 60
        percentCompletion = round((batchTotal / batchTotal) * 100, 1)
        print(f'ESM Progress: {red}{batchTotal:,}{resetColor} / {red}{batchTotal:,}'
              f'{resetColor} ({red}{percentCompletion} %{resetColor})\n'
              f'     Runtime: {red}{round(runtime, 3):,} s'
              f'{resetColor}\n'
              f'     Total Time: {red}{round(runtimeTotal, 3):,} min'
              f'{resetColor}\n')


        # Step 4: Extract per-sequence Embeddings
        tokenReps = result["representations"][numLayersESM]  # (N, seq_len, hidden_dim)
        sequenceEmbeddings = tokenReps[:, 0, :]  # [CLS] token embedding: (N, hidden_dim)

        # Convert to numpy and store substrate activity proxy
        embeddings = np.vstack(allEmbeddings)
        if predictions:
            data = np.hstack([embeddings])
            columns = [f'feat_{i}' for i in range(embeddings.shape[1])]
        else:
            values = np.array(allValues).reshape(-1, 1)
            data = np.hstack([embeddings, values])
            columns = [f'feat_{i}' for i in range(embeddings.shape[1])] + ['activity']

        # Process Embeddings
        subEmbeddings = pd.DataFrame(data, index=batchSubs, columns=columns)
        print(f'Substrate Embeddings shape: '
              f'{pink}{sequenceEmbeddings.shape}{resetColor}\n\n')
        print(f'Embeddings saved at:\n'
              f'     {greenDark}{pathEmbeddings}{resetColor}\n\n')
        subEmbeddings.to_csv(pathEmbeddings)

        return subEmbeddings
