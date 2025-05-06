# PURPOSE: This script contains the functions that you will need to process your NGS data



from Bio import SeqIO
from Bio.Seq import Seq
from Bio import BiopythonWarning
import gzip
import math
from itertools import combinations, product
import logomaker
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pandas as pd
import pickle as pk
import seaborn as sns
import sys
import time
import threading
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
greyDark = '\033[38;5;102m'
purple = '\033[38;2;189;22;255m'
magenta = '\033[38;2;255;0;128m'
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



# =================================== Define Functions ===================================
def filePaths(enzyme):
    if enzyme == 'ELN' or enzyme == 'hNE':
        inFileNamesInitialSort = ['ELN-I_S1_L001', 'ELN-I_S1_L002']
        inFileNamesFinalSort = ['ELN-R4_S2_L001', 'ELN-R4_S2_L002']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme == 'IDE Prev':
        inFileNamesInitialSort = ['IDE-S1_L001', 'IDE-S1_L002',
                                  'IDE-S1_L003', 'IDE-S1_L004']
        inFileNamesFinalSort = ['IDE-S2_L001', 'IDE-S2_L002',
                                'IDE-S2_L003', 'IDE-S2_L004']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    elif enzyme == 'IDE':
        inFileNamesInitialSort = ['IDE-I_S3_L001', 'IDE-I_S3_L002']
        inFileNamesFinalSort = ['IDE-F_S5_L001', 'IDE-F_S5_L002']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme == 'Mpro':
        inFileNamesInitialSort = ['Mpro-I_S1_L001', 'Mpro-I_S1_L002',
                                  'Mpro-I_S1_L003', 'Mpro-I_S1_L004']
        inFileNamesFinalSort = ['Mpro-R4_S3_L001', 'Mpro-R4_S3_L002',
                                'Mpro-R4_S3_L003', 'Mpro-R4_S3_L004']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme == 'Mpro2':
        inFileNamesInitialSort = ['Mpro2-I_S1_L001', 'Mpro2-I_S1_L002',
                                  'Mpro2-I_S1_L003', 'Mpro2-I_S1_L004']
        inFileNamesFinalSort = ['Mpro2-R4_S3_L001', 'Mpro2-R4_S3_L002',
                                'Mpro2-R4_S3_L003', 'Mpro2-R4_S3_L004']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme == 'MMP7':
        inFileNamesInitialSort = ['MMP7-I_S3_L001', 'MMP7-I_S3_L002']
        inFileNamesFinalSort = ['MMP7-R4_S4_L001', 'MMP7-R4_S4_L002']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    elif enzyme == 'Fyn':
        inFileNamesInitialSort = ['Fyn-I_S6_L001', 'Fyn-I_S6_L002']
        inFileNamesFinalSort = ['Fyn-F_S1_L001', 'Fyn-F_S1_L002']
        inAAPositions = ['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']
    elif enzyme == 'Src':
        inFileNamesInitialSort = ['Src-I_S4_L001', 'Src-I_S4_L002']
        inFileNamesFinalSort = ['Src-F_S2_L001', 'Src-F_S2_L002']
        inAAPositions = ['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']
    elif enzyme == 'VEEV':
        inFileNamesInitialSort = ['VE-I_S1_L001']
        inFileNamesFinalSort = ['VE-R4_S2_L001']
        inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']
    else:
        print(f'{orange}ERROR: There are no file names for {cyan}{enzyme}{orange}\n'
              f'       Add information to the "filePaths" function in '
              f'{os.path.basename(__file__)}\n')
        sys.exit()

    return inFileNamesInitialSort, inFileNamesFinalSort, inAAPositions



def pressKey(event):
    if event.key == 'escape':
        plt.close()



def includeCommas(x):
    return f'{x:,}'



class NGS:
    def __init__(self, enzymeName, substrateLength, fixedAA, fixedPosition, excludeAAs,
                 excludeAA, excludePosition, minCounts, figEMSquares, xAxisLabels,
                 xAxisLabelsBinned, residueLabelType, printNumber, showNValues, findMotif,
                 filePath, filesInit, filesFinal, saveFigures, setFigureTimer):
        self.enzymeName = enzymeName
        self.fixedAA = fixedAA
        self.fixedPosition = fixedPosition
        self.excludeAAs = excludeAAs
        self.excludeAA = excludeAA
        self.excludePosition = excludePosition
        self.minSubCount = minCounts
        self.fixedSubSeq = None
        self.substrateLength = substrateLength
        self.figEMSquares = figEMSquares
        if figEMSquares:
            self.figSizeEM = (5, 8) # (width, height)
        else:
            self.figSizeEM = (9.5, 8)
        self.figSize = (9.5, 8)
        self.figSizeMini = (self.figSize[0], 6)
        self.xAxisLabels = xAxisLabels
        self.xAxisLabelsBinned = xAxisLabelsBinned
        self.residueLabelType = residueLabelType
        self.labelSizeTitle = 18
        self.labelSizeAxis = 16
        self.labelSizeTicks = 13
        self.lineThickness = 1.5
        self.tickLength = 4
        self.residues = defaultResidues
        self.letters = [residue[2] for residue in self.residues]
        self.colorsAA = NGS.residueColors(self)
        if printNumber is None:
            self.printNumber = 0
        else:
            self.printNumber = printNumber # Define number of translated seq to print
        self.fileSize = []
        self.countExtractedSubs = []
        self.percentUnusableDNASeqs = []
        self.showSampleSize = showNValues
        self.nSubsInitial = 0
        self.nSubsFinal = 0
        self.delta = 0
        self.findMotif = findMotif



        self.filesInit = filesInit
        self.filesFinal = filesFinal
        self.saveFigures = saveFigures
        self.pathFolder = filePath
        self.pathSaveData = os.path.join(self.pathFolder, 'Data')
        self.pathSaveFigs = os.path.join(self.pathFolder, 'Figures')
        self.pathFilteredSubs = None
        self.pathFilteredCounts = None



        self.setFigureTimer = setFigureTimer
        self.figureTimerDuration = 0.5
        self.saveFigureIteration = 0
        self.figureResolution = 300
        self.selectedSubstrates = []
        self.selectedDatapoints = []
        self.rectangles = []

        np.set_printoptions(suppress=True) # Prevent data from printing in sci notation
        np.seterr(divide='ignore')

        # Verify directory paths
        if not os.path.exists(self.pathFolder):
            print(f'{orange}ERROR: Folder not found\n'
                  f'Check input: "{cyan}inPathFolder{orange}"\n'
                  f'     inPathFolder = {self.pathFolder}')
            sys.exit()
        if self.pathSaveData is not None:
            if os.path.exists(self.pathSaveData):
                os.makedirs(self.pathSaveData, exist_ok=True)
        if self.pathSaveFigs is not None:
            if os.path.exists(self.pathSaveFigs):
                os.makedirs(self.pathSaveFigs, exist_ok=True)



    def alert(self, soundPath):
        # This function can be used to play .mp3 files
        # Used it to let you know when a process has been completed
        from playsound import playsound

        if os.path.exists(soundPath):
            threading.Thread(target=playsound, args=(soundPath,)).start()
        else:
            print(f'{orange}ERROR: The alerts sound was not found at:\n'
                  f'     {soundPath}{resetColor}\n')



    def residueColors(self):
        color = ['darkgreen', 'firebrick', 'deepskyblue', 'pink', 'navy', 'black', 'gold']
                 # Aliphatic, Acidic, Basic, Hydroxyl, Amide, Aromatic, Sulfur

        colorsAA = {
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

        return colorsAA



    def loadAndTranslate(self, filePath, fileName, fileType, fixedSubs,
                         startSeq, endSeq, printQS):
        # Define file location
        if '/' in filePath:
            fileLocation = filePath + '/' + fileName + '.' + fileType
        else:
            fileLocation = filePath + '\\' + fileName + '.' + fileType

        # Determine read direction
        if '_R1_' in fileName:
            # Extract the substrates
            subSequence = NGS.translate(self, path=fileLocation, type=fileType,
                                        fixData=fixedSubs, startSeq=startSeq,
                                        endSeq=endSeq, printQS=printQS)
        elif '_R2_' in fileName:
            # Extract the substrates
            subSequence = NGS.reverseTranslate(self, path=fileLocation, type=fileType,
                                               fixData=fixedSubs, startSeq=startSeq,
                                               endSeq=endSeq, printQS=printQS)

        return subSequence



    def translate(self, path, type, fixData, startSeq, endSeq, printQS):
        print('============================ Translate: Forward Read '
              '============================')
        subSequence = {}
        totalSeqsDNA = 0

        # Evaluate file path
        gZipped = False
        if not os.path.isfile(path):
            pathZipped = path + '.gz'
            if os.path.isfile(pathZipped):
                gZipped = True
                path = pathZipped
            else:
                print(f'{orange}ERROR: File location does not lead to a file\n'
                      f'     {path}\n'
                      f'     {pathZipped}')
                sys.exit()
        print(f'File Location:\n'
              f'     {path}\n\n')


        # Define fixed substrate tag
        if fixData:
            fixedSubSeq = NGS.genDatasetTag(self)
            print(f'Evaluating fixed Library:{purple} {fixedSubSeq}{resetColor}\n')


        # Load the data & extract the substrates
        if gZipped:
            # Open the file
            with (gzip.open(path, 'rt', encoding='utf-8') as file):
                data = SeqIO.parse(file, type)
                warnings.simplefilter('ignore', BiopythonWarning)

                # Print expressed DNA sequences
                printedSeqs = 0
                for datapoint in data:
                    if printedSeqs == self.printNumber:
                        if fixData:
                            print(f'\nNote: The displayed substrates were not selected for fixed'
                                  f'{purple} {fixedSubSeq}{resetColor}\n'
                                  f'      However the extracted substrates will meet this '
                                  f'restriction\n')

                        throwaway = totalSeqsDNA - self.printNumber
                        throwawayPercent = (throwaway / totalSeqsDNA) * 100
                        print(f'\nExtracting {self.printNumber} Substrates\n'
                              f'     Number of discarded sequences until {red}'
                              f'{self.printNumber} substrates{resetColor} were found in'
                              f'{purple} R1{resetColor}: {red}{throwaway:,}{resetColor}\n'
                              f'     {greyDark}Percent throwaway{resetColor}:'
                              f'{red} {round(throwawayPercent, 3)} %{resetColor}\n\n')
                        break

                    # Select full DNA seq
                    DNA = str(datapoint.seq)
                    totalSeqsDNA += 1
                    if printQS:
                        quality = datapoint.letter_annotations["phred_quality"]
                        print(f'DNA sequence: {DNA}\n     QS - Forward: {quality}')
                    else:
                        print(f'DNA sequence: {DNA}')

                    # Inspect full DNA seq
                    if startSeq in DNA and endSeq in DNA:
                        start = DNA.find(startSeq) + len(startSeq)
                        end = DNA.find(endSeq) # Find the substrate end index

                        # Extract substrate DNA seq
                        substrate = DNA[start:end].strip() 
                        print(f'     Inspected substrate: '
                              f'{greenLightB}{substrate}{resetColor}')
                        if len(substrate) == self.substrateLength * 3:
                            # Express substrate
                            substrate = str(Seq.translate(substrate))
                            print(f'     Inspected Substrate:'
                                  f'{greenLightB} {substrate}{resetColor}')

                            # Inspect substrate seq: PRINT ONLY
                            if 'X' not in substrate and '*' not in substrate:
                                print(f'     Extracted substrate: '
                                      f'{pink}{substrate}{resetColor}')
                                printedSeqs += 1
                # Collect expressed DNA sequences
                totalSeqsDNA = 0
                if fixData:
                    for datapoint in data:
                        # Select full DNA seq
                        DNA = str(datapoint.seq)
                        totalSeqsDNA += 1

                        # Inspect full DNA seq
                        if startSeq in DNA and endSeq in DNA:
                            # Find beginning & end indices for the substrate
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)

                            # Extract substrate DNA seq
                            substrate = DNA[start:end].strip() 
                            if len(substrate) == self.substrateLength * 3:
                                # Express substrate
                                substrate = str(Seq.translate(substrate))

                                # Inspect substrate seq: Keep good fixed datapoints
                                if 'X' not in substrate and '*' not in substrate:
                                    if len(self.fixedAA[0]) == 1:
                                        if substrate[self.fixedPosition[0] - 1] \
                                            in self.fixedAA:
                                            if substrate in subSequence:
                                                subSequence[substrate] += 1
                                            else:
                                                subSequence[substrate] = 1
                                    else:
                                        if substrate[self.fixedPosition[0] - 1] \
                                            in self.fixedAA[0]:
                                            if substrate in subSequence:
                                                subSequence[substrate] += 1
                                            else:
                                                subSequence[substrate] = 1
                else:
                    for datapoint in data:
                        # Select full DNA seq
                        DNA = str(datapoint.seq)
                        totalSeqsDNA += 1

                        # Inspect full DNA seq
                        if startSeq in DNA and endSeq in DNA:
                            # Find begining & end indices for the substrate
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)  # Find the substate end index

                            # Extract substrate DNA seq
                            substrate = DNA[start:end].strip()
                            if len(substrate) == self.substrateLength * 3:
                                substrate = str(Seq.translate(substrate))

                                # Inspect substrate seq: Keep good datapoints
                                if 'X' not in substrate and '*' not in substrate:
                                    # if len(substrate) == self.substrateLength:
                                    if substrate in subSequence:
                                        subSequence[substrate] += 1
                                    else:
                                        subSequence[substrate] = 1
        else:
            # Open the file
            with open(path, 'r') as file:
                data = SeqIO.parse(file, type)
                warnings.simplefilter('ignore', BiopythonWarning)

                # Print expressed DNA sequences
                printedSeqs = 0
                for datapoint in data:
                    if printedSeqs == self.printNumber:
                        throwaway = totalSeqsDNA - self.printNumber
                        throwawayPercent = (throwaway / totalSeqsDNA) * 100
                        print(f'\nExtracting {self.printNumber} Substrates\n'
                              f'     Number of discarded sequences until {red}'
                              f'{self.printNumber} substrates{resetColor} were found in'
                              f'{purple} R1{resetColor}: {red}{throwaway:,}{resetColor}\n'
                              f'     {greyDark}Percent throwaway{resetColor}:'
                              f'{red} {round(throwawayPercent, 3)} %{resetColor}\n\n')
                        break

                    # Select full DNA seq
                    DNA = str(datapoint.seq)
                    totalSeqsDNA += 1

                    if printQS:
                        quality = datapoint.letter_annotations["phred_quality"]
                        print(f'DNA sequence: {DNA}\n     QS - Forward: {quality}')
                    else:
                        # print(f'Expressed DNA sequence: {AA}')
                        print(f'DNA sequence: {DNA}')

                    # Inspect full DNA seq
                    if startSeq in DNA and endSeq in DNA:
                        # Find begining & end indices for the substrate
                        start = DNA.find(startSeq) + len(startSeq)
                        end = DNA.find(endSeq)

                        # Extract substrate DNA seq
                        substrate = DNA[start:end].strip()
                        print(f'     Inspected substrate: '
                              f'{greenLightB}{substrate}{resetColor}')
                        if len(substrate) == self.substrateLength * 3:
                            # Express substrate
                            substrate = str(Seq.translate(substrate))
                            print(f'     Inspected Substrate:'
                                  f'{greenLightB} {substrate}{resetColor}')

                            # Inspect substrate seq: PRINT ONLY
                            if 'X' not in substrate and '*' not in substrate:
                                print(f'     Extracted substrate: '
                                      f'{pink}{substrate}{resetColor}')
                                printedSeqs += 1
                # Collect expressed DNA sequences
                totalSeqsDNA = 0
                if fixData:
                    for datapoint in data:
                        # Select full DNA seq
                        DNA = str(datapoint.seq)
                        totalSeqsDNA += 1

                        # Inspect full DNA seq
                        if startSeq in DNA and endSeq in DNA:
                            # Find begining & end indices for the substrate
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)

                            # Extract substrate DNA seq
                            substrate = DNA[start:end].strip() 
                            if len(substrate) == self.substrateLength * 3:
                                # Express substrate
                                substrate = str(Seq.translate(substrate))

                                # Inspect substrate seq: Keep good fixed datapoints
                                if 'X' not in substrate and '*' not in substrate:
                                    if len(self.fixedAA[0]) == 1:
                                        if substrate[self.fixedPosition[0] - 1] \
                                            in self.fixedAA:
                                            if substrate in subSequence:
                                                subSequence[substrate] += 1
                                            else:
                                                subSequence[substrate] = 1
                                    else:
                                        if substrate[
                                            self.fixedPosition[0] - 1] \
                                            in self.fixedAA[0]:
                                            if substrate in subSequence:
                                                subSequence[substrate] += 1
                                            else:
                                                subSequence[substrate] = 1
                else:
                    for datapoint in data:
                        # Select full DNA seq
                        DNA = str(datapoint.seq)
                        totalSeqsDNA += 1

                        # Inspect full DNA seq
                        if startSeq in DNA and endSeq in DNA:
                            # Find begining & end indices for the substrate
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)

                            # Extract substrate DNA seq
                            substrate = DNA[start:end].strip() 
                            if len(substrate) == self.substrateLength * 3:
                                # Express substrate
                                substrate = str(Seq.translate(substrate))

                                # Inspect substrate seq: Keep good datapoints
                                if 'X' not in substrate and '*' not in substrate:
                                    if substrate in subSequence:
                                        subSequence[substrate] += 1
                                    else:
                                        subSequence[substrate] = 1
        # Verify if substrates have been extracted
        if len(subSequence) == 0:
            print(f'\nNo substrates were extracted from file at:\n{path}\n\n'
                  f'Recommend: adjust variables\n'
                  f'     startSeq: {red}{startSeq}{resetColor}\n'
                  f'     endSeq: {red}{endSeq}{resetColor}')
            sys.exit()
        else:
            extractionCount = sum(subSequence.values())
            throwaway = (totalSeqsDNA - extractionCount)
            throwawayPercent = (throwaway / totalSeqsDNA) * 100
            self.fileSize.append(totalSeqsDNA)
            self.countExtractedSubs.append(extractionCount)
            self.percentUnusableDNASeqs.append(throwawayPercent)
            print(f'Evaluate All DNA Sequences in{purple} R1{resetColor}:\n'
                  f'     Total DNA sequences in the file: '
                  f'{red}{totalSeqsDNA:,}{resetColor}\n'
                  f'     Number of extracted Substrates: '
                  f'{red}{extractionCount:,}{resetColor}\n'
                  f'     {greyDark}Percent throwaway{resetColor}:'
                  f'{red} {round(throwawayPercent, 3)} %{resetColor}\n\n')
        # sys.exit()

        # Rank the substrates
        subSequence = dict(sorted(subSequence.items(), key=lambda x: x[1], reverse=True))

        return subSequence



    def reverseTranslate(self, path, type, fixData, startSeq, endSeq, printQS):
        print('============================ Translate: Reverse Read '
              '============================')
        subSequence = {}
        totalSeqsDNA = 0

        # Evaluate file path
        gZipped = False
        if not os.path.isfile(path):
            pathZipped = path + '.gz'
            if not os.path.isfile(pathZipped):
                print(f'{orange}ERROR: File location does not lead to a file'
                      f'\n     {path}\n     {pathZipped}')
                sys.exit()
            else:
                gZipped = True
                path = pathZipped
        print(f'File Location:\n     {path}\n\n')


        # Define fixed substrate tag
        if fixData:
            fixedSubSeq = NGS.genDatasetTag(self)
            print(f'Evaluating fixed Library:{purple} {fixedSubSeq}{resetColor}\n')


        if gZipped:
            with gzip.open(path, 'rt', encoding='utf-8') as file: # Open the file
                data = SeqIO.parse(file, type)
                warnings.simplefilter('ignore', BiopythonWarning)

                # Print expressed DNA sequences
                printedSeqs = 0
                for datapoint in data:
                    if printedSeqs == self.printNumber:
                        if fixData:
                            print(f'\nNote: The displayed substrates were not selected '
                                  f'for fixed{purple} {fixedSubSeq}{resetColor}\n'
                                  f'      However the extracted substrates will meet '
                                  f'this restriction\n')

                        throwaway = totalSeqsDNA - self.printNumber
                        throwawayPercent = (throwaway / totalSeqsDNA) * 100
                        print(f'\nExtracting {self.printNumber} Substrates\n'
                              f'     Number of discarded sequences until {red}'
                              f'{self.printNumber} substrates{resetColor} were found in'
                              f'{purple} R2{resetColor}: {red}{throwaway:,}{resetColor}\n'
                              f'     {greyDark}Percent throwaway{resetColor}:'
                              f'{red} {round(throwawayPercent, 3)} %{resetColor}\n\n')
                        break

                    # Select full DNA seq
                    DNA = str(datapoint.seq)
                    DNA = Seq(DNA).reverse_complement()
                    totalSeqsDNA += 1
                    if printQS:
                        quality = datapoint.letter_annotations["phred_quality"]
                        print(f'DNA sequence: {DNA}\n     QS - Forward: {quality}')
                    else:
                        # print(f'Expressed DNA sequence: {AA}')
                        print(f'DNA sequence: {DNA}')

                    # Inspect full DNA seq
                    if startSeq in DNA and endSeq in DNA:
                        # Find begining & end indices for the substrate
                        start = DNA.find(startSeq) + len(startSeq)
                        end = DNA.find(endSeq)

                        # Extract substrate DNA seq
                        substrate = DNA[start:end].strip()
                        print(f'     Inspected substrate: '
                              f'{greenLightB}{substrate}{resetColor}')
                        if len(substrate) == self.substrateLength * 3:
                            # Express substrate
                            substrate = str(Seq.translate(substrate))
                            print(f'     Inspected Substrate:'
                                  f'{greenLightB} {substrate}{resetColor}')

                            # Inspect substrate seq: PRINT ONLY
                            if 'X' not in substrate and '*' not in substrate:
                                print(f'     Extracted substrate: '
                                      f'{pink}{substrate}{resetColor}')
                                printedSeqs += 1
                # Collect expressed DNA sequences
                totalSeqsDNA = 0
                if fixData:
                    for datapoint in data:
                        # Select full DNA seq
                        DNA = str(datapoint.seq)
                        DNA = Seq(DNA).reverse_complement()
                        totalSeqsDNA += 1

                        # Inspect full DNA seq
                        if startSeq in DNA:  # and endSeq in DNA:
                            # Find beginning & end indices for the substrate
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)

                            # Extract substrate DNA seq
                            substrate = DNA[start:end].strip()
                            if len(substrate) == self.substrateLength * 3:
                                # Express substrate
                                substrate = str(Seq.translate(substrate))
                                
                                # Inspect substrate seq: Keep good fixed datapoints
                                if 'X' not in substrate and '*' not in substrate:
                                    if len(self.fixedAA[0]) == 1:
                                        if substrate[self.fixedPosition[0] - 1] \
                                            in self.fixedAA:
                                            if substrate in subSequence:
                                                subSequence[substrate] += 1
                                            else:
                                                subSequence[substrate] = 1
                                    else:
                                        if substrate[self.fixedPosition[0] - 1] \
                                            in self.fixedAA[0]:
                                            if substrate in subSequence:
                                                subSequence[substrate] += 1
                                            else:
                                                subSequence[substrate] = 1
                else:
                    for datapoint in data:
                        # Select full DNA seq
                        DNA = str(datapoint.seq)
                        DNA = Seq(DNA).reverse_complement()
                        totalSeqsDNA += 1

                        # Inspect full DNA seq
                        if startSeq in DNA:  # and endSeq in DNA:
                            # Find begining & end indices for the substrate
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)

                            # Extract substrate DNA seq
                            substrate = DNA[start:end].strip()
                            if len(substrate) == self.substrateLength * 3:
                                # Express substrate
                                substrate = str(Seq.translate(substrate))

                                # Inspect substrate seq
                                if 'X' not in substrate and '*' not in substrate:
                                    if substrate in subSequence:
                                        subSequence[substrate] += 1
                                    else:
                                        subSequence[substrate] = 1
        else:
            with open(path, 'r') as file: # Open the file
                data = SeqIO.parse(file, type)
                warnings.simplefilter('ignore', BiopythonWarning)

                # Print expressed DNA sequences
                printedSeqs = 0
                for datapoint in data:
                    if printedSeqs == self.printNumber:
                        throwaway = totalSeqsDNA - self.printNumber
                        throwawayPercent = (throwaway / totalSeqsDNA) * 100
                        print(f'\nExtracting {self.printNumber} Substrates\n'
                              f'     Number of discarded sequences until {red}'
                              f'{self.printNumber} substrates{resetColor} were found in'
                              f'{purple} R2{resetColor}: {red}{throwaway:,}{resetColor}\n'
                              f'     {greyDark}Percent throwaway{resetColor}:'
                              f'{red} {round(throwawayPercent, 3)} %{resetColor}\n\n')
                        break

                    # Select full DNA seq
                    DNA = str(datapoint.seq)
                    DNA = Seq(DNA).reverse_complement()
                    totalSeqsDNA += 1
                    if printQS:
                        quality = datapoint.letter_annotations["phred_quality"]
                        print(f'DNA sequence: {DNA}\n     QS - Forward: {quality}')
                    else:
                        # print(f'Expressed DNA sequence: {AA}')
                        print(f'DNA sequence: {DNA}')

                    # Inspect full DNA seq
                    if startSeq in DNA and endSeq in DNA:
                        # Find begining & end indices for the substrate
                        start = DNA.find(startSeq) + len(startSeq)
                        end = DNA.find(endSeq)

                        # Extract substrate DNA seq
                        substrate = DNA[start:end].strip()
                        print(f'     Inspected substrate: '
                              f'{greenLightB}{substrate}{resetColor}')
                        if len(substrate) == self.substrateLength * 3:
                            # Express substrate
                            substrate = str(Seq.translate(substrate))
                            print(f'     Inspected Substrate:'
                                  f'{greenLightB} {substrate}{resetColor}')

                            # Inspect substrate seq: PRINT ONLY
                            if 'X' not in substrate and '*' not in substrate:
                                print(f'     Extracted substrate: '
                                      f'{pink}{substrate}{resetColor}')
                                printedSeqs += 1
                # Collect expressed DNA sequences
                totalSeqsDNA = 0
                if fixData:
                    for datapoint in data:
                        # Select full DNA seq
                        DNA = str(datapoint.seq)
                        DNA = Seq(DNA).reverse_complement()
                        totalSeqsDNA += 1

                        # Inspect full DNA seq
                        if startSeq in DNA and endSeq in DNA:
                            # Find begining & end indices for the substrate
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)
                            
                            # Extract substrate DNA seq
                            substrate = DNA[start:end].strip() 
                            if len(substrate) == self.substrateLength * 3:
                                # Express substrate
                                substrate = str(Seq.translate(substrate))

                                # Inspect substrate seq: Keep good fixed datapoints
                                if 'X' not in substrate and '*' not in substrate:
                                    if len(self.fixedAA[0]) == 1:
                                        if substrate[self.fixedPosition[0] - 1] \
                                            in self.fixedAA:
                                            if substrate in subSequence:
                                                subSequence[substrate] += 1
                                            else:
                                                subSequence[substrate] = 1
                                    else:
                                        if substrate[self.fixedPosition[0] - 1] \
                                            in self.fixedAA[0]:
                                            if substrate in subSequence:
                                                subSequence[substrate] += 1
                                            else:
                                                subSequence[substrate] = 1
                else:
                    for datapoint in data:
                        # Select full DNA seq
                        DNA = str(datapoint.seq)
                        DNA = Seq(DNA).reverse_complement()
                        totalSeqsDNA += 1

                        # Inspect full DNA seq
                        if startSeq in DNA and endSeq in DNA:
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)

                            # Extract substrate DNA seq
                            substrate = DNA[start:end].strip()  
                            if len(substrate) == self.substrateLength * 3:
                                # Express substrate
                                substrate = str(Seq.translate(substrate))

                                # Inspect substrate seq
                                if 'X' not in substrate and '*' not in substrate:
                                    if substrate in subSequence:
                                        subSequence[substrate] += 1
                                    else:
                                        subSequence[substrate] = 1


        # Verify if substrates have been extracted
        if len(subSequence) == 0:
            print(f'\nNo substrates were extracted from file at:\n{path}\n\n'
                  f'Recommend: adjust variables\n'
                  f'     startSeq: {red}{startSeq}{resetColor}\n'
                  f'     endSeq: {red}{endSeq}{resetColor}')
            sys.exit()
        else:
            extractionCount = sum(subSequence.values())
            throwaway = (totalSeqsDNA - extractionCount)
            throwawayPercent = (throwaway / totalSeqsDNA) * 100
            self.fileSize.append(totalSeqsDNA)
            self.countExtractedSubs.append(extractionCount)
            self.percentUnusableDNASeqs.append(throwawayPercent)
            print(f'Evaluate All DNA Sequences in{purple} R2{resetColor}:\n'
                  f'     Total DNA sequences in the file: '
                  f'{red}{totalSeqsDNA:,}{resetColor}\n'
                  f'     Number of extracted Substrates: '
                  f'{red}{extractionCount:,}{resetColor}\n'
                  f'     {greyDark}Percent throwaway{resetColor}:'
                  f'{red} {round(throwawayPercent, 3)} %{resetColor}\n\n')


        # Rank the substrates
        subSequence = dict(sorted(subSequence.items(), key=lambda x: x[1], reverse=True))

        return subSequence



    def extractionEffiency(self, files):
        print('======================== Substrate Extraction Efficiency '
              '========================')
        for index, file in enumerate(files):
            print(f'{pink}Evaluate file{resetColor}:{yellow} {file}{resetColor}\n'
                  f'     Total DNA sequences in the file: '
                  f'{red}{self.fileSize[index]:,}{resetColor}\n'
                  f'     Number of extracted Substrates: '
                  f'{red}{self.countExtractedSubs[index]:,}{resetColor}\n'
                  f'     {greenLight}Percent throwaway{resetColor}: '
                  f'{red}{round(self.percentUnusableDNASeqs[index], 3)} %{resetColor}\n')
        print('')



    def countResidues(self, substrates, datasetType):
        beginTime = time.time()
        print('============================= Calculate: AA Counts '
              '==============================')
        print(f'Unique substrate count:{red} {len(substrates):,}{resetColor}')

        # Initialize the count matrix
        countedData = pd.DataFrame(0,
                                   index=self.letters,
                                   columns=self.xAxisLabels,
                                   dtype=int)

        # Determine substrate lenght
        for substrate in substrates.keys():
            firstSub = substrate
            lengthFirstSub = len(substrate)
            if lengthFirstSub != len(self.xAxisLabels):
                binnedSubs = True
            else:
                binnedSubs = False
            break

        # Verfy consistant substrate lenghts
        countModulus = 100 # Check the Nth substrate length
        for index, substrate in enumerate(substrates.keys()):
            if index % countModulus == 0:
                # print(f'Check:{greenLightB} {index}')
                if len(substrate) != lengthFirstSub:
                    print(f'{orange}ERROR: The substrate lengths do not match\n'
                          f'     {firstSub}: {lengthFirstSub} AA\n'
                          f'     {substrate}: {len(substrate)} AA\n\n')
                    sys.exit()
        # print(f'{resetColor}')


        # Count the occurrences of each residue
        if binnedSubs:
            # Initialize the counts matrix
            countedData = pd.DataFrame(0,
                                       index=self.letters,
                                       columns=self.xAxisLabelsBinned,
                                       dtype=int)
            # Count the AAs
            for substrate, counts in substrates.items():
                indicesResidue = [self.letters.index(AA) for AA in substrate]
                for position, indexResidue in enumerate(indicesResidue):
                    countedData.iloc[indexResidue, position] += counts

            # Sum all columns
            columnSums = pd.DataFrame(np.sum(countedData, axis=0),
                                      columns=['Total Counts'],
                                      index=self.xAxisLabelsBinned)
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

        # Print counts
        columnSumsFormated = columnSums.apply(lambda col: col.map(includeCommas)).copy()
        print(f'{greyDark}{columnSumsFormated}{resetColor}\n')
        countedDataPrint = countedData.apply(lambda col: col.map(includeCommas))
        print(f'Counted data:{purple} {self.enzymeName}{resetColor}\n'
              f'{greyDark}{countedDataPrint}{resetColor}\n\n')


        # Sanity Check: Do the sums of each column match the total number of substrates?
        totalSubs = sum(countedData.iloc[:, 0])
        for indexColumn in countedData.columns:
            columnSum = sum(countedData.loc[:, indexColumn])
            if columnSum != totalSubs:
                print(
                    f'Counted data:{purple} {self.enzymeName}{resetColor}\n'
                    f'{countedData}\n\n'
                    f'{orange}ERROR: The total number of substrates '
                    f'({red}{totalSubs:,}{orange}) =/= the sum of column '
                    f'{indexColumn} ({red}{columnSum:,}{orange}){resetColor}\n')
                sys.exit()


        print(f'Total substrates: {red}{totalSubs:,}{resetColor}\n'
              f'Total Unique Substrates:{red} {len(substrates):,}{resetColor}\n\n')

        # Updata sample size
        if datasetType == 'Initial Sort':
            self.nSubsInitial = totalSubs
        else:
            self.nSubsFinal = totalSubs

        stopTime = time.time()
        endTime = stopTime - beginTime
        print(f'Runtime:{greenLightB} Count AA\n'
              f'     {red}{np.round(endTime, 3):,}s{resetColor}\n\n')

        return countedData, totalSubs


    def countResiduesBinned(self, substrates, positions, printCounts):
        print('============================= Calculate: AA Counts '
              '==============================')
        totalSubs = np.sum(list(substrates.values()))
        totalUniqueSubs = len(substrates)
        print(f'Total Substrates:  {red}{totalSubs:,}{resetColor}\n'
              f'Unique Substrates: {red}{totalUniqueSubs:,}{resetColor}\n')

        # Initialize the count matrix
        countedData = pd.DataFrame(0, index=self.letters, columns=positions,
                                   dtype=int)

        # Count the occurrences of each residue
        substrateLength = len(positions)
        for substrate, counts in substrates.items():
            for indexAA in range(substrateLength):
                if substrate[indexAA] in self.letters:
                    indexResidue = self.letters.index(substrate[indexAA])
                    countedData.iloc[indexResidue, indexAA] += counts
                else:
                    print(f'{greyDark}Outlying Substrate: {substrate}\n'
                          f'     Unexpected Residue: {substrate[indexAA]}\n'
                          f'     Position: {indexAA}{resetColor}\n')


        # Sanity Check: Do the sums of each column match the total number of substrates?
        columnSums = pd.DataFrame(np.sum(countedData, axis=0),
                                  columns=['Total AA Counts'],
                                  index=positions)
        for indexColumn in countedData.columns:
            sum = np.sum(countedData.loc[:, indexColumn])
            if sum != totalSubs:
                print(f'Counted data:{purple} '
                      f'{self.enzymeName}{resetColor}\n{countedData}\n\n'
                      f'{orange}ERROR: The total number of substrates '
                      f'({red}{totalSubs:,}{orange}) =/= the '
                      f'sum of column {indexColumn} ({red}{sum:,}{resetColor})\n')
                sys.exit()


        if printCounts:
            countedDataPrint = countedData.apply(lambda col: col.map(includeCommas))
            print(f'Counted data:{purple} {self.enzymeName}{resetColor}\n'
                  f'{greyDark}{countedDataPrint}{resetColor}\n\n')


        columnSums = columnSums.apply(lambda col: col.map(includeCommas))
        print(f'{columnSums}')
        print(f'\nTotal number of substrates: {red}{totalSubs:,}{resetColor}\n\n')

        return countedData, totalSubs


    def sampleSizeDisplay(self, sortType, datasetTag):
        print('============================== Current Sample Size '
              '==============================')
        if sortType is not None:
            if datasetTag is None:
                print(f'Sort Type:{purple} {sortType}{resetColor}')
            else:
                print(f'Sort Type:{purple} {sortType} {datasetTag}{resetColor}')

        totalSubs = self.nSubsInitial + self.nSubsFinal
        print(f'Initial Sort:{red} {self.nSubsInitial:,}{resetColor}\n'
              f'Final Sort:{red} {self.nSubsFinal:,}{resetColor}\n'
              f'Total Substrates:{greenLight} {totalSubs:,}{resetColor}\n\n')


    def sampleSizeUpdate(self, NSubs, sortType, datasetTag):
        # Update the current sample size
        if sortType == 'Initial Sort':
            self.nSubsInitial = NSubs
        elif 'Final' in sortType:
            self.nSubsFinal = NSubs
        else:
            print('============================== Current Sample Size '
                  '==============================')
            if datasetTag is None:
                print(f'Sort Type:{purple} {sortType}{resetColor}')
            else:
                print(f'Sort Type:{purple} {sortType} {datasetTag}{resetColor}')
            print(f'{orange}ERROR: The sample sizes were not updated\n\n')
            sys.exit()

        NGS.sampleSizeDisplay(self, sortType, datasetTag)



    def getFilePath(self, datasetTag, motifPath=False):
        # Define: File path
        if motifPath:
            pathSubs = os.path.join(
                self.pathSaveData,
                f'fixedMotifSubs - {self.enzymeName} - {datasetTag} - '
                f'FinalSort - MinCounts {self.minSubCount}')
            pathCounts = os.path.join(
                self.pathSaveData,
                f'fixedMotifCounts - {self.enzymeName} - {datasetTag} - '
                f'FinalSort - MinCounts {self.minSubCount}')
            pathCountsReleased = os.path.join(
                self.pathSaveData,
                f'fixedMotifCountsRel - {self.enzymeName} - {datasetTag} - '
                f'FinalSort - MinCounts {self.minSubCount}')
            paths = [pathSubs, pathCounts, pathCountsReleased]
        else:
            pathSubs = os.path.join(
                self.pathSaveData,
                f'fixedSubs - {self.enzymeName} - {datasetTag} - '
                f'FinalSort - MinCounts {self.minSubCount}')
            pathCounts = os.path.join(
                self.pathSaveData,
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
        print(f'Loading Counts:{purple} {labelFile}{resetColor}\n')
        files = []
        totalCounts = 0

        # Define: File paths
        if filter:
            if self.pathFilteredCounts is None:
                print(f'{orange}ERROR: {cyan}self.pathFilteredCounts {orange}needs '
                      f'to be defined before you can load the counts.{resetColor}')
                sys.exit()
            files = [self.pathFilteredCounts]
        else:
            if 'initial' in fileType.lower():
                fileNames = self.filesInit
            else:
                fileNames = self.filesFinal

            for fileName in fileNames:
                files.append(os.path.join(self.pathSaveData, f'counts_{fileName}'))

        print(f'Loading data:')
        for filePath in files:
            # Verify if the file exists at its specified path
            if not os.path.exists(filePath):
                print(f'{orange}ERROR: File not found\n'
                      f'     {filePath}')
                sys.exit()
            print(f'     {greenDark}{filePath}{resetColor}')
        print('')


        #  Load: AA counts
        firstFile = True
        for index, filePath in enumerate(files):
            fileName = filePath.replace(self.pathSaveData, '')
            if firstFile:
                firstFile = False
                # Load: File
                countedData = pd.read_csv(filePath, index_col=0)
                countedData = countedData.astype(int) # Convert datapoints to integers

                # Format values to have commas
                formattedCounts = countedData.to_string(
                    formatters={column: '{:,.0f}'.format for column in
                                countedData.select_dtypes(
                                    include='number').columns})

                substrateCounts = sum(countedData.iloc[:, 0])
                totalCounts += substrateCounts
                print(f'\nCounts: {greenLightB}{fileName}{resetColor}\n'
                      f'{formattedCounts}\n'
                      f'Substrate Count: {red}{substrateCounts:,}{resetColor}\n')
            else:
                # Load: File
                data = pd.read_csv(filePath, index_col=0)
                data = data.astype(int)  # Convert datapoints to integers

                # Format values to have commas
                formattedCounts = data.to_string(
                    formatters={column: '{:,.0f}'.format for column in
                                data.select_dtypes(include='number').columns})

                substrateCounts = sum(data.iloc[:, 0])
                totalCounts += substrateCounts
                print(f'\nCounts: {greenLightB}{fileName}{resetColor}\n'
                      f'{formattedCounts}\n'
                      f'Substrate Count: {red}'
                      f'{substrateCounts:,}{resetColor}\n\n')

                countedData += data
        print(f'Number of substrates in{purple} {labelFile}{resetColor}: '
              f'{red}{totalCounts:,}{resetColor}\n')

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
                      f'({cyan}{columnSum:,}{orange})')
                sys.exit()
        print('\n')

        # Undate sample size
        if 'initial' in fileType.lower():
            self.nSubsInitial = totalCounts
        elif 'final' in fileType.lower():
            self.nSubsFinal = totalCounts
        else:
            print(f'{orange}ERROR: Unknown fileType "{cyan}{fileType}{orange}"\n'
                  f'     Rename parameter as: "Initial Sort" or "Final Sort"\n')
            sys.exit()

        return countedData, totalCounts



    def loadSubstrates(self, fileNames, fileType):
        print('============================= Load: Substrate Files '
              '=============================')
        substrates = {}
        substrateTotal = 0

        print(f'Loading data in these files:')
        for fileName in fileNames:
            print(f'     {greenLightB}{fileName}{resetColor}')
        print()

        # Function to load each file
        def loadFile(fileName):
            fileLocation = os.path.join(self.pathSaveData, f'substrates_{fileName}')
            print(f'File path:\n     {greenDark}{fileLocation}{resetColor}\n')
            with open(fileLocation, 'rb') as openedFile:  # Open file
                data = pk.load(openedFile)  # Access the data
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

        # Print loaded data substrates
        print(f'Loaded Data:{purple} {fileType}{resetColor}')
        iteration = 0
        for substrate, count in substrates.items():
            iteration += 1
            print(f'     {greyDark}{substrate}{resetColor}, Counts: {red}{count:,}'
                  f'{resetColor}')
            if iteration >= self.printNumber:
                break
        print(f'\nTotal substrates:{purple} {fileType}\n'
              f'     {red} {substrateTotal:,}{resetColor}\n\n')

        return substrates, substrateTotal



    def loadUnfilteredSubs(self, loadInitial=False):
        def loadSubsThread(fileNames, fileType, result):
            subsLoaded, totalSubs = NGS.loadSubstrates(self, fileNames=fileNames,
                                                       fileType=fileType)
            result[fileType] = (subsLoaded, totalSubs)

        # Initialize result dictionary
        loadedResults = {}


        if loadInitial:
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
        else:
            # Create thread for the final substrates
            threadFinal = threading.Thread(target=loadSubsThread,
                                           args=(self.filesFinal, 'Final Sort',
                                                 loadedResults))

            # Start the thread
            threadFinal.start()

            # Wait for the thread to complete
            threadFinal.join()

            # Retrieve the loaded substrates
            substratesFinal, totalSubsFinal = loadedResults['Final Sort']

            return substratesFinal, totalSubsFinal



    def loadFixedMotifCounts(self, filePath, substrateFrame, substrateFrameAAPos,
                             frameIndices, datasetTag, sortType):
        print('=========================== Load: Fixed Motif Counts '
              '============================')
        print(f'Loading counts:{purple} {self.enzymeName} - {datasetTag}{resetColor}\n')

        frameLength = len(substrateFrame)
        countsFixedFrameAll = []

        # Load the counts
        for index, position in enumerate(self.fixedPosition):
            # Define: File paths
            tagFixedAA = f'{self.fixedAA[0]}@R{position}'
            if 'final' in sortType.lower():
                tagLabel = (f'fixedMotifCountsRel - {self.enzymeName} - {tagFixedAA} - '
                            f'FinalSort - MinCounts {self.minSubCount}')
            else:
                tagLabel = (f'fixedMotifCountsRel - {self.enzymeName} - {tagFixedAA} - '
                            f'InitialSort - MinCounts {self.minSubCount}')
            filePathFixedMotifCounts = os.path.join(filePath, tagLabel)

            # Look for the file
            if os.path.exists(filePathFixedMotifCounts):
                # Load file
                countsLoaded = pd.read_csv(filePathFixedMotifCounts, index_col=0)

                # Define fixed frame positions & extract the data
                startPosition = frameIndices[0]
                startSubPrevious = startPosition
                if index != 0:
                    # Evaluate previous fixed frame index
                    fixedPosDifference = (self.fixedPosition[index] -
                                          self.fixedPosition[index - 1])
                    startSubPrevious += fixedPosDifference
                    startSub = index + startSubPrevious - 1
                    endSub = startSub + frameLength
                else:
                    startSub = startPosition
                    endSub = frameIndices[-1] + 1
                fixedFramePos = countsLoaded.columns[startSub:endSub]
                countsFixedFrame = countsLoaded.loc[:, fixedFramePos]
                countsFixedFrame.columns = substrateFrame
                countsFixedFrameAll.append(countsFixedFrame.values)

                formattedCounts = countsFixedFrame.to_string(
                    formatters={column: '{:,.0f}'.format for column in
                                countsFixedFrame.select_dtypes(include='number').columns})
                print(f'Selecting Positions:{purple} Fixed Motif {tagFixedAA}\n'
                      f'     {greenLightB}{fixedFramePos}{resetColor}\n'
                      f'Counts:\n'
                      f'{greenLight}{formattedCounts}{resetColor}\n\n')

                # Track totals
                if index == 0:
                    totalCountsFixedFrame = countsFixedFrame
                else:
                    totalCountsFixedFrame += countsFixedFrame
            else:
                print(f'{orange}ERROR: The file was not found\n'
                      f'     {filePathFixedMotifCounts}\n\n')
                sys.exit()

        # Sum the columns
        fixedFrameColumnSums = []
        for column in substrateFrame:
            fixedFrameColumnSums.append(sum(totalCountsFixedFrame.loc[:, column]))

        # Format the DataFrame with commas
        formattedCounts = totalCountsFixedFrame.to_string(
            formatters={column: '{:,.0f}'.format for column in
                        totalCountsFixedFrame.select_dtypes(include='number').columns})

        # Print the data
        print(f'{greyDark}Combined Counts{resetColor}:{purple} {self.enzymeName} - {datasetTag}{resetColor}\n'
              f'{formattedCounts}\n')
        print('Total Counts:')
        for index, position in enumerate(substrateFrame):
            print(f'     {position}: {red}{fixedFrameColumnSums[index]:,}{resetColor}')
        print('\n')

        # Convert list into 3D array
        countsFixedFrameAll = np.stack(countsFixedFrameAll, axis=0)

        return countsFixedFrameAll, totalCountsFixedFrame



    def loadFixedMotifSubstrates(self, pathLoad, datasetTag):
        print('============================== Load: Fixed Motifs '
              '===============================')
        print(f'Loading substrates at path:\n'
              f'     {greenDark}{pathLoad}{resetColor}\n')

        # Load Data: Fixed substrates
        with open(pathLoad, 'rb') as file:
            loadedSubs = pk.load(file)

        print(f'Loaded Substrates:{purple} Fixed Motif {datasetTag}{resetColor}')
        iteration = 0
        for substrate, count in loadedSubs.items():
            print(f'Substrate:{greyDark} {substrate}{resetColor}\n'
                  f'     Count:{red} {count:,}{resetColor}')
            iteration += 1
            if iteration >= self.printNumber:
                print('\n')
                break

        return loadedSubs



    def calculateProbMotif(self, countsTotal, datasetTag):
        print('====================== Calculate: Fixed Motif Probability '
              '=======================')
        # Sum each column
        columnSums = np.sum(countsTotal, axis=0)
        columnSums = pd.DataFrame(columnSums, index=countsTotal.columns, columns=['Sum'])

        prob = pd.DataFrame(0.0, index=countsTotal.index,
                            columns=countsTotal.columns)

        # Calculate: Relative frequency
        for position in countsTotal.columns:
            prob.loc[:, position] = (countsTotal.loc[:, position] /
                                   columnSums.loc[position, 'Sum'])
        pd.set_option('display.float_format', '{:,.5f}'.format)
        print(f'Relative Frequency:{purple} {datasetTag}\n'
              f'{green}{prob}{resetColor}\n\n')
        pd.set_option('display.float_format', '{:,.3f}'.format)

        print(f'\n{red}EXIT HEERE\n\n')
        sys.exit()

        return prob



    def genDatasetTag(self):
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
                        f'{self.fixedPosition[index]}'.replace(' ', ''))
                else:
                    fixResidueList.append(
                        f'{self.fixedAA[index]}@R'
                        f'{self.fixedPosition[index]}'.replace(' ', ''))

            self.fixedSubSeq = '_'.join(fixResidueList)
            self.fixedSubSeq = self.fixedSubSeq.replace("_Fixed", ' Fixed')
        else:
            # Fix residues
            for index in range(len(self.fixedAA)):
                fixResidueList.append(
                    f'{self.fixedAA[index]}@R'
                    f'{self.fixedPosition[index]}'.replace(' ', ''))

            self.fixedSubSeq = '_'.join(fixResidueList)

        # Condense the string
        if "'" in self.fixedSubSeq:
            self.fixedSubSeq = self.fixedSubSeq.replace("'", '')

        # Clean up fixed sequence tag
        removeTag = ('Excl-Y@R1_Excl-Y@R2_Excl-Y@R3_Excl-Y@R4_Excl-Y@R6_'
                    'Excl-Y@R7_Excl-Y@R8_Excl-Y@R9')
        if removeTag in self.fixedSubSeq:
            self.fixedSubSeq = self.fixedSubSeq.replace(removeTag, '')
            self.fixedSubSeq = f'Excl Y {self.fixedSubSeq}'

        return self.fixedSubSeq



    def createCustomColorMap(self, colorType):
        colorType = colorType.lower()
        if colorType == 'counts':
            useGreen = True
            if useGreen:
                # Green
                colors = ['#FFFFFF', '#ABFF9B', '#39FF14',
                          '#2E9418', '#2E9418', '#005000']
            else:
                # Orange
                colors = ['white', 'white', '#FF76FA', '#FF50F9',
                          '#FF00F2', '#CA00DF', '#BD16FF']
        elif colorType == 'stdev':
            colors = ['white','white','#FF76FA','#FF50F9','#FF00F2','#CA00DF','#BD16FF']
        elif colorType == 'word cloud':
            # , '#F2A900', '#2E8B57', 'black'
            colors = ['#CC5500', '#F79620', '#FAA338', '#00C01E', 'black']
            # colors = ['#008631', '#39E75F','#CC5500', '#F79620', 'black']
        elif colorType == 'em':
            colors = ['navy','royalblue','dodgerblue','lightskyblue','white','white',
                      'lightcoral','red','firebrick','darkred']
        else:
            print(f'{orange}ERROR: Cannot create colormap. '
                  f'Unrecognized colorType parameter: {colorType}{resetColor}\n\n')
            sys.exit()

        # Create colormap
        if len(colors) == 1:
            colorList = [(0, colors[0]), (1, colors[0])]
        else:
            colorList = [(i / (len(colors) - 1), color) for i, color in enumerate(colors)]
        return LinearSegmentedColormap.from_list('custom_colormap', colorList)



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
            # Verify if the substrates are contain the residue(s) you wish to remove
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
                        if substrate[self.fixedPosition[0] - 1] != self.fixedAA[0]:
                            keepSub = False
                    else:
                        for indexFixed, fixedAA in enumerate(self.fixedAA):
                            indexFixAA = self.fixedPosition[indexFixed] - 1

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

                    subAA = substrate[self.fixedPosition[0] - 1]
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
                        fixIndex = self.fixedPosition[index] - 1
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
        rankedFixedSubstrates = dict(sorted(fixedSubs.items(), key=lambda x: x[1], reverse=True))

        # Print fixed substrates
        if printRankedSubs:
            iteration = 0
            fixedUniqueSubsTotal = len(rankedFixedSubstrates)
            print('Ranked Fixed Substrates:')
            if fixedUniqueSubsTotal == 0:
                print('')
                print(f'{orange}ERROR:\n'
                      f'     No substrates in {purple}{sortType}{orange} contained: '
                      f'{red}{fixedString}{resetColor}\n')
                sys.exit()
            else:
                for substrate, count in rankedFixedSubstrates.items():
                    print(f'     {pink}{substrate}{resetColor}, '
                          f'Counts:{red} {count:,}{resetColor}')
                    iteration += 1
                    if iteration >= self.printNumber:
                        break


        print(f'\nNumber of substrates with fixed {red}{fixedString}{resetColor}: '
              f'{red}{fixedSubsTotal:,}{resetColor}\n\n')

        return rankedFixedSubstrates, fixedSubsTotal



    def identifyMotif(self, entropy, minEntropy, fixFullFrame, getIndices):
        print('================================ Identify Motif '
              '=================================')
        if fixFullFrame:
            print(f'Selecting continuous motif')
        else:
            print(f'Selecting non-continuous motif')
        print(f'Minimum ΔEntropy Value:{pink} {minEntropy} Bits{resetColor}\n\n'
              f'Positional Entropy:\n'
              f'{pink}{entropy}{resetColor}\n')
        subFrame = entropy.copy()
        lastPosition = len(entropy) - 1

        # Determine Substrate Frame
        if fixFullFrame:
            for indexPos, position in enumerate(entropy.index):
                if indexPos == 0 or indexPos == lastPosition:
                    if entropy.loc[position, 'ΔEntropy'] < minEntropy:
                        subFrame.drop(position, inplace=True)
                else:
                    if entropy.loc[position, 'ΔEntropy'] < minEntropy:
                        if (entropy.iloc[indexPos - 1, 0] > minEntropy and
                                entropy.iloc[indexPos + 1, 0] > minEntropy):
                            pass
                        else:
                            subFrame.drop(position, inplace=True)
        else:
            for indexPos, position in enumerate(entropy.index):
                if entropy.loc[position, 'ΔEntropy'] < minEntropy:
                    subFrame.drop(position, inplace=True)

        # Sort the frame
        subFrameSorted = subFrame.sort_values(by='ΔEntropy', ascending=False).copy()
        print(f'Motif Frame:\n'
              f'{pink}{subFrame}{resetColor}\n\n'
              f'Ranked Motif Frame:\n'
              f'{red}{subFrameSorted}{resetColor}\n\n')

        if getIndices:
            indexSubFrameList = list(entropy.index)
            indexSubFrame = [indexSubFrameList.index(idx) for idx in subFrameSorted.index]
            return subFrameSorted, indexSubFrame
        else:
            return subFrameSorted



    def calculateProbCodon(self, codonSeq, printProbability):
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

        if printProbability:
            print('======================= Calculate: Residue Probabilities '
                  '========================')
            print(f'Possible codons for {codonSeq}:')
            # Print all possible codon combinations
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

        if printProbability:
            print('Amino Acid Probabilities:')
            for index, row in codonProbability.iterrows():
                print(f'     {index}    {round(row["Probability"] * 100, 2)} %')
            codonProb = round(sum(codonProbability["Probability"]) * 100, 2)
            print(f'Total probability of AA with {codonSeq}: {codonProb} %')
            print(f'Stop codon probability: {round(100 - codonProb, 2)} %\n\n')

        return codonProbability



    def calculateRF(self, counts, N, fileType, printRF):
        # Calculate Relative Frequency of each amino acid at a given position
        RF = counts / N

        if printRF:
            print('========================= Calculate: Relative Frequency '
                  '=========================')
            print(f'RF: {self.enzymeName} - {purple}{fileType}{resetColor}\n'
                  f'{np.round(RF, 4)}\n\n')

        return RF



    def compairRF(self, probInitial, probFinal, selectAA):
        print('======================= Evaluate Specificity: Compair RF '
              '========================')
        if selectAA in self.letters:
            residue = self.residues[self.letters.index(selectAA)][0]
        else:
            print(f'{greyDark}Residue not recognized:{red} {selectAA}{greyDark}\n'
                  f'Please check input:{red} self.fixedAA')
            sys.exit()
        print(f'Fixed Residues:{red} {self.fixedSubSeq}{resetColor}\n'
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
        ax.set_title(f'{residue} RF: {self.enzymeName} Fixed {self.fixedSubSeq}',
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
            sys.exit()
        print(f'Fixed Residues:{red} {self.fixedSubSeq}{resetColor}\n'
              f'Selected Residue:{red} {residue}{resetColor}\n')

        # Extract data
        initial = probInitial[probInitial.index.str.contains(selectAA)].T
        final = probFinal[probFinal.index.str.contains(selectAA)].T
        print(f'{purple}Initial Sort{resetColor}:\n{initial}\n')
        print(f'{purple}Final Sort{resetColor}:\n{final}\n\n')
        print(f'Pos: {self.fixedPosition}')
        final = final.drop(final.index[[int(pos) - 1 for pos in self.fixedPosition]])
        print(f'Remove fixed residues:{purple} Final Sort{resetColor}\n{final}\n\n')

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

        # Print the outliers
        if len(outliersInitial) != 0:
            print(f'Outliers:{purple} Initial Sort{resetColor}')
            for outlierPosition in outliersInitial:
                print(f'     {outlierPosition}')
            print()
        else:
            print(f'There were no{red} {residue}{resetColor} RF outliers in:'
                  f'{purple} Initial Sort{resetColor}\n')
        if len(outliersFinal) != 0:
            print(f'Outliers:{purple} Final Sort{resetColor}')
            for outlierPosition in outliersFinal:
                print(f'     {outlierPosition}')
            print('\n')
        else:
            print(f'There were no{red} {residue}{resetColor} RF outliers in:'
                  f'{purple} Final Sort{resetColor} '
                  f'fixed{red} {self.fixedSubSeq}{resetColor} \n\n')

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
        ax.set_title(f'{self.enzymeName} - {residue} RF: Fixed {self.fixedSubSeq}',
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



    def calculateEnrichment(self, initialSortRF, finalSortRF, printES):
        if len(initialSortRF.columns) == 1:
            enrichment = pd.DataFrame(0.0, index=finalSortRF.index,
                                      columns=finalSortRF.columns)
            for position in finalSortRF.columns:
                enrichment.loc[:, position] = np.log2(finalSortRF.loc[:, position] /
                                                      initialSortRF.iloc[:, 0])
        else:
            initialSortRF.columns = finalSortRF.columns
            enrichment = np.log2(finalSortRF / initialSortRF)
            enrichment.columns = finalSortRF.columns
        if printES:
            print('========================== Calculate: Enrichment Score '
                  '==========================')
            print(f'Enrichment Score:{purple} {self.fixedSubSeq}{resetColor}\n'
                  f'{enrichment.round(3)}\n\n')
        else:
            print('\n')

        return enrichment



    def fixedMotifStats(self, countsList, initialProb, substrateFrame, datasetTag):
        print('================== Statistical Evaluation: Fixed Motif Counts '
              '===================')
        print(f'Evaluate:{purple} {datasetTag}{resetColor}\n')
        countsFrameTotal = pd.DataFrame(0, index=range(0, len(self.fixedPosition)),
                                        columns=substrateFrame)
        frameProb = pd.DataFrame(0.0, index=initialProb.index,
                                 columns=substrateFrame)
        frameES = frameProb.copy()
        frameESList = []

        for index, countsFrame in enumerate(countsList):
            countsFrame = pd.DataFrame(countsFrame, index=initialProb.index,
                                       columns=substrateFrame)

            # Format values to have commas
            formattedCounts = countsFrame.to_string(
                formatters={column: '{:,.0f}'.format for column in
                            countsFrame.select_dtypes(include='number').columns})
            print(f'Counts:{purple} Fixed Motif '
                  f'{self.fixedAA[0]}@R{self.fixedPosition[index]}{resetColor}\n'
                  f'{formattedCounts}\n')

            for column in countsFrame.columns:
                countsFrameTotal.loc[index, column] = countsFrame[column].sum()

                # Calculate: RF
                frameProb.loc[:, column] = (countsFrame.loc[:, column] /
                                            countsFrameTotal.loc[index, column])

                # Calculate: ES
                frameES.loc[:, column] = np.log2(
                    frameProb.loc[:, column] / initialProb['Average RF'])

            print(f'Enrichment Score:{purple} '
                  f'Fixed Motif {self.fixedAA[0]}@R{self.fixedPosition[index]}\n'
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

        print(f'Average:{purple} Enrichment Score\n'
              f'{greyDark}{frameESAvg}{resetColor}\n\n'
              f'Standard Deviation:{purple} Enrichment Score\n'
              f'{greyDark}{frameESStDev}{resetColor}\n\n')


        # Plot: Standard deviation
        NGS.plotStats(
            self, countedData=frameESStDev, totalCounts=None,
            title=f'{self.enzymeName}\nFixed Motif {self.fixedAA[0]}@R'
                  f'{self.fixedPosition[0]}-R{self.fixedPosition[-1]}\n'
                  f'Standard Deviation', datasetTag=datasetTag, dataType='StDev')



    def heightsRF(self, counts, N, printRF):
        print('=========================== Calculate: Letter Heights '
              '===========================')
        print(f'Residue heights calculated by:{red} RF * ΔS{resetColor}\n')


        fixedPos = {}
        for indexColumn in counts.columns:
            values = counts.loc[:, indexColumn]
            if N in values.values:
                indexRow = values[values == N].index[0]
                fixedPos[indexColumn] = indexRow
        print(f'Fixed Residues:')
        if fixedPos:
            for key, value in fixedPos.items():
                print(f'     Fixed Position: {red}{key}{resetColor}, '
                      f'Residue: {red}{value}{resetColor}')
        else:
            print(f'     {red}No fixed Residues{resetColor}')
        print('\n')


        RF = counts / N

        # Print data
        if printRF:
            print(f'Experimental Counts:{purple} Final Sort{resetColor}\n{counts}\n\n')
            print(f'RF:{purple} Final Sort\n'
                  f'{greyDark}{RF.round(3)}{resetColor}\n\n')


        entropy = pd.DataFrame(0.0, index=RF.columns, columns=['ΔEntropy'])
        entropyMax = np.log2(len(RF.index))
        for indexColumn in RF.columns:
            S = 0 # Reset entropy total for a new position
            for indexRow, probRatio in RF.iterrows():
                prob = probRatio[indexColumn]
                if prob == 0:
                    continue
                else:
                    S += -prob * np.log2(prob)
            entropy.loc[indexColumn, 'ΔEntropy'] = entropyMax - S
        self.delta = sum(entropy.loc[:, 'ΔEntropy']) / self.substrateLength
        print(f'{entropy}\n\nMax Entropy: {entropyMax.round(6)}\n\n')


        heights = pd.DataFrame(0, index=counts.index, columns=counts.columns, dtype=float)
        for indexColumn in heights.columns:
            heights.loc[:, indexColumn] = (RF.loc[:, indexColumn] *
                                           entropy.loc[indexColumn, 'ΔEntropy'])

        columnTotals = [[], []]
        for indexColumn in heights.columns:
            totalPos = 0
            totalNeg = 0
            for value in heights.loc[:, indexColumn]:
                if value > 0:
                    totalPos += value
                elif value < 0:
                    totalNeg += value
            columnTotals[0].append(totalPos)
            columnTotals[1].append(totalNeg)
        yMax = max(columnTotals[0])
        yMin = min(columnTotals[1])

        if printRF:
            print(f'Residue Heights:\n{heights.round(3)}\n\n'
                  f'y Max: {red}{np.round(yMax, 4)}{resetColor}\n'
                  f'y Min: {red}{np.round(yMin, 4)}{resetColor}\n\n')

        # Set values for columns with fixed residues
        for key, value in fixedPos.items():
            heights.loc[value, key] = yMax

        return heights, fixedPos, yMax, yMin



    def enrichmentMatrix(self, counts, N, baselineProb, baselineType, printRF, scaleData,
                         normalizeFixedScores):
        print('=========================== Calculate: Letter Heights '
              '===========================')
        print(f'Residue heights calculated by: {red}log\u2082(RF Ratios){resetColor}\n'
              f'Baseline Probability: {purple}{baselineType}{resetColor}\n')

        if len(baselineProb.columns) != 1:
            baselineProb.columns = counts.columns


        fixedPos = {}
        for indexColumn in counts.columns:
            values = counts.loc[:, indexColumn]
            if N in values.values:
                indexRow = values[values == N].index[0]
                fixedPos[indexColumn] = indexRow
        print(f'Fixed Residues:')
        if fixedPos:
            for key, value in fixedPos.items():
                print(f'     Fixed Position: {red}{key}{resetColor}, '
                      f'Residue: {red}{value}{resetColor}')
        else:
            print('     No fixed Residues')
        print('\n')

        probCounts = counts / N
        probRatios = pd.DataFrame(0, index=counts.index, columns=counts.columns,
                                  dtype=float)
        if len(baselineProb.columns) == 1:
            for indexColumn in counts.columns:
                # print(f'Position: {indexColumn}')
                for indexRow in counts.index:
                    probRatio = np.log2(probCounts.loc[indexRow, indexColumn] /
                                         baselineProb.loc[indexRow, baselineProb.columns[0]])
                    probRatios.loc[indexRow, indexColumn] = probRatio
        else:
            for indexColumn in counts.columns:
                # print(f'Position: {indexColumn}')
                for indexRow in counts.index:
                    probRatio = np.log2(probCounts.loc[indexRow, indexColumn] /
                                         baselineProb.loc[indexRow, indexColumn])
                    probRatios.loc[indexRow, indexColumn] = probRatio
        probRatios.replace(-np.inf, 0, inplace=True)
        if printRF:
            print(f'RF: {purple}{baselineType}{resetColor}\n{baselineProb.round(3)}\n\n')
            print(f'RF:{purple} Final Sort{resetColor}\n{probCounts.round(3)}\n\n')

        if scaleData:
            entropy = pd.DataFrame(0.0, index=probCounts.columns, columns=['Entropy'])
            entropy['Entropy'] = entropy['Entropy'].astype(float)
            entropyMax = np.log2(len(probCounts.index))
            for indexColumn in probCounts.columns:
                S = 0 # Reset entropy total for a new position
                for indexRow, probRatio in probCounts.iterrows():
                    prob = probRatio[indexColumn]
                    if prob == 0:
                        continue
                    else:
                        S += -prob * np.log2(prob)
                entropy.loc[indexColumn, 'Entropy'] = float(entropyMax - S)


            heights = pd.DataFrame(0,
                                   index=counts.index,
                                   columns=counts.columns,
                                   dtype=float)
            for indexColumn in heights.columns:
                heights.loc[:, indexColumn] = (probRatios.loc[:, indexColumn] *
                                               entropy.loc[indexColumn, 'Entropy'])
        else:
            heights = probRatios
        heights = heights.fillna(0.0)


        columnTotals = [[], []]
        for indexColumn in heights.columns:
            totalPos = 0
            totalNeg = 0
            for value in heights.loc[:, indexColumn]:
                if value > 0:
                    totalPos += value
                elif value < 0:
                    totalNeg += value
            columnTotals[0].append(totalPos)
            columnTotals[1].append(totalNeg)
        yMax = max(columnTotals[0])
        yMin = min(columnTotals[1])

        if printRF:
            if scaleData:
                print(f'Residue Heights: log\u2082({cyan}RF Final Sort{resetColor} / '
                      f'{cyan}RF {baselineType}{resetColor})\n'
                      f'{probRatios.round(3)}\n')

                print(f'{entropy}\n\nMax Entropy: {entropyMax.round(6)}\n\n')
                print(f'{pink}Scaled Residue Heights{resetColor}:{cyan} '
                      f'ΔS * log\u2082(RF Final Sort / RF  {baselineType}{resetColor})\n'
                      f'{heights.round(5)}\n')
            else:
                print(f'{pink}Residue Heights{resetColor}:{cyan} '
                      f'log\u2082(RF Final Sort / RF {baselineType}{resetColor})\n'
                      f'{heights.round(3)}\n')
            print(f'y Max: {red}{np.round(yMax, 4)}{resetColor}\n'
                  f'y Min: {red}{np.round(yMin, 4)}{resetColor}\n\n')

        # Set values for columns with fixed residues
        if normalizeFixedScores:
            print(f'============================================'
                  f'============================================={greenLightB}\n'
                  f'============================================'
                  f'============================================={resetColor}\n'
                  f'============================================'
                  f'=============================================\n\n'
                  f'{yellow}Readjusting the fixed residue heights '
                  f'to be equal to the tallest stack{resetColor}\n\n'
                  f'============================================'
                  f'============================================={greenLightB}\n'
                  f'============================================'
                  f'============================================={resetColor}\n'
                  f'============================================'
                  f'=============================================\n')

            for key, value in fixedPos.items():
                heights.loc[value, key] = yMax

            print(f'Adjusted Residue Heights:{pink} log\u2082(RF Final Sort / RF '
                  f'{baselineType})\n{greyDark}{heights.round(3)}{resetColor}\n')

            print(f'============================================'
                  f'============================================={greenLightB}\n'
                  f'============================================'
                  f'============================================={resetColor}\n'
                  f'============================================'
                  f'=============================================\n\n'
                  f'{yellow}Readjusting the fixed residue heights '
                  f'to be equal to the tallest stack{resetColor}\n\n'
                  f'============================================'
                  f'============================================={greenLightB}\n'
                  f'============================================'
                  f'============================================={resetColor}\n'
                  f'============================================'
                  f'=============================================\n')

        return heights, fixedPos, yMax, yMin



    def KLDivergence(self, P, Q, printProb, scaler):
        print('================================= KL Divergence '
              '=================================')
        P.columns = Q.columns
        if printProb:
            print(f'Baseline Probability Distribution:\n{Q}\n\n\n'
                  f'Probability Distribution:\n{P}\n\n')

        divergence = pd.DataFrame(0, columns=Q.columns,
                                  index=[self.fixedSubSeq], dtype=float)
        divergenceMatrix = pd.DataFrame(0, columns=Q.columns,
                                        index=Q.index, dtype=float)

        for position in Q.columns:
            p = P.loc[:, position]
            q = Q.loc[:, position]
            divergence.loc[self.fixedSubSeq, position] = (
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
                                                     scaler.loc[position, 'ΔEntropy'])

        print(f'{greyDark}KL Divergence:'
              f'{pink} Fixed Final Sort - {self.fixedSubSeq}{resetColor}\n'
              f'{divergence}\n\n\n{greyDark}Divergency Matrix:'
              f'{pink} Fixed Final Sort - {self.fixedSubSeq}'
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
            print(f'Optimal Residues:{purple} {self.enzymeName} - '
                  f'Fixed {self.fixedSubSeq}{resetColor}')
            for index, data in enumerate(optimalAA, start=1):
                # Determine the number of variable residues at this position
                numberAA = len(data)
                combinations *= numberAA

                # Define substrate position
                positionSub = self.xAxisLabels[index-1]
                print(f'Position:{purple} {positionSub}{resetColor}')

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



    def calculateEntropy(self, RF, printEntropy, datasetTag=''):
        print('============================== Calculate: Entropy '
              '===============================')
        entropy = pd.DataFrame(0.0, index=RF.columns, columns=['ΔEntropy'])
        entropyMax = np.log2(len(RF.index))
        for indexColumn in RF.columns:
            S = 0 # Reset entropy total for a new position
            for indexRow, probRatio in RF.iterrows():
                prob = probRatio[indexColumn]
                if prob == 0:
                    continue
                else:
                    S += -prob * np.log2(prob)
            entropy.loc[indexColumn, 'ΔEntropy'] = entropyMax - S

        if printEntropy:
            print(f'Positional Entropy: {datasetTag}\n{entropy}\n\nMax Entropy: '
                  f'{entropyMax.round(6)}\n\n')

        return entropy



    def substrateEnrichment(self, initialSubs, finalSubs, saveData,
                            savePath, NSubs):
        print('========================= Evaluate Substrate Enrichment '
              '=========================')
        if self.fixedSubSeq == None:
            datasetType = 'NNS'
        else:
            datasetType = f'{self.fixedSubSeq}'

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
        totalSubstratesFinal = 0
        totalUniqueSubstratesFinal = len(finalSubs)

        # Evaluate the substrates
        if self.fixedSubSeq == None:
            # Process: initial sort
            print(f'Ranked Substrates:{purple} Initial Sort{resetColor} -'
                  f'{red} NNS{resetColor}')
            if totalUniqueSubstratesInitial >= self.printNumber:
                for substrate, count in initialSubs.items():
                    if iteration <= self.printNumber:
                        print(f'     {magenta}{substrate}{resetColor}, '
                              f'Counts: {greenLight}{count:,}'
                              f'{resetColor}')
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
                    print(f'     {magenta}{substrate}{resetColor}, Counts: {red}{count:,}'
                          f'{resetColor}')
                    totalSubstratesInitial += count
            iteration = 0
            # Print dataset totals
            print(f'\n     Total substrates{purple} Initial Sort{resetColor}:'
                  f'{red} {totalSubstratesInitial:,}{resetColor}\n'
                  f'     Unique substrates{purple} Initial Sort{resetColor}:'
                  f'{red} {totalUniqueSubstratesInitial:,}{resetColor}\n\n')


            # Process: final sort
            print(f'Ranked Substrates:{purple} Final Sort{resetColor} -'
                  f'{red} NNS{resetColor}')
            if totalUniqueSubstratesFinal >= self.printNumber:
                for substrate, count in finalSubs.items():
                    if iteration <= self.printNumber:
                        print(f'     {magenta}{substrate}{resetColor}, '
                              f'Counts: {red}{count:,}{resetColor}')
                        iteration += 1
                        totalSubstratesFinal += count
                    else:
                        totalSubstratesFinal += count
            else:
                print(f'{orange}The number of unique substrates '
                      f'({red}{totalUniqueSubstratesFinal}'
                      f'{orange}) is less than the number you requested to be see '
                      f'({red}{self.printNumber}{orange}){resetColor}\n')
                for substrate, count in finalSubs.items():
                    print(f'     {magenta}{substrate}{resetColor}, '
                          f'Counts: {red}{count:,}{resetColor}')
                    totalSubstratesFinal += count
            iteration = 0
            # Print dataset totals
            print(f'\n     Total substrates{purple} Final Sort{resetColor}:'
                  f'{red} {totalSubstratesFinal:,}{resetColor}\n'
                  f'     Unique substrates{purple} Final Sort{resetColor}:'
                  f'{red} {totalUniqueSubstratesFinal:,}{resetColor}\n\n')
        else:
            fixedSort = True

            # Print: Initial sort
            print(f'Ranked Substrates:{purple} Initial Sort{resetColor}{resetColor}')
            if totalUniqueSubstratesInitial >= self.printNumber:
                for substrate, count in initialSubs.items():
                    if iteration < self.printNumber:
                        print(f'     {magenta}{substrate}{resetColor}, '
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
                    print(f'     {magenta}{substrate}{resetColor}, Counts: {red}{count:,}'
                          f'{resetColor}')
                    totalSubstratesInitial += count
            print(f'\n     Total substrates{purple} Initial Sort{resetColor}:'
                  f'{red} {totalSubstratesInitial:,}{resetColor}\n'
                  f'     Unique substrates{purple} Initial Sort{resetColor}:'
                  f'{red} {totalUniqueSubstratesInitial:,}{resetColor}\n\n')
            iteration = 0


            # Process: Final sort
            print(f'Ranked Substrates:{purple} Final Sort{resetColor} -'
                  f'{red} Fixed {self.fixedSubSeq}{resetColor}')
            if totalUniqueSubstratesFinal >= self.printNumber:
                for substrate, count in finalSubs.items():
                    if iteration < self.printNumber:
                        print(f'     {magenta}{substrate}{resetColor}, '
                              f'Counts: {red}{count:,}{resetColor}')
                        iteration += 1
                        totalSubstratesFinal += count
                    else:
                        totalSubstratesFinal += count
            else:
                print(f'{orange}The number of unique substrates '
                      f'({red}{totalUniqueSubstratesFinal}'
                      f'{orange}) is less than the number you requested to be see '
                      f'({red}{self.printNumber}{orange}){resetColor}')
                for substrate, count in finalSubs.items():
                    print(f'     {magenta}{substrate}{resetColor}, Counts: {red}{count:,}'
                          f'{resetColor}')
                    totalSubstratesFinal += count
            print(f'\n     Total substrates{purple} Final Sort{resetColor}:'
                  f'{red} {totalSubstratesFinal:,}{resetColor}\n'
                  f'     Unique substrates{purple} Final Sort{resetColor}:'
                  f'{red} {totalUniqueSubstratesFinal:,}{resetColor}\n\n')
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
                probFinal = count / totalSubstratesFinal
                probInitial = countInitial / totalSubstratesInitial
                enrichment = np.log2(probFinal/probInitial)
                enrichedSubs[substrate] = enrichment
        else:
            for substrate, count in finalSubs.items():
                if substrate in initialSubs.keys():
                    countInitial = initialSubs[substrate]
                else:
                    countInitial = 1
                probFinal = count / totalSubstratesFinal
                probInitial = countInitial / totalSubstratesInitial
                enrichment = np.log2(probFinal/probInitial)
                enrichedSubs[substrate] = enrichment
        # sys.exit()

        # Sort enrichment dictionary
        enrichedSubs = dict(sorted(enrichedSubs.items(),
                                   key=lambda x: x[1], reverse=True))

        # Print top enriched substrates
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
            if '/' in savePath:
                filePathCSV = (f'{savePath}/{self.enzymeName} '
                               f'--- {datasetType} --- Enriched Subs.csv')
            else:
                filePathCSV = (f'{savePath}\\{self.enzymeName} '
                               f'--- {datasetType} --- Enriched Subs.csv')

            # Convert dictionaries to a data frame and save as an Excel file
            clipDataset = False
            if totalSubstratesInitial > NSubs:
                clipDataset = True

                # Did you ask for too many substrates?
                if NSubs == 10**6:
                    NSubs -= 1
                elif NSubs > 10**6:
                    print(f'     The list of substrates in the {purple} Initial Sort'
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
            if totalSubstratesFinal > NSubs:
                clipDataset = True

                # Did you ask for too many substrates?
                if NSubs > 10**6:
                    print(f'     The list of substrates in the {purple} Final Sort'
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

            # Print the sample sizes
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



    def ESM(self, substrates, collectionNumber, useSubCounts, subPositions, datasetTag):
        print('=========================== Convert To Numerical: ESM '
              '===========================')
        print(f'Dataset:{purple} {self.enzymeName} - {datasetTag}{resetColor}\n\n'
              f'Collecting{red} {collectionNumber:,}{resetColor} substrates\n'
              f'Total unique substrates:{red} {len(substrates):,}{resetColor}\n')

        import esm

        # Extract: Datapoints
        iteration = 0
        collectedCountsTotal = 0
        evaluateSubs = {}
        for substrate, count in substrates.items():
            evaluateSubs[str(substrate)] = count
            iteration += 1
            collectedCountsTotal += count
            if iteration >= collectionNumber:
                break
        sampleSize = len(evaluateSubs)
        print(f'Collected substrates:{red} {sampleSize:,}{resetColor}\n'
              f'Total Counts:{red} {collectedCountsTotal:,}{resetColor}\n\n')

        # Step 1: Convert substrates to ESM model format and generate embeddings
        subs = []
        if useSubCounts:
            counts = []
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
                  f'{resetColor}\n'
                  f'     With: {cyan}esm.pretrained.esm2_t33_650M_UR50D()'
                  f'{resetColor}\n')
            sys.exit()

        print(f'Batch Tokens:{greenLightB} {batchTokens.shape}{resetColor}\n'
              f'{greenLight}{batchTokens}{resetColor}\n\n')
        slicedTokens = pd.DataFrame(batchTokens[:, 1:-1],
                                    index=batchSubs,
                                    columns=subPositions)
        if useSubCounts:
            slicedTokens['Counts'] = counts
        print(f'Sliced Tokens:\n'
              f'{greenLight}{slicedTokens}{resetColor}\n\n')

        return slicedTokens, batchSubs, sampleSize



    def plotPCA(self, substrates, data, indices, numberOfPCs, fixedTag, N, fixedSubs,
            saveTag):
        print('====================================== PCA '
              '======================================')
        print(f'Tag: {fixedTag}\n'
              f'Save: {saveTag}\n')
        import matplotlib.patheffects as path_effects
        from matplotlib.widgets import RectangleSelector
        from sklearn.decomposition import PCA


        # Initialize lists for the clustered substrates
        self.selectedSubstrates = []
        self.selectedDatapoints = []
        rectangles = []

        # Define: Dataset tag
        if fixedTag is None:
            print(f'Dataset:{purple} {self.enzymeName} - Unfiltered{resetColor}\n')
        else:
            print(f'Dataset:{purple} {self.enzymeName} - {fixedTag}{resetColor}\n')
            if 'Excl' in fixedTag:
                fixedTag = fixedTag.replace('Excl', 'Exclude')

        # Define component labels
        pcaHeaders = []
        for componetNumber in range(1, numberOfPCs + 1):
            pcaHeaders.append(f'PC{componetNumber}')
        headerCombinations = list(combinations(pcaHeaders, 2))


        # # Cluster the datapoints
        # Step 1: Apply PCA on the standardized data
        pca = PCA(n_components=numberOfPCs)  # Adjust the number of components as needed
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        dataPCA = pca.fit_transform(data)
        # loadings = pca.components_.T

        # Step 2: Create a DataFrame for PCA results
        dataPCA = pd.DataFrame(dataPCA, columns=pcaHeaders, index=indices)
        print(f'PCA Data:{red} # of componets = {numberOfPCs}\n'
              f'{greenLight}{dataPCA}{resetColor}\n\n')

        # Step 3: Print explained variance ratio
        varRatio = pca.explained_variance_ratio_ * 100
        print(f'Explained Variance Ratio: '
              f'{red}{" ".join([f"{x:.3f}" for x in varRatio])}{resetColor} %\n\n')


        # Define: Figure parameters
        if fixedSubs:
            title = (f'{self.enzymeName}\n'
                     f'{fixedTag}\n'
                     f'{N:,} Unique Substrates')
        else:
            title = (f'{self.enzymeName}\n'
                     f'{N:,} Unique Substrates'),


        # Plot the data
        for componets in headerCombinations:
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
                        padding = 0.4
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
            plt.scatter(dataPCA[componets[0]], dataPCA[componets[1]],
                        c='#CC5500', edgecolor='black')
            plt.xlabel(f'Principal Component {componets[0][-1]} ({varRatio[0]})',
                       fontsize=self.labelSizeAxis)
            plt.ylabel(f'Principal Component {componets[1][-1]} ({varRatio[1]})',
                       fontsize=self.labelSizeAxis)
            plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')


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
            fig.tight_layout()
            plt.show()


        # Save the Figure
        if self.saveFigures:
            # Define: Save location
            figLabel = (f'{self.enzymeName} - PCA - {fixedTag} - '
                        f'{N} - MinCounts {self.minSubCount}.png')
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
            print(f'Update: Make this able to shift the frame '
                  f'if you plot binned substrates')
            collectedSubs = []
            for index, substrateSet in enumerate(self.selectedSubstrates):
                print(f'Substrate Set:{greenLightB} {index + 1}{resetColor}')
                iteration = 0
                collectionSet = {}
                for substrate in substrateSet:
                    collectionSet[substrate] = substrates[substrate]

                # Sort collected substrates and add to the list
                collectionSet = dict(sorted(collectionSet.items(),
                                            key=lambda x: x[1], reverse=True))
                collectedSubs.append(collectionSet)

                # Print collected substrates
                for substrate, count in collectionSet.items():
                    print(f'     {greyDark}{substrate}{resetColor}:'
                          f'{red} {count:,}{resetColor}')
                    iteration += 1
                    if iteration >= self.printNumber:
                        print('\n')
                        break

            return collectedSubs
        else:
            return None



    def suffixTree(self, substrates, N, entropySubFrame, indexSubFrame, entropyMin,
                   datasetTag, dataType):
        print('================================== Suffix Tree '
              '==================================')
        if datasetTag is None:
            print(f'Dataset:{purple} {self.enzymeName} - Unfixed{resetColor}\n')
        else:
            print(f'Dataset:{purple} {self.enzymeName} {dataType} {datasetTag}'
                  f'{resetColor}\n')


        from Trie import Trie


        trie = Trie() # Initialize Trie
        motifs = {}
        indexStart = min(indexSubFrame)
        indexEnd = max(indexSubFrame)

        # Print substrates
        iteration = 0
        substrates = dict(sorted(substrates.items(), key=lambda item: item[1],
                                 reverse=True))
        for substrate, count in substrates.items():
            iteration += 1
            print(f'Substrate:{greyDark} {substrate}{resetColor}\n'
                  f'     Count:{red} {count:,}{resetColor}')
            if iteration >= self.printNumber:
                break
        print('\n')

        # Find motif positions based on the entropy threshold
        indexPos = []
        for index in entropySubFrame.index:
            posEntropy = entropySubFrame.loc[index, 'ΔEntropy']
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

        # Print motifs
        print(f'Extracted Motifs:')
        for index, (motif, count) in enumerate(motifs.items()):
            print(f'{index+1}:{yellow} {motif}{resetColor} '
                  f'Count:{red} {count:,}{resetColor}')
        print('\n')

        # Print trie
        print(f'Extracted Trie:')
        for index, (seq, count) in enumerate(motifTrie.items()):
            print(f'{index + 1}:{pink} {seq}{resetColor} '
                  f'Count:{red} {count:,}{resetColor}')
        print('\n')

        # Calculate: RF
        motifTable = NGS.evaluateSubtrees(self, trie=trie, motifTrie=motifTrie)

        # Plot the Trie
        NGS.plotTrie(self, trie=trie, motifTable=motifTable, countsMotif=countsMotif,
                     datasetTag=datasetTag)



    def plotCounts(self, countedData, totalCounts, datasetTag):
        # Remove commas from string values and convert to float
        countedData = countedData.applymap(lambda x:
                                           float(x.replace(',', ''))
                                           if isinstance(x, str) else x)

        # Create heatmap
        cMapCustom = NGS.createCustomColorMap(self, colorType='Counts')

        # Convert the counts to a data frame for Seaborn heatmap
        if self.residueLabelType == 0:
            countedData.index = [residue[0] for residue in self.residues]
        elif self.residueLabelType == 1:
            countedData.index = [residue[1] for residue in self.residues]
        elif self.residueLabelType == 2:
            countedData.index = [residue[2] for residue in self.residues]

        # Plot the heatmap with numbers centered inside the squares
        fig, ax = plt.subplots(figsize=self.figSize)
        heatmap = sns.heatmap(countedData, annot=True, fmt=',d', cmap=cMapCustom,
                              cbar=True, linewidths=self.lineThickness-1,
                              linecolor='black', square=False, center=None,
                              annot_kws={'fontweight': 'bold'})
        ax.set_xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        ax.set_ylabel('Residue', fontsize=self.labelSizeAxis)

        if datasetTag != self.enzymeName:
            datasetTag = f'{self.enzymeName}\n{datasetTag}'
        if self.showSampleSize:
            if totalCounts is None:
                ax.set_title(datasetTag, fontsize=self.labelSizeTitle, fontweight='bold')
            else:
                ax.set_title(f'{datasetTag}\nN={totalCounts:,}',
                             fontsize=self.labelSizeTitle, fontweight='bold')
        else:
            ax.set_title(datasetTag, fontsize=self.labelSizeTitle, fontweight='bold')
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



    def plotPositionalEntropy(self, entropy, fixedDataset, fixedTag, avgDelta):
        # Figure parameters
        maxS = np.log2(len(self.letters))
        yMax = maxS + 0.2
        entropyMax = maxS # max(entropy.loc[:, "ΔEntropy"])
        self.delta = entropy['ΔEntropy'].mean()
        # deltaStDev = entropy['ΔEntropy'].std()
        if fixedDataset:
            title = f'{self.enzymeName}: Fixed {fixedTag}'
        else:
            title = f'{self.enzymeName}: Unfiltered'


        # Map entropy values to colors using the colormap
        colors = [(0, 'navy'),
                  (0.3/entropyMax, 'navy'),
                  (0.7/entropyMax, 'dodgerblue'),
                  (0.97/entropyMax, 'white'),
                  (0.98/entropyMax, 'white'),
                  (1.0/entropyMax, 'white'),
                  (1.65/entropyMax, 'red'),
                  (3/entropyMax, 'firebrick'),
                  (1, 'darkred')]
        colorBar = LinearSegmentedColormap.from_list('custom_colormap', colors)


        # Map entropy values to colors using the colormap
        normalize = Normalize(vmin=0, vmax=yMax) # Normalize the entropy values
        cMap = [colorBar(normalize(value)) for value in entropy['ΔEntropy'].astype(float)]

        # Plotting the entropy values as a bar graph
        fig, ax = plt.subplots(figsize=self.figSizeMini)
        plt.bar(entropy.index, entropy['ΔEntropy'], color=cMap,
                edgecolor='black', linewidth=self.lineThickness, width=0.8)
        plt.xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        plt.ylabel('ΔS', fontsize=self.labelSizeAxis, rotation=0, labelpad=15)
        if avgDelta:
            plt.title(f'{title}\nAverage ΔS = {self.delta:.5f}',
                      fontsize=self.labelSizeTitle, fontweight='bold')
        else:
            plt.title(f'\n{title}',
                      fontsize=self.labelSizeTitle, fontweight='bold')
        plt.subplots_adjust(top=0.898, bottom=0.098, left=0.121, right=0.917)

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
            if fixedDataset:
                figLabel = (f'{self.enzymeName} - Positional Entropy - '
                            f'Filter {fixedTag} - MinCounts {self.minSubCount}.png')
            else:
                figLabel = (f'{self.enzymeName} - Positional Entropy - '
                            f'Unfiltered - MinCounts {self.minSubCount}.png')
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



    def plotLibraryProbDist(self, probInitial, probFinal, codonType, fixedTag):
        # Inspect data
        if probInitial is None and probFinal is None:
            print(f'{orange}ERROR: both of the inputs for probInitial and '
                  f'probFinal cannot be None.{resetColor}\n\n')
            sys.exit()

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
        if codonType == fixedTag:
            yMax = np.ceil(probFinal.values.max() * 10) / 10


        def plotFig(probability, sortType):
            print('======================= Plot: AA Probability Distribution '
                  '=======================')
            if codonType == fixedTag:
                print(f'Plotting Probability Distribution:'
                      f'{purple} {codonType} codon{resetColor}')
            else:
                print(f'Plotting Probability Distribution:'
                      f'{purple} {self.enzymeName} {sortType}{resetColor}')
            print(f'{probability}\n')

            fig, ax = plt.subplots(figsize=self.figSize)
            plt.ylabel('Probability Distribution', fontsize=self.labelSizeAxis)
            if sortType == 'Initial Sort':
                plt.title(f'Unsorted {self.enzymeName} Library',
                          fontsize=self.labelSizeTitle, fontweight='bold')
            else:
                if fixedTag is None:
                    plt.title(f'Sorted {self.enzymeName} Library',
                              fontsize=self.labelSizeTitle, fontweight='bold')
                else:
                    if codonType == fixedTag:
                        plt.title(f'{codonType} Codon',
                        fontsize=self.labelSizeTitle, fontweight='bold')
                    else:
                        plt.title(f'Sorted {self.enzymeName} Library - '
                                  f'{fixedTag}', fontsize=self.labelSizeTitle,
                                  fontweight='bold')
            plt.subplots_adjust(top=0.926, bottom=0.068, left=0.102, right=0.979)


            # Set tick parameters
            ax.tick_params(axis='both', which='major', length=self.tickLength,
                           width=self.lineThickness)

            # Set x-ticks
            if codonType == fixedTag:
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
                if fixedTag is None:
                    figLabel = (f'AA Distribution - {self.enzymeName} - Unfiltered - '
                                f'{sortType} - Y Max {yMax} - {codonType} - '
                                f'MinCounts {self.minSubCount}.png')
                else:
                    if codonType == fixedTag:
                        figLabel = f'AA Distribution - {codonType} Codon.png'
                    else:
                        figLabel = (f'AA Distribution - {self.enzymeName} - {fixedTag} - '
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
        if plotInitial:
            plotFig(probability=probInitial, sortType='Initial Sort')
        if plotFinal:
            if codonType == fixedTag:
                plotFig(probability=probFinal, sortType=fixedTag)
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
                          f'ΔS = {entropyScores.loc[position, "ΔEntropy"]:.3f}, '
                          f'Shannon Entropy = {shannonS:.0f}',
                          fontsize=self.labelSizeTitle, fontweight='bold')
            else:
                plt.title(f'{self.enzymeName}: '
                          f'Amino Acid Distribution at {position}\n'
                          f'ΔS = {entropyScores.loc[position, "ΔEntropy"]:.3f}, '
                          f'Shannon Entropy = {shannonS:.3f}',
                          fontsize=self.labelSizeTitle, fontweight='bold')
            plt.subplots_adjust(top=0.898, bottom=0.1, left=0.129, right=0.936)
            plt.ylim(0, yMax)

            # Set tick parameters
            ax.tick_params(axis='both', which='major', length=self.tickLength,
                           labelsize=self.labelSizeTicks)

            # Set x-ticks
            xTicks = np.arange(0, len(self.letters), 1)
            ax.set_xticks(xTicks)  # Set the positions of the ticks
            ax.set_xticklabels(self.letters, rotation=0, ha='center',
                               fontsize=self.labelSizeTicks)
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_markeredgewidth(self.lineThickness) # Set tick width

            # Set y-ticks
            tickStepSize = 0.2
            yTicks = np.arange(0, yMax + tickStepSize, tickStepSize)
            yTickLabels = [f'{tick:.0f}' if tick == 0 or tick == 1 else f'{tick:.1f}'
                           for tick in yTicks]
            ax.set_yticks(yTicks)  # Set the positions of the ticks
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



    def plotEnrichmentScores(self, scores, dataType, title, motifFilter,
                             duplicateFigure, saveTag):
        print('============================ Plot: Enrichment Score '
              '=============================')
        print(f'Dataset:{purple} {saveTag}{resetColor}\n\n')

        # Create heatmap
        cMapCustom = NGS.createCustomColorMap(self, colorType='EM')

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
                                  cbar=True, linewidths=self.lineThickness-1,
                                  linecolor='black', square=self.figEMSquares,
                                  center=None, vmax=cBarMax, vmin=cBarMin,
                                  annot_kws={'fontweight': 'bold'})
        ax.set_xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        ax.set_ylabel('Residue', fontsize=self.labelSizeAxis)
        if self.showSampleSize:
            title += (f'\nN Unsorted = {self.nSubsInitial:,}\n'
                      f'N Sorted = {self.nSubsFinal:,}')
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
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


        # Inspect dataset
        duplicate = False
        # Save the figure
        if self.saveFigures:
            if 'Scaled' in dataType:
                datasetType = 'EM Scaled'
            elif 'Enrichment' in dataType:
                datasetType = 'EM'
            else:
                print(f'{orange}ERROR: What do I do with this dataset type -'
                      f'{cyan} {dataType}{resetColor}\n\n')
                sys.exit()

            # Define: Save location
            if motifFilter:
                duplicate = True
                figLabel = (f'{self.enzymeName} - {datasetType} '
                            f'{self.saveFigureIteration} - {saveTag} - '
                            f'MinCounts {self.minSubCount}.png')
            else:
                figLabel = (f'{self.enzymeName} - {datasetType} - '
                            f'{saveTag} - MinCounts {self.minSubCount}.png')
            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                if self.findMotif and duplicate:
                    # Turn off figure autosave
                    self.saveFigures = False
                    print(f'{yellow}WARNING{resetColor}: '
                          f'{yellow}The figure already exists at the path\n'
                          f'     {saveLocation}\n\n'
                          f'We will not overwrite the figure{resetColor}\n\n')
                else:
                    # Save duplicate figures
                    if duplicateFigure:
                        copyNumber = 1
                        fileFound = True
                        while fileFound:
                            # Define: Save location
                            figLabel = (f'{self.enzymeName} - {datasetType} '
                                        f'{copyNumber} - {saveTag} - '
                                        f'MinCounts {self.minSubCount}.png')
                            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

                            if not os.path.exists(saveLocation):
                                print(f'Saving figure at path:\n'
                                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                                fig.savefig(saveLocation, dpi=self.figureResolution)
                                self.saveFigureIteration = copyNumber
                                fileFound = False
                            else:
                                copyNumber += 1
                    else:
                        # Dont save the duplicated figure
                        print(f'{yellow}WARNING{resetColor}: '
                              f'{yellow}The figure already exists at the path\n'
                              f'     {saveLocation}\n\n'
                              f'We will not overwrite the figure{resetColor}\n\n')
            else:
                self.findMotif = False
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)



    def plotMotif(self, data, dataType, bigLettersOnTop, title, yMax, yMin, showYTicks,
                  addHorizontalLines, motifFilter, duplicateFigure, saveTag):
        print('============================= Plot: Sequence Motif '
              '==============================')
        print(f'Motif type:{purple} {dataType}{resetColor}\n'
              f'Dataset:{purple} {saveTag}{resetColor}\n\n')

        # Set local parameters
        if bigLettersOnTop:
            stackOrder = 'big_on_top'
        else:
            stackOrder = 'small_on_top'

        # Rename column headers
        dataColumnsReceived = data.columns
        data.columns = range(len(data.columns))

        # Set -inf to zero
        if data.isin([np.inf, -np.inf]).any().any():
            data.replace([np.inf, -np.inf], 0, inplace=True)
            yMin = min(data[data < 0].sum())


        # Plot the sequence motif
        fig, ax = plt.subplots(figsize=self.figSize)
        motif = logomaker.Logo(data.transpose(), ax=ax, color_scheme=self.colorsAA,
                               width=0.95, stack_order=stackOrder)
        # Set figure title
        if self.showSampleSize:
            if dataType.lower() == 'weblogo':
                title += f'\nN = {self.nSubsFinal:,}'
                if showYTicks:
                    figBorders = [0.852, 0.075, 0.112, 0.938]
                else:
                    figBorders = [0.852, 0.075, 0.112, 0.938]
            else:
                title +=  (f'\nN Unsorted = {self.nSubsInitial:,}\n'
                           f'N Sorted = {self.nSubsFinal:,}')
                figBorders = [0.852, 0.075, 0.164, 0.938]
        else:
            if dataType.lower() == 'weblogo':
                figBorders = [0.852, 0.075, 0.112, 0.938]
            else:
                figBorders = [0.852, 0.075, 0.164, 0.938]
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
        if len(dataColumnsReceived) == len(self.xAxisLabels):
            motif.ax.set_xticks([pos for pos in range(len(self.xAxisLabels))])
            motif.ax.set_xticklabels(self.xAxisLabels, fontsize=self.labelSizeTicks,
                                     rotation=0, ha='center')
        else:
            motif.ax.set_xticks([pos for pos in range(len(dataColumnsReceived))])
            motif.ax.set_xticklabels(dataColumnsReceived, fontsize=self.labelSizeTicks,
                                     rotation=0, ha='center')

        for tick in motif.ax.xaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness) # Set tick width

        # Set y-ticks
        if 'enrichment' in dataType.lower():
            yTicks = [yMin, 0, yMax]
            yTickLabels = [f'{tick:.2f}' if tick != 0 else f'{int(tick)}'
                           for tick in yTicks]
            yLimitUpper = yMax
            yLimitLower = yMin
        else:
            if showYTicks:
                yTicks = range(0, 5)
                yTickLabels = [f'{tick:.0f}' if tick != 0 else f'{int(tick)}'
                               for tick in yTicks]
                yLimitUpper = 4.32
                yLimitLower = 0

                if addHorizontalLines:
                    for tick in yTicks:
                        motif.ax.axhline(y=tick, color='black', linestyle='--',
                                         linewidth=self.lineThickness)
            else:
                yTicks = [yMin, 0, yMax]
                yTickLabels = [f'{tick:.2f}' if tick != 0 else f'{int(tick)}'
                               for tick in yTicks]
                yLimitUpper = yMax
                yLimitLower = yMin

        motif.ax.set_yticks(yTicks)
        motif.ax.set_yticklabels(yTickLabels, fontsize=self.labelSizeTicks)
        motif.ax.set_ylim(yLimitLower, yLimitUpper)
        for tick in motif.ax.yaxis.get_major_ticks():
            tick.tick1line.set_markeredgewidth(self.lineThickness) # Set tick width

        # Label the axes
        motif.ax.set_xlabel('Substrate Position', fontsize=self.labelSizeAxis)
        if dataType.lower() == 'weblogo':
            motif.ax.set_ylabel('Bits', fontsize=self.labelSizeAxis)
        else:
            motif.ax.set_ylabel(dataType, fontsize=self.labelSizeAxis)

        # Set horizontal line
        motif.ax.axhline(y=0, color='black', linestyle='-', linewidth=self.lineThickness)

        # Evaluate dataset for fixed residues
        spacer = np.diff(motif.ax.get_xticks()) # Find the space between each tick
        spacer = spacer[0] / 2

        # Use the spacer to set a gray background to fixed residues
        for index, position in enumerate(self.xAxisLabels):
            if position in self.fixedPosition:
                # Plot grey boxes on each side of the xtick
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

        # Inspect dataset
        duplicate = False
        # Save the figure
        if self.saveFigures:
            if dataType.lower() == 'weblogo':
                datasetType = dataType
            elif 'scaled' in dataType.lower():
                datasetType = 'Logo'
            else:
                print(f'{orange}ERROR: What do I do with this dataset type -'
                      f'{red} {dataType}{resetColor}\n\n')
                sys.exit()

            # Define: Save location
            if motifFilter:
                duplicate = True
                figLabel = (f'{self.enzymeName} - {datasetType} '
                            f'{self.saveFigureIteration} - {saveTag} - MinCounts '
                            f'{self.minSubCount}.png')
            else:
                figLabel = (f'{self.enzymeName} - {datasetType} - '
                            f'{saveTag} - MinCounts {self.minSubCount}.png')
            saveLocation = os.path.join(self.pathSaveFigs, figLabel)

            # Save figure
            if os.path.exists(saveLocation):
                if self.findMotif and duplicate:
                    # Turn off figure autosave
                    self.saveFigures = False
                    print(f'{yellow}WARNING{resetColor}: '
                          f'{yellow}The figure already exists at the path\n'
                          f'     {saveLocation}\n\n'
                          f'We will not overwrite the figure{resetColor}\n\n')
                else:
                    # Save duplicate figures
                    if duplicateFigure:
                        copyNumber = 1
                        fileFound = True
                        while fileFound:
                            # Define: Save location
                            figLabel = (f'{self.enzymeName} - {figLabel} {copyNumber} - '
                                        f'{saveTag} - MinCounts {self.minSubCount}.png')
                            saveLocation = os.path.join(self.pathSaveFigs, figLabel)
    
                            if not os.path.exists(saveLocation):
                                print(f'Saving figure at path:\n'
                                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                                fig.savefig(saveLocation, dpi=self.figureResolution)
                                self.saveFigureIteration = copyNumber
                                fileFound = False
                            else:
                                copyNumber += 1
                    else:
                        # Dont save the duplicated figure
                        print(f'{yellow}WARNING{resetColor}: '
                              f'{yellow}The figure already exists at the path\n'
                              f'     {saveLocation}\n\n'
                              f'We will not overwrite the figure{resetColor}\n\n')
            else:
                self.findMotif = False
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=self.figureResolution)



    def plotStats(self, countedData, totalCounts, title, datasetTag, dataType):
        print('========================= Plot: Statistical Evaluation '
              '==========================')
        # Create heatmap
        cMapCustom = NGS.createCustomColorMap(self, colorType=dataType)

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
        if self.showSampleSize:
            if totalCounts is None:
                ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
            else:
                ax.set_title(f'{title}\nN={totalCounts:,}',
                             fontsize=self.labelSizeTitle, fontweight='bold')
        else:
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
            labelText = label.get_text()  # Get the text of the label
            try:
                labelValue = float(labelText)  # Convert to a float
                if labelValue.is_integer():  # Check if it's an integer
                    cbarLabels.append(int(labelValue))  # Append as an integer
            except ValueError:
                print(f'{orange}ERROR: Unable to plot the{cyan} {dataType}{orange} '
                      f'label{cyan} {label}\n\n')
                sys.exit()
        cbar.set_ticks(cbarLabels)  # Set the positions of the ticks
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
            # Define: Save location
            figLabel = (f'EM - {self.enzymeName} - {dataType} - {datasetTag} - '
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



    def plotWordCloud(self, clusterSubs, clusterIndex, title, saveTag):
        print('=============================== Plot: Word Cloud '
              '================================')
        if clusterIndex is not None:
            print(f'Selecting PCA Population:{red} {clusterIndex + 1}{resetColor}')
        else:
            print(f'Substrates:{purple} {saveTag}{resetColor}')
        iteration = 0
        for substrate, count in clusterSubs.items():
            print(f'     {greyDark}{substrate}{resetColor}, '
                  f'Count:{red} {count:,}{resetColor}')
            iteration += 1
            if iteration == self.printNumber:
                break
        print('\n')


        # Create word cloud
        cmap = NGS.createCustomColorMap(self, colorType='Word Cloud')
        wordcloud = (WordCloud(
            width=950,
            height=800,
            background_color='white',
            min_font_size=10, # Minimum font size
            max_font_size=120, # Maximum font size
            scale=5,  # Increase scale for larger words
            colormap=cmap
        ).generate_from_frequencies(clusterSubs))


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
            # Define: Save location
            figLabel = (f'{self.enzymeName} - Words - '
                        f'{saveTag} - MinCounts {self.minSubCount}.png')
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



    def extractMotif(self, substrates, substrateFrame, frameIndicies, datasetTag):
        print('================================= Extract Motif '
              '=================================')
        print(f'Binning Substrates:{purple} {datasetTag}{resetColor}\n'
              f'Start Position:{greenLightB} {substrateFrame[frameIndicies[0]]}{resetColor}\n'
              f'   Start Index:{greenLightB} {frameIndicies[0]}{resetColor}\n'
              f'End Position:{greenLightB} {substrateFrame[frameIndicies[-1]]}{resetColor}\n'
              f'   End Index:{greenLightB} {frameIndicies[-1]}{resetColor}\n\n')
        frameLength = len(substrateFrame)
        sys.exit()


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
                fixedPosDifference = (self.fixedPosition[index] -
                                      self.fixedPosition[index - 1])
                startSubPrevious += fixedPosDifference
                # print(f'Pos Curr:{purple} {self.fixedPosition[index]}{resetColor}\n'
                #       f'Pos Prev:{purple} {self.fixedPosition[index - 1]}{resetColor}')
                # print(f'     Start Diff:{greenLightB} {fixedPosDifference}{resetColor}\n'
                #       f'     Start Prev:{greenLightB} {startSubPrevious}{resetColor}')
                startSub = index + startSubPrevious - 1
                endSub = startSub + frameLength
            else:
                startSub = startPosition
                endSub = frameIndicies[-1] + 1
            # print(f'Start:{red} {startSub}{resetColor}\n'
            #       f'Stop:{red} {endSub}{resetColor}\n\n')

            # Print substrates
            print(f'Fixed Motif:{purple} {self.fixedAA[0]}@{self.fixedPosition[index]}'
                  f'{resetColor}')
            iteration = 0
            for substrate, count in subsFixedFrame.items():
                print(f'Substrate:{greyDark} {substrate}{resetColor}\n'
                      f'    Frame:{greenLight} {substrate[startSub:endSub]}{resetColor}\n'
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

        # Print binned substrates
        iteration = 0
        print(f'Binned Substrates{resetColor}:{purple} {datasetTag}{resetColor}')
        for substrate, count in motifs.items():
            print(f'Substrate:{yellow} {substrate}{resetColor}\n'
                  f'    Count:{red} {count:,}{resetColor}')
            iteration += 1
            if iteration == self.printNumber:
                print('\n')
                break
        print(f'Total Substrates:{red} {countTotalSubstrates:,}{resetColor}\n'
              f'Unique Substrates:{red} {countUniqueSubstrates:,}{resetColor}\n\n')

        return motifs, countTotalSubstrates



    def plotBinnedSubstrates(self, substrates, countsTotal, datasetTag, dataType,
                             title, numDatapoints, barColor, barWidth):
        print('============================ Plot: Binned Substrates '
              '============================')
        xValues = []
        yValues = []
        iteration = 0
        if dataType == 'Probability':
            for substrate, count in substrates.items():
                xValues.append(str(substrate))
                yValues.append(count / countsTotal)
                iteration += 1
                if iteration == numDatapoints:
                    break
        else:
            for substrate, count in substrates.items():
                xValues.append(str(substrate))
                yValues.append(count)
                iteration += 1
                if iteration == numDatapoints:
                    break

        if dataType == 'Counts':
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
        elif dataType == 'Probability':
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
            spacer = 0.2
            yMax = math.ceil(max(yValues)) + spacer
            yMin = math.floor(min(yValues))

        # Plot the data
        fig, ax = plt.subplots(figsize=self.figSize)
        bars = plt.bar(xValues, yValues, color=barColor, width=barWidth)
        plt.ylabel(dataType, fontsize=self.labelSizeAxis)
        plt.title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        plt.axhline(y=0, color='black', linewidth=self.lineThickness)
        plt.ylim(yMin, yMax)
        # if dataType == 'Probability':
        #     plt.subplots_adjust(top=0.873, bottom=0.12, left=0.101, right=0.979)
        # elif dataType == 'Counts':
        #     plt.subplots_adjust(top=0.873, bottom=0.12, left=0.13, right=0.979)
        # else:
        #     plt.subplots_adjust(top=0.873, bottom=0.12, left=0.1, right=0.979)

        # Set the edge color
        for bar in bars:
            bar.set_edgecolor('black')

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
            figLabel = (f'{self.enzymeName} - Binned Substrates - {dataType} - '
                        f'{datasetTag} - MinCounts {self.minSubCount}.png')
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
                        motifs) else "")  # Fill missing values with empty strings
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

            # Print the current node's path and level
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
