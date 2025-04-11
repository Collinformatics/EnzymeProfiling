import gzip
import os
import pandas as pd
import pickle as pk
import sys
import threading
import time
import warnings
from Bio import SeqIO
from Bio import BiopythonWarning
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from functions import NGS



# ========================================== User Inputs =========================================
# Input 1: File Location Information
inFileName = ['Mpro2-R4_S3_L002_R1_001', 'Mpro2-R4_S3_L003_R1_001']
inEnzymeName = inFileName[0].split('-')[0]
inBasePath = f'/Users/ca34522/Documents/Research/NGS/{inEnzymeName}/Fastq'
inSavePath = inBasePath

# Input 2: Substrate Parameters
inEnzymeName = inFileName[0].split('-')[0]
inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
inSubstrateLength = len(inAAPositions)
inShowSampleSize = True # Include the sample size in your figures

# Input 3: Define Variables Used To Extract The Substrates
inFixResidues = True # True: fix AAs in the substrate, False: Don't fix AAs, plot raw the data
inFixedResidue = ['Q']
inFixedPosition = [5]
inNumberOfDatapoints = 20
inSaveAsText = False # False: save as FASTA
inStartSeqR1 = 'AAAGGCAGT' # Define sequences that flank your substrate
inEndSeqR1 = 'GGTGGAAGT'
inStartSeqR2 = 'AAAGGCAGT' # KGS: AAAGGCAG
inEndSeqR2 = 'GGTGGAAGT' # GGS: GGTGGAAGT
inPrintQualityScores = False # QSs are "phred quality" scores



# ======================================== Set Parameters ========================================
# Colors:
white = '\033[38;2;255;255;255m'
red = '\033[91m'
orange = '\033[38;2;247;151;31m'
yellow = '\033[38;2;255;217;24m'
green = '\033[38;2;5;232;49m'
purple = '\033[38;2;230;0;255m'
resetColor = '\033[0m'

# Print options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)



# ======================================= Define Functions =======================================
def fastaConversion(filePath, fileNames, fileType, startSeq, endSeq, printQS):
    # Define file locations
    fileLocations = []
    saveLocations = []
    saveLocationsTxt = []
    for fileName in fileNames:
        if '/' in filePath:
            fileLocations.append(filePath + '/' + fileName + '.' + fileType)
            if inFixResidues:
                saveLocations.append(filePath + '/' + fileName + ' Fixed ' +
                                     fixedSubSeq + '.fasta')
                saveLocationsTxt.append(filePath + '/' + fileName + ' Fixed ' +
                                     fixedSubSeq + '.txt')
            else:
                saveLocations.append(filePath + '/' + fileName + '.fasta')
                saveLocationsTxt.append(filePath + '/' + fileName + '.txt')
        else:
            fileLocations.append(filePath + '\\' + fileName + '.' + fileType)
            if inFixResidues:
                saveLocations.append(filePath + '\\' + fileName + ' Fixed ' +
                                     fixedSubSeq + '.fasta')
                saveLocationsTxt.append(filePath + '\\' + fileName + ' Fixed ' +
                                     fixedSubSeq + '.txt')
            else:
                saveLocations.append(filePath + '\\' + fileName + '.fasta')
                saveLocationsTxt.append(filePath + '\\' + fileName + '.txt')

    # Evaluate file path
    gZipped = False
    for indexPath, path in enumerate(fileLocations):
        print('=================================== Load: Fastq Files '
              '===================================')
        if not os.path.isfile(path):
            pathZipped = path + '.gz'
            if os.path.isfile(pathZipped):
                gZipped = True
                path = pathZipped
            else:
                print(f'{red}ERROR: File location does not lead to a file\n'
                      f'     {path}\n'
                      f'     {pathZipped}')
                sys.exit()
        print(f'Loading{purple} FASTQ{resetColor} file at:\n'
              f'     {path}\n\n')

        # Load data
        data = []
        if gZipped:
            # Open the file
            with (gzip.open(path, 'rt', encoding='utf-8') as file):
                dataLoaded = SeqIO.parse(file, fileType)
                warnings.simplefilter('ignore', BiopythonWarning)

                # Extract datapoints
                substrateCount = 0
                if inSaveAsText:
                    for index, datapoint in enumerate(dataLoaded):
                        # Select full DNA seq
                        DNA = str(datapoint.seq)

                        # Extract substrate DNA
                        if startSeq in DNA and endSeq in DNA:
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)  # Find the substate end index
                            substrate = DNA[start:end].strip()  # Extract substrate DNA seq
                            if 'N' not in substrate:
                                if len(substrate) == len(inAAPositions) * 3:
                                    substrate = Seq.translate(substrate)
                                    if '*' not in substrate:
                                            substrateCount += 1
                                            data.append(substrate)
                        # data.append(SeqRecord(seq=datapoint.seq, id=datapoint.id))
                        if substrateCount == inNumberOfDatapoints:
                            break


                    # Extract fixed substrates
                    if inFixResidues:
                        fixedData = []
                        for substrate in data:
                            selectAA = substrate[inFixedPosition[0] - 1]
                            if selectAA in inFixedResidue:
                                fixedData.append(substrate)
                        data = fixedData

                    # Save the data as text files
                    print('=================================== Save: Fasta Files '
                          '===================================')
                    numDatapoints = len(data)
                    print(f'{purple}Text{resetColor} file:{red} {numDatapoints:,}'
                          f'{resetColor} substrates\n')
                    if numDatapoints != 0:
                        savePath = saveLocationsTxt[indexPath]
                        with open(savePath, 'w') as fileSave:
                            for substrate in data:
                                fileSave.write(f'{substrate}\n')

                        # print(f'Fasta:\n{data}')
                        print(f'Saving{purple} Text{resetColor} file at:\n'
                              f'     {savePath}\n\n')
                    else:
                        print(f'The data was not saved, no substrates were found\n\n')
                else:
                    for index, datapoint in enumerate(dataLoaded):
                        # Select full DNA seq
                        DNA = str(datapoint.seq)

                        # Extract substrate DNA
                        if startSeq in DNA and endSeq in DNA:
                            start = DNA.find(startSeq) + len(startSeq)
                            end = DNA.find(endSeq)  # Find the substate end index
                            substrate = DNA[start:end].strip()  # Extract substrate DNA seq
                            if len(substrate) == len(inAAPositions) * 3:
                                substrate = Seq.translate(substrate)
                                if '*' not in substrate:
                                    substrateCount += 1
                                    data.append(SeqRecord(seq=substrate, id=datapoint.id))
                        # data.append(SeqRecord(seq=datapoint.seq, id=datapoint.id))
                        if substrateCount == inNumberOfDatapoints:
                            break


                    # Extract fixed substrates
                    if inFixResidues:
                        fixedData = []
                        for substrate in data:
                            selectAA = substrate[inFixedPosition[0] - 1]
                            if selectAA in inFixedResidue:
                                fixedData.append(substrate)
                        data = fixedData

                    # Save the data as fasta files
                    print('=================================== Save: Fasta Files '
                          '===================================')
                    numDatapoints = len(data)
                    print(f'{purple}FASTA{resetColor} file:{red} {numDatapoints:,}'
                          f'{resetColor} substrates\n')
                    if numDatapoints != 0:
                        savePath = saveLocations[indexPath]
                        with open(savePath, 'w') as fasta_file:
                            SeqIO.write(data, fasta_file, 'fasta')

                        # print(f'Fasta:\n{data}')
                        print(f'Saving{purple} FASTA{resetColor} file at:\n'
                              f'     {savePath}\n\n')
                    else:
                        print(f'The data was not saved, no substrates were found\n\n')




def fixSubstrateSequence(fixAA, fixPosition):
    fixResidueList = []
    for index in range(len(fixAA)):
        fixResidueList.append(f'{fixAA[index]}@R{fixPosition[index]}')

    fixedSeq = '_'.join([f'{seq}' for seq in fixResidueList])

    # Condence the string
    if "'" in fixedSeq:
        fixedSeq = fixedSeq.replace("'", '')
    if " " in fixedSeq:
        fixedSeq = fixedSeq.replace(" ", '')

    return fixedSeq



# ========================================= Run The Code =========================================
fixedSubSeq = fixSubstrateSequence(fixAA=inFixedResidue, fixPosition=inFixedPosition)

fastaConversion(filePath=inBasePath, fileNames=inFileName, fileType='fastq',
                startSeq=inStartSeqR1, endSeq=inEndSeqR1, printQS=inPrintQualityScores)
