import gzip
import os
import pandas as pd
import sys
import warnings
from Bio import SeqIO
from Bio import BiopythonWarning
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



# This script will convert a FASTQ file to a FASTA, or Text file
# FASTA:
    # Pros: Contains more information than the .txt file
    # Cons: Larger file size than .txt
# Text:
    # Pros: Lighter file sizes with faster uploads speeds
    # Cons: Less information than the FASTA, as we only save the substrate sequences



# ===================================== User Inputs ======================================
# Input 1: File Location Information
inFileName = ['Mpro2-R4_S3_L002_R1_001', 'Mpro2-R4_S3_L003_R1_001']
inEnzymeName = inFileName[0].split('-')[0]
inBasePath = f'/path/{inEnzymeName}'
inFASTQPath = os.path.join(inBasePath, 'Fastq')
inSavePath = os.path.join(inBasePath, 'Extracted Data')

# Input 2: Substrate Parameters
inEnzymeName = inFileName[0].split('-')[0]
inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
inSubstrateLength = len(inAAPositions)
inShowSampleSize = True # Include the sample size in your figures

# Input 3: Define Variables Used To Extract The Substrates
inFixResidues = True # True: fix AAs in the substrate
inFixedResidue = ['Q']
inFixedPosition = [5]
inNumberOfDatapoints = 100
inSaveAsText = True # False: save as a larger FASTA file
inStartSeqR1 = 'AAAGGCAGT' # Define sequences that flank your substrate
inEndSeqR1 = 'GGTGGAAGT'
inStartSeqR2 = inStartSeqR1
inEndSeqR2 = inStartSeqR2
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



# =================================== Define Functions ===================================
def fastaConversion(filePath, savePath, fileNames, fileType, startSeq, endSeq, printQS):
    # Define file locations
    fileLocations = []
    saveLocations = []
    saveLocationsTxt = []
    for fileName in fileNames:
        fileLocations.append(os.path.join(filePath, f'{fileName}.{fileType}'))
        if inFixResidues:
            saveLocations.append(os.path.join(
                savePath,
                f'{fileName}-Fixed {fixedSubSeq}-N {inNumberOfDatapoints}.fasta'))
            saveLocationsTxt.append(os.path.join(
                savePath,
                f'{fileName}-Fixed {fixedSubSeq}-N {inNumberOfDatapoints}.txt'))
        else:
            saveLocations.append(os.path.join(savePath, fileName, '.fasta'))
            saveLocationsTxt.append(os.path.join(savePath, fileName, '.txt'))


    # Evaluate file path
    gZipped = False
    for indexPath, path in enumerate(fileLocations):
        print('=============================== Load: Fastq Files '
              '===============================')
        if not os.path.isfile(path):
            pathZipped = f'{path}.gz'
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
                            indexStart = DNA.find(startSeq) + len(startSeq)
                            indexEnd = DNA.find(endSeq)
                            substrate = DNA[indexStart:indexEnd].strip()
                            if 'N' not in substrate:
                                if len(substrate) == len(inAAPositions) * 3:
                                    substrate = Seq.translate(substrate)
                                    if '*' not in substrate:
                                        if inFixResidues:
                                            selectAA = substrate[inFixedPosition[0] - 1]
                                            if selectAA in inFixedResidue:
                                                substrateCount += 1
                                                data.append(substrate)
                                        else:
                                            substrateCount += 1
                                            data.append(substrate)
                        if substrateCount == inNumberOfDatapoints:
                            break

                    # Save the data as text files
                    print('=============================== Save: Fasta Files '
                          '===============================')
                    numDatapoints = len(data)
                    print(f'Extracted datapoints:{red} {numDatapoints:,}'
                          f'{resetColor} substrates\n')
                    if numDatapoints != 0:
                        savePath = saveLocationsTxt[indexPath]
                        with open(savePath, 'w') as fileSave:
                            for substrate in data:
                                fileSave.write(f'{substrate}\n')

                        print(f'Saving a{yellow} Text{resetColor} file at:\n'
                              f'     {savePath}\n\n')
                    else:
                        print(f'The data was not saved, no substrates were found\n\n')
                else:
                    for index, datapoint in enumerate(dataLoaded):
                        # Select full DNA seq
                        DNA = str(datapoint.seq)

                        # Extract substrate DNA
                        if startSeq in DNA and endSeq in DNA:
                            indexStart = DNA.find(startSeq) + len(startSeq)
                            indexEnd = DNA.find(endSeq)
                            substrate = DNA[indexStart:indexEnd].strip()  # Extract substrate DNA
                            if len(substrate) == len(inAAPositions) * 3:
                                substrate = Seq.translate(substrate)
                                if '*' not in substrate:
                                    if inFixResidues:
                                        selectAA = substrate[inFixedPosition[0] - 1]
                                        if selectAA in inFixedResidue:
                                            substrateCount += 1
                                            data.append(SeqRecord(seq=substrate,
                                                                  id=datapoint.id))
                                    else:
                                        substrateCount += 1
                                        data.append(SeqRecord(seq=substrate,
                                                              id=datapoint.id))
                        if substrateCount == inNumberOfDatapoints:
                            break


                    # Save the data as fasta files
                    print('=============================== Save: Fasta Files '
                          '===============================')
                    numDatapoints = len(data)
                    print(f'Extracted datapoints:{red} {numDatapoints:,}'
                          f'{resetColor} substrates\n')
                    if numDatapoints != 0:
                        savePath = saveLocations[indexPath]
                        with open(savePath, 'w') as fasta_file:
                            SeqIO.write(data, fasta_file, 'fasta')

                        print(f'Saving a{yellow} FASTA{resetColor} file at:\n'
                              f'     {savePath}\n\n')
                    else:
                        print(f'The data was not saved, no substrates were found\n\n')



def fixSubstrateSequence(fixAA, fixPosition):
    fixResidueList = []
    for index in range(len(fixAA)):
        fixResidueList.append(f'{fixAA[index]}@R{fixPosition[index]}')

    fixedSeq = '_'.join([f'{seq}' for seq in fixResidueList])

    # Condense the string
    if "'" in fixedSeq:
        fixedSeq = fixedSeq.replace("'", '')
    if " " in fixedSeq:
        fixedSeq = fixedSeq.replace(" ", '')

    return fixedSeq



# ========================================= Run The Code =========================================
fixedSubSeq = fixSubstrateSequence(fixAA=inFixedResidue, fixPosition=inFixedPosition)

fastaConversion(filePath=inFASTQPath, savePath=inSavePath, fileNames=inFileName,
                fileType='fastq', startSeq=inStartSeqR1, endSeq=inEndSeqR1,
                printQS=inPrintQualityScores)
