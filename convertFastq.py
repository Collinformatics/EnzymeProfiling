import gzip
import math
import os
import pandas as pd
import sys
import time
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
# Input 1: File Parameters
inFileName = [['Mpro2-I_S1_L002_R1_001', 'Mpro2-I_S1_L003_R1_001'],
              ['Mpro2-R4_S3_L002_R1_001', 'Mpro2-R4_S3_L003_R1_001']]
inFileName = inFileName[0]
inEnzymeName = inFileName[0].split('-')[0] if isinstance(inFileName, list) \
    else inFileName.split('-')[0]
inBasePath = f'/path/{inEnzymeName}'
inFASTQPath = os.path.join(inBasePath, 'Fastq')
inSavePath = os.path.join(inBasePath, f'Data - FromFastq')
if not os.path.exists(inSavePath):
    os.makedirs(inSavePath, exist_ok=True)
inSaveAsText = False # False: save as a larger FASTA file

# Input 2: Substrate Parameters
inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
inSubstrateLength = len(inAAPositions)
inShowSampleSize = True # Include the sample size in your figures

# Input 3: Define Variables Used To Extract The Substrates
inScanRange = True
inFixResidues = False # True: fix AAs in the substrate
inFixedResidue = ['Q']
inFixedPosition = [5]
inNumberOfDatapoints = 10**6
inPrintNSubs = 10
inStartSeqR1 = 'AAAGGCAGT' # Define sequences that flank your substrate
inEndSeqR1 = 'GGTGGAAGT'
inStartSeqR2 = inStartSeqR1
inEndSeqR2 = inStartSeqR2
inPrintQualityScores = False # QSs are "phred quality" scores



# ======================================== Set Parameters ========================================
# Colors:
greyDark = '\033[38;2;144;144;144m'
white = '\033[38;2;255;255;255m'
purple = '\033[38;2;189;22;255m'
magenta = '\033[38;2;255;0;128m'
pink = '\033[38;2;255;0;242m'
cyan = '\033[38;2;22;255;212m'
blue = '\033[38;5;51m'
green = '\033[38;2;5;232;49m'
greenLight = '\033[38;2;204;255;188m'
greenDark = '\033[38;2;30;121;13m'
yellow = '\033[38;2;255;217;24m'
orange = '\033[38;2;247;151;31m'
red = '\033[91m'
resetColor = '\033[0m'


# Print options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.3f}'.format)



# =================================== Define Functions ===================================
def fastaConversion(filePath, savePath, fileNames, fileType, startSeq, endSeq):
    # Define file locations
    fileLocations = []
    saveLocations = []
    saveLocationsTxt = []
    files = []
    if isinstance(fileNames, list):
        fileTag = f'{" - ".join(fileNames)} - N Seqs'
        for fileName in fileNames:
            files.append(fileName)
            fileLocations.append(os.path.join(filePath, f'{fileName}.{fileType}'))
    else:
        fileTag = f'{fileNames} - N Seqs'
        files.append(fileNames)
        fileLocations.append(os.path.join(filePath, f'{fileNames}.{fileType}'))

    # Define save location
    if inFixResidues:
        saveLocations.append(os.path.join(
            savePath, f'{fileTag}-Fixed {fixedSubSeq}.fasta'))
        saveLocationsTxt.append(os.path.join(
            savePath, f'{fileTag}-Fixed {fixedSubSeq}.txt'))
    else:
        saveLocations.append(os.path.join(savePath, f'{fileTag}.fasta'))
        saveLocationsTxt.append(os.path.join(savePath, f'{fileTag}.txt'))

    # Evaluate: File path
    global firstRound
    data = []
    substrateCount = 0
    initialFile = True
    for indexPath, path in enumerate(fileLocations):
        if substrateCount == inNumberOfDatapoints:
            break

        if firstRound:
            print('\n=============================== Load: Fastq Files '
                  '===============================')
            print(f'Loading{purple} {files[indexPath]}{resetColor}\n'
                  f'File path:\n'
                  f'     {path}\n\n')
        if not os.path.isfile(path):
            pathZipped = f'{path}.gz'
            if os.path.isfile(pathZipped):
                path = pathZipped
            else:
                print(f'{red}ERROR: File location does not lead to a file\n'
                      f'     {path}\n'
                      f'     {pathZipped}')
                sys.exit()


        # Open the file
        openFn = gzip.open if path.endswith('.gz') else open  # Define open function
        with openFn(path, 'rt') as file:  # 'rt' = read text mode
            if '.fastq' in path or '.fq' in path:
                dataLoaded = SeqIO.parse(file, 'fastq') # Load data
                warnings.simplefilter('ignore', BiopythonWarning)
            elif '.fasta' in path or '.fa' in path:
                dataLoaded = SeqIO.parse(file, 'fasta')
                warnings.simplefilter('ignore', BiopythonWarning)
            else:
                print(f'ERROR: Unrecognized file\n     {path}')

            # Print data
            if firstRound:
                printN = 0
                for index, datapoint in enumerate(dataLoaded):
                    # Select full DNA seq
                    DNA = str(datapoint.seq)
                    QS = datapoint.letter_annotations['phred_quality']

                    # Extract substrate DNA
                    if startSeq in DNA and endSeq in DNA:
                        indexStart = DNA.find(startSeq) + len(startSeq)
                        indexEnd = DNA.find(endSeq)
                        substrate = DNA[indexStart:indexEnd].strip()
                        QSSub = QS[indexStart:indexEnd]
                        if 'N' not in substrate:
                            if len(substrate) == len(inAAPositions) * 3:
                                substrate = Seq.translate(substrate)
                                if '*' not in substrate:
                                    printData = True
                                    if inFixResidues:
                                        selectAA = substrate[inFixedPosition[0] - 1]
                                        if selectAA not in inFixedResidue:
                                            printData = False
                                    if printData:
                                        printN += 1
                                        print(f'DNA: {DNA}\n'
                                              f'QS: {QS}\n'
                                              f'Sub: {greenLight}{substrate}'
                                              f'{resetColor}\n'
                                              f'QS Sub: {QSSub}\n')
                                        if printN >= inPrintNSubs:
                                            print()
                                            break

            # Extract datapoints
            if initialFile:
                print('================================ Get Substrates '
                      '=================================')
                initialFile = False
            print(f'Selecting {red}{(inNumberOfDatapoints-substrateCount):,}{resetColor} '
                  f'substrates from file: {purple}{files[indexPath]}{resetColor}')

            if inSaveAsText:
                timeStart = time.time()
                for index, datapoint in enumerate(dataLoaded):
                    # Select full DNA seq
                    DNA = str(datapoint.seq)
                    QS = datapoint.letter_annotations['phred_quality']

                    # Extract substrate DNA
                    if startSeq in DNA and endSeq in DNA:
                        indexStart = DNA.find(startSeq) + len(startSeq)
                        indexEnd = DNA.find(endSeq)
                        substrate = DNA[indexStart:indexEnd].strip()
                        QSSub = QS[indexStart:indexEnd]
                        if 'N' not in substrate:
                            if len(substrate) == len(inAAPositions) * 3:
                                substrate = Seq.translate(substrate)
                                if '*' not in substrate:
                                    if any(score < 20 for score in QSSub):
                                        continue

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
            else:
                timeStart = time.time()
                for index, datapoint in enumerate(dataLoaded):

                    # Select full DNA seq
                    DNA = str(datapoint.seq)

                    # Extract substrate DNA
                    if startSeq in DNA and endSeq in DNA:
                        indexStart = DNA.find(startSeq) + len(startSeq)
                        indexEnd = DNA.find(endSeq)
                        substrate = DNA[indexStart:indexEnd].strip()
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
        timeEnd = time.time()
        timeRun = timeEnd-timeStart
        print(f'Extracted substrates: {red}{substrateCount:,}{resetColor}\n'
              f'Runtime: {round(timeRun, 2):,} s\n')
    if firstRound:
        firstRound = False

    # Save the data
    numDatapoints = len(data)
    if numDatapoints != 0:
        if inSaveAsText:
            savePath = saveLocationsTxt[0]
            savePath = savePath.replace('N Seqs', f'N {numDatapoints}')
            print(f'Saving a{yellow} Text{resetColor} file at: '
                  f'N = {red}{numDatapoints:,}{resetColor}\n'
                  f'     {savePath}\n\n')

            # Write data in a text file
            with open(savePath, 'w') as fileSave:
                fileSave.write('\n'.join(data))
        else:
            numDatapoints = len(data)
            savePath = saveLocations[0]
            savePath = savePath.replace('N Seqs', f'N {numDatapoints}')
            print(f'Saving a{yellow} fasta{resetColor} file at: '
                  f'N = {red}{numDatapoints:,}{resetColor}\n'
                  f'     {savePath}\n\n')

            # Write data in a fasta file
            with open(savePath, 'w') as fasta_file:
                SeqIO.write(data, fasta_file, 'fasta')
    else:
        print(f'The data was not saved, no substrates were found\n\n')
    if numDatapoints < inNumberOfDatapoints:
        print('====================================== End '
              '======================================')
        print(f'{yellow}The maximum number of substrates has been extracted '
              f'from the file(s):{purple}')
        for file in fileNames:
            print(f'     {file}')
        print(f'{resetColor}\n')
        sys.exit()



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



# ===================================== Run The Code =====================================
fixedSubSeq = fixSubstrateSequence(fixAA=inFixedResidue, fixPosition=inFixedPosition)

# Convert file
firstRound = True
print('============================= Extracting Substrates =============================')
if inScanRange:
    exponent = math.floor(math.log10(inNumberOfDatapoints))
    print(f'Saving files with N substrates:')
    for val in range(1, 10):
        inNumberOfDatapoints = val * 10 ** exponent
        print(f'     {red}{inNumberOfDatapoints:,}{resetColor}')
    print('')
    time.sleep(1)

    for val in range(1, 10):
        inNumberOfDatapoints = val*10**exponent
        fastaConversion(filePath=inFASTQPath, savePath=inSavePath, fileNames=inFileName,
                        fileType='fastq', startSeq=inStartSeqR1, endSeq=inEndSeqR1)
else:
    print(f'Saving file with N substrates:\n'
          f'     {red}{inNumberOfDatapoints:,}{resetColor}\n')
    fastaConversion(filePath=inFASTQPath, savePath=inSavePath, fileNames=inFileName,
                    fileType='fastq', startSeq=inStartSeqR1, endSeq=inEndSeqR1)
