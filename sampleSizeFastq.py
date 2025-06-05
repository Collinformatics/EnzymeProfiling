import gzip
import os
import pandas as pd
import sys
import warnings
from Bio import SeqIO
from Bio import BiopythonWarning



# This script will load a FASTQ and count the number of DNA sequences



# ===================================== User Inputs ======================================
# Input 1: File Location Information 'Mpro2-I_S1_L001_R1_001',
inFileName = ['Mpro2-I_S1_L001_R2_001',
              'Mpro2-I_S1_L002_R1_001', 'Mpro2-I_S1_L002_R2_001',
              'Mpro2-I_S1_L003_R1_001', 'Mpro2-I_S1_L003_R2_001',
              'Mpro2-I_S1_L004_R1_001', 'Mpro2-I_S1_L004_R2_001']
inEnzymeName = inFileName[0].split('-')[0]
inBasePath = f'/Users/ca34522/Documents/Research/NGS/{inEnzymeName}'
inFASTQPath = os.path.join(inBasePath, 'Fastq')



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
def countSequences(filePath, fileNames, fileType):
    print('=============================== Load: Fastq Files '
          '===============================')
    totalSequences = 0

    # Define file locations
    fileLocations = []

    for fileName in fileNames:
        fileLocations.append(os.path.join(filePath, f'{fileName}.{fileType}'))

    # Evaluate: File path
    gZipped = False
    for indexPath, path in enumerate(fileLocations):
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
              f'     {path}')

        # Load data
        totalSeqsInFile = 0
        if gZipped:
            # Open the file
            with (gzip.open(path, 'rt', encoding='utf-8') as file):
                dataLoaded = SeqIO.parse(file, fileType)
                warnings.simplefilter('ignore', BiopythonWarning)

                # Count sequences
                for _ in dataLoaded:
                    totalSeqsInFile += 1
            print(f'N Seqs: {red}{totalSeqsInFile:,}{resetColor}\n')
            totalSequences += totalSeqsInFile
    print(f'Total DNA Sequences: {red}{totalSequences:,}{resetColor}\n\n')



# ========================================= Run The Code =========================================
countSequences(filePath=inFASTQPath, fileNames=inFileName, fileType='fastq')
