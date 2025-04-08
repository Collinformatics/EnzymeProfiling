# PURPOSE: Load in a file (fastq or fasta), extract substrate sequences, and count AA
           # occurrences at each position in the substrates
# IMPORTANT: File names must include information regarding the direction of expression
    # Forward Read: denoted by "R1"
    # Reverse Read: denoted by "R2"
# Designating the sorted library
    # If there is an "F" or "R4" in the file names of your final sort, ignore this
    # If not, then find the line "# Define: File type" and add in a string that is
        # unique to your final sort files to the conditional statement, so that we
        # correctly define "fileType"


import os
import pickle as pk
from functions import NGS



# ===================================== User Inputs ======================================
# Input 1: File Location
inFileName = ['Src-F_S2_L001_R1_001', 'Src-F_S2_L001_R2_001'] # Define file name(s)
inEnzymeName = inFileName[0].split('-')[0]
inBasePath = f'/Users/ca34522/Documents/Research/NGS/{inEnzymeName}'
inFilePath = os.path.join(inBasePath, 'Fastq') # Define the fastq folder pathway
inFileType = 'fastq' # Define the file type
inSavePath = os.path.join(inBasePath, 'Extracted Data')
inSaveFileName = 'Src-F_S2_L001'
inAlertPath = '/Users/ca34522/Documents/Python/Sounds/Bells.mp3'

# Input 2: Substrate Parameters
inAAPositions = ['R1','R2','R3','R4','R5','R6','R7','R8']
inShowSampleSize = True # Include the sample size in your figures
inCodonSequence = 'NNS' # Base probabilities of degenerate codons (can be N, S, or K)

# Input 3: Define Variables Used To Extract The Substrates
inFixedLibrary = False
inFixedResidue = []
inFixedPosition = []
inPrintNumber = 10 # Print peptide sequences to validate substrate extraction
inStartSeqR1 = 'AAAGGCAGT' # Define DNA sequences that flank your substrate
inEndSeqR1 = 'GGTGGAAGT'
inStartSeqR2 = inStartSeqR1 # KGS: AAAGGCAGT
inEndSeqR2 = inEndSeqR1 # GGS: GGTGGAAGT
inPrintQualityScores = False # QSs are "phred quality" scores

# Input 4: Plotting The Data
inPrintCounts = True
inPlotCounts = True
inCountMapYLabel = 2  # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter
inCountsColorMap = ['white','white','#FF76FA','#FF50F9','#FF00F2','#CA00DF','#BD16FF']
inFigureTitleSize = 18
inFigureLabelSize = 16
inFigureTickSize = 13
inFigureSize = (10, 8) # (width, height)
inFigureBorders = [0.882, 0.075, 0.18, 1] # Top, bottom, left, right



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=inEnzymeName, substrateLength=len(inAAPositions),
          fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          excludeAAs=None, excludeAA=None, excludePosition=None,
          minCounts=0, colorsCounts=inCountsColorMap, colorStDev=None,
          colorsEM=None, colorsMotif=None, xAxisLabels=inAAPositions,
          xAxisLabelsBinned=None, residueLabelType=inCountMapYLabel,
          titleLabelSize=inFigureTitleSize, axisLabelSize=inFigureLabelSize,
          tickLabelSize=inFigureTickSize, printNumber=inPrintNumber,
          showNValues=inShowSampleSize, saveFigures=False, savePath=None,
          savePathFigs=None, setFigureTimer=None)



# ===================================== Run The Code =====================================
# Initialize substrate dictionary
substrates = {}

# Extract the substrates
loadR1 = False
loadR2 = False
substrates = {}
substratesR1 = {}
substratesR2 = {}
for fileName in inFileName:
    if '_R1_' in fileName:
        substratesR1 = ngs.loadAndTranslate(filePath=inFilePath, fileName=fileName,
                                            fileType=inFileType, fixedSubs=inFixedLibrary,
                                            startSeq=inStartSeqR1, endSeq=inEndSeqR1,
                                            printQS=inPrintQualityScores)
        loadR1 = True
    elif '_R2_' in fileName:
        substratesR2 = ngs.loadAndTranslate(filePath=inFilePath, fileName=fileName,
                                            fileType=inFileType, fixedSubs=inFixedLibrary,
                                            startSeq=inStartSeqR2, endSeq=inEndSeqR2,
                                            printQS=inPrintQualityScores)
        loadR2 = True


# Combine substrate dictionaries
if loadR1 and loadR2:
    for key, value in substratesR1.items():
        if key in substrates:
            substrates[key] += value
        else:
            substrates[key] = value
    for key, value in substratesR2.items():
        if key in substrates:
            substrates[key] += value
        else:
            substrates[key] = value
elif loadR1:
    substrates = substratesR1
elif loadR2:
    substrates = substratesR2

# Define: File type
if 'F' or 'R4' in inFileName[0]:
    fileType = 'Final Sort'
else:
    fileType = 'Initial Sort'

# Count the occurrences of each residue
counts, totalSubs = ngs.countResidues(substrates=substrates,
                                      printCounts=inPrintCounts,
                                      datasetType=fileType)

# Notification: Finish analysis
ngs.alert(soundPath=inAlertPath)

# Save the data
if '/' in inFilePath:
    savePathSubstrate = inSavePath+'/substrates_'+inSaveFileName
    savePathCounts = inSavePath+'/counts_'+inSaveFileName
else:
    savePathSubstrate = inSavePath + '\\substrates_' + inSaveFileName
    savePathCounts = inSavePath + '\\counts_' + inSaveFileName
with open(savePathSubstrate, 'wb') as file:
    pk.dump(substrates, file)
counts.to_csv(savePathCounts)
print('================================ Saving The Data ================================')
print(f'Data saved at:\n'
      f'    {savePathSubstrate}\n'
      f'    {savePathCounts}\n\n')

# Display extraction efficiency
ngs.extractionEffiency(files=inFileName)

# Plot the data
if inPlotCounts:
    ngs.plotCounts(countedData=counts, totalCounts=totalSubs,
                   title=f'{inEnzymeName}\n{inSaveFileName}', figSize=inFigureSize)
