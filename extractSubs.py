# PURPOSE: Load in a file (fastq or fasta), extract substrate sequences, and count AA
           # occurrences at each position in the substrates
# IMPORTANT: File names must include information regarding the direction of expression
    # Forward Read: denoted by "R1"
    # Reverse Read: denoted by "R2"
# Designating the sorted library
    # If there is an "F" or "R4" in the file names of your final sort, ignore this
    # If not, then find the line: "# Define: File type"
        # and add in a string that is unique to your final sort files to the
        # conditional statement, so that we correctly define "fileType"


import os
from functions import NGS



# ===================================== User Inputs ======================================
# Input 1: File Location
inFileName = ['Fyn-I_S6_L001_R1_001', 'Fyn-I_S6_L001_R2_001'] # Define file name(s)
inEnzymeName = inFileName[0].split('-')[0]
inPathFolder = f'/path/{inEnzymeName}'
inFilePath = os.path.join(inPathFolder, 'Fastq') # Define the fastq folder name
inFileType = 'fastq' # Define the file type

# Input 2: Saving The Data
inSaveFileName = 'Fyn-I_S6_L001' # Add this name to filePaths(enzyme) in functions.py

# Input 3: Substrate Parameters
inAAPositions = ['R1','R2','R3','R4','R5','R6','R7','R8']

# Input 4: Substrate Recognition
inPrintNumber = 10
inStartSeqR1 = 'AAAGGCAGT' # Define DNA sequences that flank your substrate
inEndSeqR1 = 'GGTGGAAGT' # KGS: AAAGGCAGT, GGS: GGTGGAAGT
inStartSeqR2 = inStartSeqR1
inEndSeqR2 = inEndSeqR1

# Input 5: Define Variables Used To Extract The Substrates
inFixedLibrary = False
inFixedResidue = ['']
inFixedPosition = []

# Input 6: Miscellaneous
inAlertPath = '/path/Bells.mp3' # Play a sound to let you know the script is done
inPrintQualityScores = False # Phred quality scores



# =================================== Initialize Class ===================================
ngs = NGS(enzymeName=inEnzymeName, substrateLength=len(inAAPositions),
          filterSubs=inFixedLibrary, fixedAA=inFixedResidue,
          fixedPosition=inFixedPosition, excludeAAs=None, excludeAA=None,
          excludePosition=None, minCounts=0, figEMSquares=False,
          xAxisLabels=inAAPositions, printNumber=inPrintNumber, showNValues=True,
          bigAAonTop=False, findMotif=False, folderPath=inPathFolder, filesInit=None,
          filesFinal=None, plotPosS=False, plotFigEM=False, plotFigLogo=False,
          plotFigWebLogo=False, plotFigWords=False, saveFigures=False,
          setFigureTimer=None, expressDNA=True)



# ===================================== Run The Code =====================================
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
                                      datasetType=fileType)

# Notification: Finish analysis
ngs.alert(soundPath=inAlertPath)

# Save the data
ngs.saveData(substrates=substrates, counts=counts, saveTag=inSaveFileName)

# Display extraction efficiency
ngs.extractionEfficiency(files=inFileName)

# Plot the data
ngs.plotCounts(countedData=counts, totalCounts=totalSubs)
