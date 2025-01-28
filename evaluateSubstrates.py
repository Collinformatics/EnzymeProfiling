# PURPOSE: This code will load in your extracted substrates for processing

# IMPORTANT: Process all of your data with extractSubstrates before using this script


import numpy as np
import os
import sys
import pickle as pk
import pandas as pd
import threading
import time
from functions import filePaths, NGS



# ===================================== User Inputs ======================================
# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inBaseFilePath = '/path/folder'
inFilePath = f'{inBaseFilePath}/{inEnzymeName}/Extracted Data'
inSavePath = inFilePath
inSavePathFigures = f'{inBaseFilePath}/{inEnzymeName}/Figures'
inFileNamesInitialSort, inFileNamesFinalSort, inAAPositions = filePaths(enzyme=inEnzymeName)
inSaveFigures = True

# Input 2: Processing The Data
inPlotPositionalEntropy = True
inPlotEnrichmentMap = True
inPlotEnrichmentMotif = True
inPlotWeblogoMotif = False
inPlotWordCloud = True
inPlotSubstrateRF = False
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
                       # If False: use "inFileNamesInitialSort" for baseline probabilities

# Input 3: Computational Parameters
inFixResidues = False # True: fix AAs in the substrate, False: Don't fix AAs, plot raw the data
inFixedResidue = ['Q']
inFixedPosition = [5]
inExcludeResidues = False # Do you want to remove any AAs from your collection of substrate
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inMinDeltaS = 0.45
inFixEntireSubstrateFrame = True
inMinimumSubstrateCount = 10
inPrintFixedSubs = True
inFigureTitleSize = 18
inFigureLabelSize = 16
inFigureTickSize = 13
inEvaluateOS = False
inMaxResidueCount = 4
inShowSampleSize = True # Include the sample size in your figures
inSubstrateLength = len(inAAPositions)

# Input 4: PCA
inRunPCA = False
inBinSubsPCA = False
inIndexNTerminus = 2 # Define bounds for binned substrate
inBinnedSubstrateLenght = 5 # Define the length of you substrate
inFramePositons = [inIndexNTerminus-1,
                   inIndexNTerminus+inBinnedSubstrateLenght-1]
inAAPositionsBinned = inAAPositions[inFramePositons[0]:inFramePositons[-1]]
inNumberOfPCs = 2
inTotalSubsPCA = int(10000)
inEncludeSubstrateCounts = False
inExtractPopulations = False
inPlotPositionalEntropyPCAPopulations = False
inAdjustZeroCounts = False # Prevent counts of 0 in PCA EM & Motif

# Input 5: Probability Distributions
inDFDistMaxY = 0.35

# Input 6: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False
inTitleEnrichmentMap = inEnzymeName
inYLabelEnrichmentMap = 2 # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter
inPrintSelectedSubstrates = 1 # Set = 1, to print substrates with fixed residue
inFigureSize = (9.5, 8) # (width, height)
if inBinSubsPCA:
    inFigureBorders = [0.852, 0.075, 0.117, 0.998] # Top, bottom, left, right
else:
    inFigureBorders = [0.882, 0.075, 0.117, 0.998]
inFigureAsSquares = (4.5, 8)
inFigureBordersAsSquares = [0.882, 0.075, 0.075, 0.943]
inEnrichmentColorMap = ['navy','royalblue','dodgerblue','lightskyblue','white','white',
                        'lightcoral','red','firebrick','darkred']

# Input 7: Plot Sequence Motif
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

# Input 8: Evaluate Substrate Enrichment
inEvaluateSubstrateEnrichment = False
inSaveEnrichedSubstrates = False
inNumberOfSavedSubstrates = 10**6

# Input 9: Evaluate Specificity
inPlotShannonEntropy = False
inCompairRF = False # Plot RF distirbutions of a given AA
inCompairAA = 'V' # Select the AA of interest
inCompairYMax = 0.4 # Set the y-axis for the RF compairson figure
inCompairYMin = 0.0



# =================================== Setup Parameters ===================================
startTime = time.time()
if inShowEnrichmentAsSquares:
    # Set figure dimension when plotting EM plots as squares
    figSizeEM = inFigureAsSquares
    figBordersEM = inFigureBordersAsSquares
else:
    # Set figure dimension when plotting EM plots as rectangles
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


# Colors:
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
ngs = NGS(enzymeName=inEnzymeName, substrateLength=inSubstrateLength,
          fixedAA=inFixedResidue, fixedPosition=inFixedPosition,
          minCounts=inMinimumSubstrateCount, colorsCounts=inCountsColorMap,
          colorStDev=inStDevColorMap, colorsEM=inEnrichmentColorMap,
          colorsMotif=inLetterColors, xAxisLabels=inAAPositions,
          xAxisLabelsBinned=inAAPositionsBinned, residueLabelType=inYLabelEnrichmentMap,
          titleLabelSize=inFigureTitleSize, axisLabelSize=inFigureLabelSize,
          tickLabelSize=inFigureTickSize, printNumber=inPrintNumber,
          showNValues=inShowSampleSize, saveFigures=inSaveFigures, savePath=inSavePath,
          savePathFigs=inSavePathFigures, setFigureTimer=None)
