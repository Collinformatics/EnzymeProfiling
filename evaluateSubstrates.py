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



# ========================================== User Inputs =========================================
# Input 1: File Location
inEnzymeName = 'Mpro2'
inBaseFilePath = 'path/folder'
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
inFixedResidue = ['Q'] # ['L','M'],['L','F','W','Y'],
inFixedPosition = [5]
inExcludeResidues = False # Do you want to remove any AAs from your collection of substrate
inExcludedResidue = ['A','A']
inExcludedPosition = [9,10]
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
