from functions import pressKey
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


inEnzymeName = 'Mpro2'
inFixedFramePositions = ['P4', 'P3', 'P2', 'P1', 'P1\'', 'P2\'', 'P3\'', 'P4\''] #

# Input 3: Computational Parameters
inFixResidues = True # True: fix AAs in the substrate, False: Don't fix AAs, plot raw the data
inFixedResidue = ['Q']
inFixedPositions = [4, 5, 6]
inFramePositons = [0, 7] # Define bounds for the fixed frame
inPrintFixedSubs = True
inPredictSubstrateEnrichmentScores = True
inExcludeResidues = False # Do you want to remove any AAs from your collection of substrate
inExcludedResidue = ['Q']
inExcludedPosition = [8]
inFigureTitleSize = 18
inFigureLabelSize = 16
inFigureTickSize = 13
inLineThickness = 1.5
inTickLength = 4

# Input 4: Processing The Data
inPlotEnrichmentMotif = False
inPlotEnrichmentMap = False
inPredictKnownSubs = True
inPlotBinnedSubCounts = True
inPlotBinnedSubEnrichment = True
inPlotCounts = False
inCountsColorMap = ['white', 'white', 'lightcoral', 'red', 'firebrick', 'darkred']
inShowSampleSize = False # Include the sample size in your figures

# Input 5: Printing The Data
inPrintLoadedSubs = True
inPrintSampleSize = True
inPrintCounts = True
inPrintRF = True
inPrintEntopy = True
inPrintEnrichmentData = True
inPrintMotifData = True
inPrintNumber = 10
inCodonSequence = 'NNS' # Base probabilities of degenerate codons (can be N, S, or K)
inUseCodonProb = False # If True: use "inCodonSequence" for baseline probabilities

# Input 6: Plot Heatmap
inShowEnrichmentScores = True
inShowEnrichmentAsSquares = False
inTitleEnrichmentMap = (f'{inEnzymeName}\n'
                        f'Fixed Frame {inFixedResidue[0]}'
                        f'@R{inFixedPositions[0]}-R{inFixedPositions[-1]}')
inTitleEnrichmentMap = f'{inEnzymeName}'
inTitleEnrichmentMapScaled = (f'{inEnzymeName}\n'
                              f'Scaled Fixed Frame {inFixedResidue[0]}'
                              f'@R{inFixedPositions[0]}-R{inFixedPositions[-1]}')
inYLabelEnrichmentMap = 2 # 0 for full Residue name, 1 for 3-letter code, 2 for 1 letter
inPrintSelectedSubstrates = 1 # Set = 1, to print substrates with fixed residue
inFigureSize = (9.5, 8) # (width, height)
inFigureBorders = [0.882, 0.075, 0.117, 0.967] # Top, bottom, left, right
inFigureAsSquares = (4.5, 8)
inFigureBordersAsSquares = [0.94, 0.075, 0.075, 0.943]
inEnrichmentColorMap = ['navy', 'royalblue', 'dodgerblue', 'lightskyblue', 'white', 'white',
                        'lightcoral', 'red', 'firebrick', 'darkred']

# Input 7: Plot Sequence Motif
inNormLetters = False # Normalize fixed letter heights
inPlotWeblogoMotif = False
inShowWeblogoYTicks = True
inAddHorizontalLines = False
inPlotNegativeWeblogoMotif = False
inMotifTitle = inEnzymeName
inBigLettersOnTop = True
inFigureSizeMotif = inFigureSize
inFigureBordersMotifYTicks = [0.94, 0.075, 0.07, 0.98] # [top, bottom, left, right]
inFigureBordersMotifMaxYTick = [0.94, 0.075, 0.102, 0.98]
inFigureBordersNegativeWeblogoMotif = [0.94, 0.075, 0.078, 0.98]
inFigureBordersEnrichmentMotif = [0.94, 0.075, 0.138, 0.98]
inLetterColors = ['darkgreen', 'firebrick', 'deepskyblue', 'pink', 'navy', 'black', 'gold']
                  # Aliphatic, Acidic, Basic, Hydroxyl, AmMpro22, Aromatic, Sulfur

# Input 8: Evaluate Known Substrates
inKnownSubs = ['VVLQSGFR', 'VVMQSGFR', 'VVVQSGFR', 'VVGQSGFR', 'VVLHSGFR', 'VVLMSGFR',
               'VVLYSGFR', 'IVLQSGFR', 'KVLQSGFR', 'VYLQSGFR', 'VGLQSGFR', 'VVLQAGFR',
               'VVLQNGFR', 'VVLQIGFR', 'VVLQSPFR',
               'AVLQSGFR', 'VTFQSAVK', 'ATVQSKMS', 'ATLQAIAS', 'VKLQNNEL', 'VRLQAGNA',
               'PMLQSADA', 'TVLQAVGA', 'ATLQAENV', 'TRLQSLEN', 'PKLQSSQA',
               'VTFQGKFK', 'PLMQSADA', 'PKLQASQA']
inSubsPred = ['VVLQSGFR', 'VVMQSGFR', 'VVVQSGFR', 'VVGQSGFR', 'VVLHSGFR', 'VVLMSGFR',
              'VVLYSGFR', 'IVLQSGFR', 'KVLQSGFR', 'VYLQSGFR', 'VGLQSGFR', 'VVLQAGFR',
              'VVLQNGFR', 'VVLQIGFR', 'VVLQSPFR'] # Double: VVLQSPFR
inSubsCovid = ['AVLQSGFR', 'VTFQSAVK', 'ATVQSKMS', 'ATLQAIAS', 'VKLQNNEL', 'VRLQAGNA',
               'PMLQSADA', 'TVLQAVGA', 'ATLQAENV', 'TRLQSLEN', 'PKLQSSQA']
inSubCovidColor = 'black'
inSubsSARS = ['VTFQGKFK', 'PLMQSADA', 'PKLQASQA']
inSubSARSColor = 'grey'
inDatapointColor = ['#CC5500', '#CC5500', '#CC5500', '#CC5500', '#CC5500', '#CC5500',
                    '#CC5500', '#CC5500', '#CC5500', '#CC5500', '#CC5500', '#CC5500',
                    '#CC5500', '#CC5500', '#CC5500',
                    'black', 'black', 'black', 'black', 'black', 'black',
                    'black', 'black', 'black', 'black', 'black',
                    '#F79620', '#F79620', '#F79620']


inKnownSubsStartIndex = 0
inKnownTarget = ['nsp4/5', 'nsp5/6', 'nsp6/7', 'nsp7/8', 'nsp8/9', 'nsp9/10',
                 'nsp10/12', 'nsp12/13', 'nsp13/14', 'nsp14/15', 'nsp15/16']
# inDatapointColor = '#CC5500'
inBarWidth = 0.75
inBarColor = '#CC5500'

# Input 9: Evaluate Substrate With Fixed Frame
inPlotEnrichedSubstrateFrame = True
inPrintLoadedFrames = True
inPlotBinnedSubs = 30



# ======================================== Set Parameters ========================================
if inShowEnrichmentAsSquares:
    # Set figure dimension when plotting EM plots as squares
    figSizeEM = inFigureAsSquares
    figBordersEM = inFigureBordersAsSquares
else:
    # Set figure dimension when plotting EM plots as rectangles
    figSizeEM = inFigureSize
    figBordersEM = inFigureBorders

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


def plotKnownSubPrediction():
    print('================================= Rank Known Substrates '
          '=================================')

    substrates = {
        "ATLQAENV": 6.915391686,
        "ATLQAIAS": 5.974816893,
        "ATVQSKMS": 0.8655595173,
        "AVLQSGFR": 5.55764466,
        "IVLQSGFR": 5.711949158,
        "KVLQSGFR": 4.342788918,
        "PKLQASQA": 5.259179405,
        "PKLQSSQA": 5.167261885,
        "PLMQSADA": 4.541013113,
        "PMLQSADA": 6.745837998,
        "TRLQSLEN": 5.03887008,
        "TVLQAVGA": 5.644710202,
        "VGLQSGFR": 3.36114118,
        "VKLQNNEL": 2.965594448,
        "VRLQAGNA": 6.732328298,
        "VTFQGKFK": 4.626522297,
        "VTFQSAVK": 2.7657616,
        "VVGQSGFR": -0.4574221781,
        "VVLHSGFR": 2.41369248,
        "VVLMSGFR": 1.682469002,
        "VVLQAGFR": 6.478588254,
        "VVLQIGFR": 3.277549563,
        "VVLQNGFR": 3.759282483,
        "VVLQSGFR": 6.168589689,
        "VVLQSPFR": 3.933666618,
        "VVLYSGFR": -1.546786043,
        "VVMQSGFR": 5.012006811,
        "VVVQSGFR": 0.01899718477,
        "VYLQSGFR": 5.619216201
    }

    yMax = 8
    yMin = -2


    # Prep data for the figure
    xValues = []
    yValues = []
    for substrate, score in substrates.items():
        xValues.append(substrate)
        yValues.append(score)
        print(f'Substrate:{red} {substrate}{resetColor}\n'
              f'     Score:{yellow} {np.round(score, 2)}{resetColor}\n')

    substrateColors = {}
    for index, substrate in enumerate(substrates.keys()):
        substrateColors[substrate] = inDatapointColor[index]


    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.bar(xValues, yValues, color=inDatapointColor, width=inBarWidth)
    plt.ylabel('Predicted Score', fontsize=inFigureLabelSize)
    plt.title(f'{inEnzymeName}\n'
          f'Predicting Substrate Affinity - SArKS',
              fontsize=inFigureTitleSize, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=inFigureTickSize)
    plt.xticks(rotation=90, ha='center')
    plt.axhline(y=0, color='black', linewidth=inLineThickness)
    plt.ylim(yMin, yMax)

    # Set the thickness of the figure border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(inLineThickness)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=inTickLength,
                   labelsize=inFigureTickSize, width=inLineThickness)

    fig.tight_layout()
    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()

plotKnownSubPrediction()
