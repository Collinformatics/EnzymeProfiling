import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import sys



# Mising Data: 'Kinetics-100nM SNILQAGFR'

# Input 1: Select Dataset
inEnzymeName = 'Mpro2' # 'SAVLQSGFR-Raw data and analyzed data' #
inFileNames = [
    'Kinetics-50nM SVPLQAGFR', 'Kinetics-100nM AVLQSGFR', 'Kinetics-100nM SAVLQSGFR',
    'Kinetics-100nM SNILQAGFR', 'Kinetics-100nM SPILQAGFR', 'Kinetics-100nM SPIMQAGFR',
    'Kinetics-100nM SVIHQAGFR', 'Kinetics-100nM SVILQAGFR', 'Kinetics-100nM SVILQTGFR',
    'Kinetics-100nM SVIMQAGFR', 'Kinetics-100nM SVPLQAGFR'
]
inFileName = inFileNames[3]
inPathFolder = f'Enzymes/{inEnzymeName}/Kinetics/Data'
# inPathFolder = f'Enzymes/{inEnzymeName}/Previous/Kinetics1'
inSheetName = ['Product standard', 'Sub_Raw data']
inDatasetLabel = inFileName.split('-')[1]
inEnzymeConc = inDatasetLabel.split(' ')[0]
inSubstrate = inDatasetLabel.split(' ')[1]

# Input 2: Process Data
inBurstKinetics = False
inBurstProduct = 0.12
inConcentrationUnit = 'μM'
inPlotAllFigs = True
inStdCurveStartIndex = 0
inConcCutoff = 10 # Max percentage of substrate concentration in rxn plot
inMaxCovariance = 0.3
inRoundDec = 3

# Saving the data
inSaveFigures = True
inFigureResolution = 300 # dpi

# If product release is rate limiting what
# Conc dependence of kinetics, [sub] is not relevant so it might make Km drop


# =================================== Setup Parameters ===================================
pd.set_option('display.float_format', '{:,.2f}'.format)

# Colors:
white = '\033[38;2;255;255;255m'
greyDark = '\033[38;2;144;144;144m'
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

# Verify directory paths
pathFigures = os.path.join(inPathFolder, f'Figures')
if not os.path.exists(pathFigures):
    os.makedirs(pathFigures, exist_ok=True)



# ================================== Define Functions ====================================
def pressKey(event):
    if event.key == 'escape':
        plt.close()
    elif event.key == 'e':
        sys.exit()
    elif event.key == 'r':
        python = sys.executable # Doesnt seem to work on windows?
        os.execl(python, python, *sys.argv)



def inspectData(data):
    hasNaN = data.isna().values.any()
    if hasNaN:
        print(f'{orange}ERROR: The dataset contains NaN values')
        sys.exit()



def loadExcel(sheetName='', loadStandard=False):
    print('=================================== Load Data '
          '===================================')
    print(f'Loading excel sheet: {pink}{sheetName}{resetColor}')
    path = os.path.join(inPathFolder, inFileName)
    if '.xlsx' not in path:
        path += '.xlsx'
    print(f'Loading file at path:\n     {greenDark}{path}{resetColor}\n\n')

    # Load data
    if loadStandard:
        data = pd.read_excel(path, sheet_name=sheetName, header=None)

        # Remove NaN
        dataFiltered = data.dropna(how='all')
        dataFiltered = dataFiltered.dropna(axis=1, how='all')

        # Set Headers
        dataFiltered.columns = list(dataFiltered.iloc[0])  # Set the first row as header
        dataFiltered = dataFiltered[1:]
        dataFiltered = dataFiltered.reset_index(drop=True)

        print(f'Raw Data: {pink}{sheetName}{resetColor}\n{data}\n\n\n'
              f'Loaded Data: {pink}{sheetName}{resetColor}\n{dataFiltered}\n\n')
        inspectData(data=dataFiltered)
    else:
        data = pd.read_excel(path, sheet_name=sheetName, header=None, index_col=[0])
        print(f'Raw Data: {pink}{sheetName}{resetColor}\n{data}\n\n')
        indexHeader = data.index[1]

        # Identify the indices for each substrate concentration
        columnHeaders = data.iloc[0].tolist()
        indexConc = {}
        for index, conc in enumerate(columnHeaders):
            if not pd.isna(conc):
                if conc in indexConc.keys():
                    indexConc[conc].append(index)
                else:
                    indexConc[conc] = [index]


        # Group data by substrate concentration
        dataFiltered = {}
        for conc, value in indexConc.items():
            # Get only the data rows, remove the first 2 rows (NaN and Time)
            filteredIndex = data.index[2:]
            triplicate = data.loc[filteredIndex, [data.columns[i] for i in value]].copy()

            # Redefine column headers
            newHeaders = data.iloc[1, value].tolist() # row 1 = 'G1', 'H1', 'I1'
            triplicate.columns = newHeaders

            # Inspect time dtype
            if isinstance(triplicate.index[0], datetime.time):
                triplicate.index = [
                    (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6) / 60
                    for t in triplicate.index
                ]
            triplicate.index.name = indexHeader
            dataFiltered[conc] = triplicate

        print(f'Loaded Data: {pink}{sheetName}{resetColor}')
        for conc, value in dataFiltered.items():
            print(f'Concentration: {purple}{conc} {inConcentrationUnit}{resetColor}\n'
                  f'{value}\n\n')
            inspectData(data=value)

    return dataFiltered



def processStandardCurve(psc, plotFig=False):
    print('======================= Processing Product Standard Curve '
          '=======================')
    psc.loc[:, 'Ratio'] = psc.iloc[:, 1] / psc.iloc[:, 0]
    print(f'Product Standard Curve:\n{psc}\n\n'
          f'Collecting data starting at index: {red}{inStdCurveStartIndex}{resetColor}\n'
          f'{psc.iloc[inStdCurveStartIndex:, :]}\n')
    x = pd.to_numeric(psc.iloc[inStdCurveStartIndex+1:, 0], errors='raise')
    y = pd.to_numeric(psc.iloc[inStdCurveStartIndex+1:, 1], errors='raise')
    N = len(x)  # Number of datapoints in the curve

    # Evaluate: X axis
    maxValue = math.ceil(max(x))
    magnitude = math.floor(math.log10(maxValue))
    unit = 10 ** (magnitude - 1)
    xMax = math.ceil(maxValue / (10 * unit)) * (10 * unit) # Jump to next clean boundary
    if xMax <= maxValue:
        xMax += 5 * unit
    # print(f'X Max: {xMax}')
    xMin = 0

    # Evaluate: Y axis
    maxValue = math.ceil(max(y))
    magnitude = math.floor(math.log10(maxValue))
    unit = 10 ** (magnitude - 1)
    yMax = math.ceil(maxValue / (10 * unit)) * (10 * unit) # Jump to next clean boundary
    if yMax <= maxValue:
        yMax += 5*unit
    # print(f'Y Max: {yMax}\n')
    yMin = 0

    # Fit the datapoints
    slope, intercept = np.polyfit(x, y, 1) # Degree 1 polynomial = linear
    fitLine = slope * x + intercept
    fit = f'y = {slope:.3f}x + {intercept:.3f}'
    print(f'Fit: {red}{fit}{resetColor}')

    # Calculate R² values
    rSquared = r2_score(y, fitLine)
    print(f'R² Value: {red}{round(rSquared, inRoundDec)}{resetColor}\n\n')

    # plotFig = False
    # print(f'{orange}WARNING: We are skipping the Std Curve Plot')
    if plotFig:
        # Scatter plot
        fig, ax = plt.subplots(figsize=(9.5, 8))
        plt.scatter(x, y, color='#BF5700', marker='o',
                    edgecolors='black', linewidths=0.8)
        plt.title(f'\nProduct Standard Curve\n{inEnzymeConc} {inEnzymeName}\n'
                  f'{inSubstrate}', fontsize=18, fontweight='bold')
        plt.xlabel(f'{psc.columns[0]} ({inConcentrationUnit})', fontsize=16)
        plt.ylabel(f'{psc.columns[1]}', fontsize=16)
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
        plt.grid(True)
        plt.plot(x, fitLine, color='black',
                 label=f'{fit}\nR² = {round(rSquared, inRoundDec)}')
        plt.tight_layout()
        plt.legend(fontsize=12, prop={'weight': 'bold'})
        fig.canvas.mpl_connect('key_press_event', pressKey)
        plt.show()

        # Save figure
        if inSaveFigures:
            saveTag = f'PSC - N {N} - {inEnzymeName} {inDatasetLabel}'
            saveLocation = os.path.join(pathFigures, f'{saveTag}.png')
            if os.path.exists(saveLocation):
                print(f'{yellow}The figure was not saved\n\n'
                      f'File was already found at path:\n'
                      f'     {saveLocation}{resetColor}\n\n')
            else:
                print(f'Saving figure at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                fig.savefig(saveLocation, dpi=inFigureResolution)
    plotFig = True
    return slope, N


def formatValues(values):
    newValues = []
    for value in values:
        if f'{value}'.endswith('.0'):
            newValues.append(int(value))
        else:
            newValues.append(value)
    return newValues



def processKinetics(slope, datasets, N, plotFig=False):
    print('=========================== Processing Kinetics Data '
          '============================')
    print(f'Slope: {red}{round(slope, inRoundDec)}{resetColor}\n')

    # Process florescence data
    print('Statistical analysis:')
    for conc, data in datasets.items():
        # Statistical analysis
        data.loc[:, 'Avg'] = data.loc[:, :].mean(axis=1)
        data.loc[:, 'StDev'] = data.std(axis=1)
        data.loc[:, 'CoVar'] = data.loc[:, 'StDev'] / data.loc[:, 'Avg']
        data.loc[:, '[Prod]'] = data.loc[:, 'Avg'] / slope
        print(f'Substrate concentration: {purple}{conc} ({inConcentrationUnit})'
              f'{resetColor}\n{data}\n\n')


    # Plot individual reactions
    reactionSlopes = {}
    for conc, values in datasets.items():
        if isinstance(conc, str):
            if 'uM' in conc:
                conc = conc.replace('uM', '')
                conc = float(conc)
            elif  inConcentrationUnit in conc:
                conc = conc.replace(inConcentrationUnit, '')
                conc = float(conc)
        cutoff = conc / inConcCutoff
        print(f'Concentration: {purple}{conc} {inConcentrationUnit}{resetColor}\n'
              f'     Cutoff: {red}{round(cutoff, inRoundDec)} {inConcentrationUnit}'
              f'{resetColor}')

        # Filter datapoints
        x = pd.to_numeric(values.index, errors='coerce')
        y = pd.to_numeric(values.loc[:, '[Prod]'], errors='coerce')
        mask = y <= cutoff
        x = x[mask]
        y = y[mask]


        if len(x) != 0:
            # Evaluate: X axis
            maxValue = math.ceil(max(x))
            magnitude = math.floor(math.log10(maxValue))
            unit = 10 ** (magnitude - 1)
            stepX = unit * 5
            xMax = math.ceil(maxValue / (10 * unit)) * (10 * unit)
            if magnitude == 0 and xMax - stepX > max(x):
                xMax -= stepX
            print(f'Max Value: {maxValue}\n'
                  f'X Max: {xMax}\n'
                  f'  Mag: {magnitude}\n'
                  f' Unit: {unit}\n'
                  f' Step: {stepX}\n')
            xMin = 0

            # Evaluate: Y axis
            maxValue = math.ceil(max(y))
            magnitude = math.floor(math.log10(maxValue))
            unit = 10 ** (magnitude - 1)
            yMin = 0
            if magnitude == 0:
                if maxValue > 1:
                    # print(f'Conc (1): {conc}')
                    stepY = unit * 5
                    yMax = math.ceil(maxValue / (10 * unit)) * (10 * unit)
                elif max(y) < 0.5:
                    # print(f'Conc (2): {conc}')
                    stepY = unit * 1
                    yMax = math.ceil(max(y * 10))/10
                else:
                    # print(f'Conc (3): {conc}')
                    stepY = unit
                    yMax = maxValue + stepY
                if yMax <= maxValue:
                    yMax += stepY
            else:
                print(f'Conc (4): {conc}')
                yMax = maxValue
                stepY = yMax / 5
            print(f'Y: {list(y)}\n'
                  f'Max V: {maxValue}\n'
                  f'Y Max: {yMax}\n'
                  f'  Mag: {magnitude}\n'
                  f' Unit: {unit}\n'
                  f' Step: {stepY}\n')


            # Fit the datapoints
            slope, intercept = np.polyfit(x, y, 1) # Degree 1 polynomial = linear
            fitLine = slope * x + intercept
            fit = f'y = {slope:.3f}x + {intercept:.3f}'
            print(f'Fit: {red}{fit}{resetColor}\n')
            reactionSlopes[conc] = slope

            # Predictions
            predictedY = slope * np.array(x) + intercept

            # Calculate R² values
            rSquared = r2_score(y, predictedY)
            print(f'R² Value: {red}{round(rSquared, inRoundDec)}{resetColor}\n\n')
            fit += f'\nR² = {round(rSquared, inRoundDec)}'


            if plotFig:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(9.5, 8))
                plt.scatter(x, y, color='#BF5700', marker='o',
                            edgecolors='black', linewidths=0.8)
                plt.title(f'\n{inEnzymeConc} {inEnzymeName}\n{inSubstrate}\n'
                          f'Substrate Concentration: {conc} {inConcentrationUnit}\n'
                          f'Maximum Product Concentration: {round(cutoff, 2)}',
                          fontsize=18, fontweight='bold')
                plt.xlabel(f'Time', fontsize=16)
                plt.ylabel(f'Product Concentration', fontsize=16)

                # Format axes
                xTicks = np.arange(xMin, xMax + stepX, stepX)
                ax.set_xticks(xTicks)
                ax.set_xticklabels(formatValues(xTicks))
                plt.xlim(xMin, xMax)
                # ax.set_yticks(yTicks)
                # ax.set_yticklabels(formatValues(yTicks))
                plt.ylim(yMin, yMax)



                plt.grid(True)
                plt.plot(x, fitLine, label=fit, color='#F8971F')
                plt.tight_layout()
                plt.legend(fontsize=12, prop={'weight': 'bold'})
                fig.canvas.mpl_connect('key_press_event', pressKey)
                plt.show()

                # Save figure
                if inSaveFigures:
                    saveTag = (f'Kinetics - {inEnzymeName} {inDatasetLabel} '
                               f'{conc}{inConcentrationUnit} - PSC N {N}')
                    saveTag = saveTag.replace('.','_')
                    saveLocation = os.path.join(pathFigures, f'{saveTag}.png')
                    if os.path.exists(saveLocation):
                        print(f'{yellow}The figure was not saved\n\n'
                              f'File was already found at path:\n'
                              f'     {saveLocation}{resetColor}\n\n')
                    else:
                        print(f'Saving figure at path:\n'
                              f'     {greenDark}{saveLocation}{resetColor}\n\n')
                        fig.savefig(saveLocation, dpi=inFigureResolution)

    return reactionSlopes



def processKineticsBurst(slope, datasets, plotFig, N):
    print('======================== Processing Burst Kinetics Data '
          '=========================')
    print(f'Slope: {red}{round(slope, inRoundDec)}{resetColor}\n')
    slopeRelease = []

    # Process florescence data
    print('Statistical analysis:')
    for conc, data in datasets.items():
        # Statistical analysis
        data.loc[:, 'Avg'] = data.loc[:, :].mean(axis=1)
        data.loc[:, 'StDev'] = data.std(axis=1)
        data.loc[:, 'CoVar'] = data.loc[:, 'StDev'] / data.loc[:, 'Avg']
        data.loc[:, '[Prod]'] = data.loc[:, 'Avg'] / slope
        print(f'Substrate concentration: {purple}{conc}{resetColor}\n{data}\n\n')



    # Plot individual reactions
    reactionSlopes = {}
    for conc, values in datasets.items():
        if conc >= 5:
            continue
        print('======================================='
              '======================================')
        if isinstance(conc, str):
            if 'uM' in conc:
                conc = conc.replace('uM', '')
                conc = float(conc)
            elif  inConcentrationUnit in conc:
                conc = conc.replace(inConcentrationUnit, '')
                conc = float(conc)
        cutoff = conc / inConcCutoff
        print(f'Dataset: {purple}{conc} {inConcentrationUnit}{resetColor}\n'
              f'Cutoff: {red}{round(cutoff, inRoundDec)} {inConcentrationUnit}'
              f'{resetColor}'
              f'\nConc Cutoff: {red}{inConcCutoff}{resetColor}')

        # Filter datapoints
        x = pd.to_numeric(values.index, errors='coerce')
        y = pd.to_numeric(values.loc[:, '[Prod]'], errors='coerce')
        mask = y <= cutoff
        x = x[mask]
        y = y[mask]

        # Get datapoints
        x = np.array(x)
        y = np.array(y)

        if len(x) == 0:
            print()
            continue

        # Order the datasets
        if x[0] > x[1]:
            print(f'{yellow}The first x value is > the second\n'
                  f'     X: {x[0]} > {x[1]}\n\n'
                  f'The order of the datapoints has been reversed to prevent an error '
                  f'while splitting the data.{resetColor}\n\n')
            x = reversed(x)
            y = reversed(y)


        # Determine the split index
        burstEndIndex = 0
        yLimit = min(y) + inBurstProduct
        for index, value in enumerate(y):
            if value <= yLimit:
                burstEndIndex = index + 1
        if burstEndIndex == 1:
            # Adjust index
            print(f'     {yellow}Adjusting Burst Index: '
                  f'{cyan}{burstEndIndex} -> {burstEndIndex + 1}{resetColor}')
            burstEndIndex += 1
        if burstEndIndex < len(x):
            print(f'     Burst Domain: {red}{inBurstProduct}{resetColor}\n'
                  f'     Split Index: {red}{burstEndIndex}{resetColor}\n'
                  f'     Burst Range: {pink}{round(y[burstEndIndex], inRoundDec)}'
                  f'{resetColor} - {pink}{round(y[0], inRoundDec)} {inConcentrationUnit}'
                  f'{resetColor}\n')
        else:
            # Dont split the dataset
            burstEndIndex = 0

        # Set axis
        xMin, xMax, xTicks, yMin, yMax, yTicks = setAxes(x=x, y=y, conc=conc)

        # Evaluate splitting
        if burstEndIndex == 0: # Dont split the dataset
            # Fit the datapoints
            slope, intercept = np.polyfit(x, y, 1)  # Degree 1 polynomial = linear
            fitLine = slope * x + intercept
            fit = f'y = {slope:.3f}x + {intercept:.3f}'
            print(f'     Fit: {red}{fit}{resetColor}')
            reactionSlopes[conc] = slope

            # Predictions
            predictedY = slope * np.array(x) + intercept

            # Calculate R² values
            rSquared = r2_score(y, predictedY)
            print(f'      R²: {red}{round(rSquared, inRoundDec)}{resetColor}\n\n')
            fit += f'\nR² = {round(rSquared, inRoundDec)}'


            if plotFig:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(9.5, 8))
                plt.scatter(x, y, color='#BF5700', marker='o',
                            edgecolors='black', linewidths=0.8)
                plt.plot(x, fitLine, label=fit, color='#F8971F')
                plt.title(f'\n{inEnzymeConc} {inEnzymeName}\n{inSubstrate}\n'
                          f'Substrate Concentration: {conc} {inConcentrationUnit}\n'
                          f'Maximum Product Concentration: '
                          f'{round(cutoff, 2)} {inConcentrationUnit}',
                          fontsize=18, fontweight='bold')
                plt.xlabel(f'Time (min)', fontsize=16)
                plt.ylabel(f'Product Concentration ({inConcentrationUnit})',
                           fontsize=16)

                # Format axes
                ax.set_xticks(xTicks)
                ax.set_xticklabels(formatValues(xTicks))
                plt.xlim(xMin, xMax)
                ax.set_yticks(yTicks)
                ax.set_yticklabels(formatValues(yTicks))
                plt.ylim(yMin, yMax)

                plt.grid(True)
                plt.tight_layout()
                plt.legend(fontsize=12, prop={'weight': 'bold'})
                fig.canvas.mpl_connect('key_press_event', pressKey)
                plt.show()

                # Save figure
                if inSaveFigures:
                    saveTag = (f'Kinetics Burst - {inEnzymeName} {inDatasetLabel} '
                               f'{conc}{inConcentrationUnit} - PSC N {N}')
                    saveTag = saveTag.replace('.', '_')
                    saveLocation = os.path.join(pathFigures, f'{saveTag}.png')
                    if os.path.exists(saveLocation):
                        print(f'{yellow}The figure was not saved\n\n'
                              f'File was already found at path:\n'
                              f'     {saveLocation}{resetColor}\n\n')
                    else:
                        print(f'Saving figure at path:\n'
                              f'     {greenDark}{saveLocation}{resetColor}\n\n')
                        fig.savefig(saveLocation, dpi=inFigureResolution)
        else:
            # Phase 1: Burst
            timeBurst = x[:burstEndIndex]
            productBurst = y[:burstEndIndex]

            # Phase 2: Steady-State
            timeSteady = x[burstEndIndex:]
            productSteady = y[burstEndIndex:]

            # Fit the datapoints
            slope1, intercept1 = np.polyfit(
                timeBurst, productBurst, 1) # Fit burst phase
            slope2, intercept2 = np.polyfit(
                timeSteady, productSteady, 1) # Fit steady-state phase
            fit1 = f'y = {slope1:.3f}x + {intercept1:.3f}'
            fit2 = f'y = {slope2:.3f}x + {intercept2:.3f}'

            reactionSlopes[conc] = slope1
            slopeRelease.append(slope2)

            # Predictions
            predictedBurst = slope1 * np.array(timeBurst) + intercept1
            predictedSteady = slope2 * np.array(timeSteady) + intercept2

            # Calculate R² values
            rSquared1 = r2_score(productBurst, predictedBurst)
            rSquared2 = r2_score(productSteady, predictedSteady)

            fit1 += f'\nR² = {round(rSquared1, inRoundDec)}'
            fit2 += f'\nR² = {round(rSquared2, inRoundDec)}'
            print(f'Burst Domain:\n'
                  f'     Length: {red}{len(timeBurst)}{resetColor}\n'
                  f'         R²: {red}{round(rSquared1, inRoundDec)}{resetColor}\n\n'
                  f'Steady Domain:\n'
                  f'     Length: {red}{len(timeSteady)}{resetColor}\n'
                  f'         R²: {red}{round(rSquared2, inRoundDec)}{resetColor}\n\n')

            if plotFig:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(9.5, 8))
                plt.scatter(x, y, color='#BF5700', marker='o',
                            edgecolors='black', linewidths=0.8)
                plt.plot(timeBurst, slope1 * timeBurst + intercept1,
                         label=fit1, color='#F8971F') # Burst fit
                plt.plot(timeSteady, slope2 * timeSteady + intercept2,
                         label=fit2, color='black') # Steady-state fit
                plt.title(f'\n{inEnzymeConc} {inEnzymeName}\n{inSubstrate}\n'
                          f'Substrate Concentration: {conc} {inConcentrationUnit}\n'
                          f'Maximum Product Concentration: '
                          f'{round(cutoff, 2)} {inConcentrationUnit}',
                          fontsize=18, fontweight='bold')
                plt.xlabel(f'Time (min)', fontsize=16)
                plt.ylabel(f'Product Concentration ({inConcentrationUnit})',
                           fontsize=16)

                # Format axes
                ax.set_xticks(xTicks)
                ax.set_xticklabels(formatValues(xTicks))
                plt.xlim(xMin, xMax)
                # ax.set_yticks(yTicks)
                # ax.set_yticklabels(formatValues(yTicks))
                plt.ylim(yMin, yMax)

                plt.grid(True, color='black')
                plt.tight_layout()
                plt.legend(fontsize=12, prop={'weight': 'bold'})
                fig.canvas.mpl_connect('key_press_event', pressKey)
                plt.show()

                # Save figure
                if inSaveFigures:
                    saveTag = (f'Kinetics Burst - {inEnzymeName} {inDatasetLabel} '
                               f'{conc}{inConcentrationUnit} - PSC N {N}')
                    saveTag = saveTag.replace('.', '_')
                    saveLocation = os.path.join(pathFigures, f'{saveTag}.png')
                    if os.path.exists(saveLocation):
                        print(f'{yellow}The figure was not saved\n\n'
                              f'File was already found at path:\n'
                              f'     {saveLocation}{resetColor}\n\n')
                    else:
                        print(f'Saving figure at path:\n'
                              f'     {greenDark}{saveLocation}{resetColor}\n\n')
                        fig.savefig(saveLocation, dpi=inFigureResolution)

    # Evaluate: Release rate
    releaseAvg, releaseStDev = np.mean(slopeRelease), np.std(slopeRelease)
    print(f'Release Rate:\n'
          f'     Average: {red}{releaseAvg:.3e} {inConcentrationUnit}/min'
          f'{resetColor}\n'
          f'      St Dev: {red}{releaseStDev:.3e} {inConcentrationUnit}/min'
          f'{resetColor}\n\n')

    return reactionSlopes



def setAxes(x, y, conc):
    # Evaluate: X axis
    maxValue = math.ceil(max(x))
    magnitude = math.floor(math.log10(maxValue))
    unit = 10 ** (magnitude - 1)
    stepX = unit * 5
    xMax = math.ceil(maxValue / (10 * unit)) * (10 * unit)
    if magnitude == 0 and xMax - stepX > max(x):
        xMax -= stepX
    print(f'Max Value: {maxValue}\n'
          f'X Max: {xMax}\n'
          f'  Mag: {magnitude}\n'
          f' Unit: {unit}\n'
          f' Step: {stepX}\n')
    xMin = 0
    xTicks = np.arange(xMin, xMax + stepX, stepX)


    # Evaluate: Y axis
    maxValue = math.ceil(max(y))
    magnitude = math.floor(math.log10(maxValue))
    unit = 10 ** (magnitude - 1)
    yMin = 0
    print(f'{purple}')
    if magnitude == 0:
        if maxValue > 1:
            print(f'Conc (1): {conc}')
            stepY = unit * 5
            yMax = math.ceil(maxValue / (10 * unit)) * (10 * unit)
        elif max(y) < 0.5:
            print(f'Conc (2): {conc}')
            stepY = unit * 1
            yMax = math.ceil(max(y * 10)) / 10
        else:
            print(f'Conc (3): {conc}')
            stepY = unit
            yMax = maxValue + stepY
        if yMax <= maxValue:
            yMax += stepY
    else:
        print(f'Conc (4): {conc}')
        yMax = maxValue
        stepY = yMax / 5
    print(f'{resetColor}')


    # Inspect yMax
    if isinstance(yMax, float):
        if not f'{yMax}'.endswith('.0') and not f'{yMax}'.endswith('.5'):
            num, dec = int(str(yMax).split('.')[0]), int(str(yMax).split('.')[1])
            # print(f'Y Max: {yMax}\n'
            #       f'  Num: {num}\n'
            #       f'  Dec: {dec}')
            # print(f'  dec % 2 == 0 {red}{dec % 2 == 0}{resetColor}\n'
            #       f'  dec % 2 == 5 {red}{dec % 2 == 5}{resetColor}')
            # Make the decimal be divisible by 2 or 5
            while not dec % 2 == 0 and not dec % 5 == 0:
                dec += 1
                # print(f'Dec: {dec}')
                # print(f'  dec % 2 == 0 {red}{dec % 2 == 0}{resetColor}\n'
                #       f'  dec % 2 == 5 {red}{dec % 5 == 0}{resetColor}')
                yMax = num + (dec / 10)
                # print(f'New yMax: {yMax}')


    def fitTicks(maximum, minimum):
        print('Setting Axis Ticks:')
        print(f'     Max: {maximum}\n'
              f'     Min: {minimum}\n')
        ticks = None
        steps = [10, 6, 5, 4, 3, 2.5, 2, 1.5, 1,
                 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.125, 0.075, 0.05]

        for step in steps:
            if step > maximum / 2:
                continue
            if maximum % step == 0:
                ticks = np.arange(minimum, maximum + step, step)
                print(f'L({step}): {len(ticks)}')
                if len(ticks) <= 2:
                    print(f'Skip: {ticks}')
                    continue
                print(f'Ticks ({step}): {ticks}\n\n')
                break
            print()

        return ticks

    # Adjust yMax
    if maxValue <= 1.5:
        maxConc = y[-1]
        print(f'Y Values: {red}{y[-1]}{resetColor}{resetColor}\n'
              f'   Y Max: {red}{yMax}{resetColor}\n'
              f'   Max V: {red}{maxValue}{resetColor}\n'
              f'Max Conc: {red}{maxConc}{resetColor}\n')
        print(f'1: {red}{yMax - 0.2}{resetColor}\n'
              f'2: {red}{maxConc}{resetColor}\n')

        while yMax - 0.15 > maxConc:
            yMax -= 0.1
            print(f'{cyan}Reduce!{resetColor}\nY New: {yMax}\n')

    yTicks = fitTicks(yMax, yMin)
    while yTicks is None:
        yMax += 0.1
        yMax = round(yMax*10, 1) / 10
        print(f'{orange}New Y: {yMax}{resetColor}')
        yTicks = fitTicks(yMax, yMin)


    print(f'max(Y): {round(max(y), inRoundDec)}\n'
          f'Max V: {maxValue}\n'
          f'Y Max: {yMax}\n'
          f'  Mag: {magnitude}\n'
          f' Unit: {unit}\n'
          f' Step: {stepY}\n')

    return xMin, xMax, xTicks, yMin, yMax, yTicks



def MichaelisMenten(velocity, N):
    print('=============================== Michaelis-Menten '
          '================================')
    dec = 2
    substrateConc = list(velocity.keys())


    print('Velocities:')
    v = []
    for key, value in velocity.items():
        v.append(value)
        print(f'Substrate: {purple}{key}{resetColor}\n'
              f'     v: {red}{round(value, 4)}{resetColor}\n')
    print()

    # Michaelis-Menten function
    def MM(substrate, Vmax, Km):
        return (Vmax * substrate) / (Km + substrate)


    # Fit data
    popt, pcov = curve_fit(MM, substrateConc, v, bounds=(0, np.inf))
    vMax, Km = popt

    # Fit the data
    xFit = np.linspace(0, max(substrateConc) * 1.2, 100)
    yFit = MM(xFit, vMax, Km)

    # Calculate R² values
    predictedV = MM(np.array(substrateConc), vMax, Km)
    rSquared = r2_score(v, predictedV)
    fit = f'R² = {rSquared:.3f}'

    print(f'Dataset: {purple}{inEnzymeConc} {inEnzymeName} {inSubstrate}{resetColor}\n'
          f'     Km = {red}{Km:.3f}{resetColor}\n'
          f'     Vmax: {red}{vMax:.3f}{resetColor}\n'
          f'     R²: {red}{rSquared:.3f}{resetColor}\n')

    # Plot the data
    fig, ax = plt.subplots(figsize=(9.5, 8))
    plt.scatter(substrateConc, v, color='#BF5700', edgecolors='black', linewidths=0.8)
    plt.plot(xFit, yFit, color='#F8971F', label=fit)
    plt.title(f'Michaelis-Menten\n{inEnzymeConc} {inEnzymeName}\n{inSubstrate}\n'
              f'Km = {round(Km, dec)} {inConcentrationUnit}\n'
              f'Vmax = {round(vMax, dec)} {inConcentrationUnit}/min',
              fontsize=18, fontweight='bold')
    plt.xlabel(f'[Substrate] ({inConcentrationUnit})', fontsize=16)
    plt.ylabel(f'Velocity ({inConcentrationUnit}/min)', fontsize=16)
    plt.tight_layout()
    plt.legend(fontsize=12, prop={'weight': 'bold'})
    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()

    # Save figure
    if inSaveFigures:
        saveTag = f'MM - {inEnzymeName} {inDatasetLabel} - {inSubstrate} - PSC N {N}'
        if inBurstKinetics:
            saveTag = saveTag.replace('MM', 'MM - Burst')
        saveLocation = os.path.join(pathFigures, f'{saveTag}.png')
        if os.path.exists(saveLocation):
            print(f'{yellow}The figure was not saved\n\n'
                  f'File was already found at path:\n'
                  f'     {saveLocation}{resetColor}\n\n')
        else:
            print(f'Saving figure at path:\n'
                  f'     {greenDark}{saveLocation}{resetColor}\n\n')
            fig.savefig(saveLocation, dpi=inFigureResolution)



# ====================================== Load data =======================================
# Load Data
prodStdCurve = loadExcel(sheetName=inSheetName[0], loadStandard=True)
fluorescence = loadExcel(sheetName=inSheetName[1])

sys.exit()

# Calculate: Product Standard Curve
m, N= processStandardCurve(psc=prodStdCurve, plotFig=inPlotAllFigs)

# Process kinetics
if inBurstKinetics:
    v = processKineticsBurst(slope=m, datasets=fluorescence, plotFig=inPlotAllFigs, N=N)
else:
    v = processKinetics(slope=m, datasets=fluorescence, plotFig=inPlotAllFigs, N=N)
MichaelisMenten(velocity=v, N=N)
