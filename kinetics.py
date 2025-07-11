import datetime
from time import time_ns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import linregress
import sys




# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inFileName = 'Kinetics template-60nM SVILQAPFR'
inPathFolder = f'{inEnzymeName}/Kinetics'
inSheetName = ['Product standard', 'Sub_Raw data']
inSubstrate = inFileName.split('-')[1]

# Input 2: Process Data
inBurstKinetics = True
inConcentrationUnit = 'uM'
inPlotAllFigs = True
inStdCurveStartIndex = 0
inConcCutoff = 10 # Max percentage of substrate concentration in rxn plot
inMaxCovariance = 0.3



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



# ================================== Define Functions ====================================
def pressKey(event):
    if event.key == 'escape':
        plt.close()
    elif event.key == 'e':
        sys.exit()
    elif event.key == 'r':
        python = sys.executable # Doesnt seem to work on windows?
        os.execl(python, python, *sys.argv)



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
        # sys.exit()

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


    # Fit the datapoints
    slope, intercept = np.polyfit(x, y, 1)  # Degree 1 polynomial = linear
    fitLine = slope * x + intercept
    fit = f'y = {slope:.3f}x + {intercept:.3f}'
    print(f'Fit: {red}{fit}{resetColor}\n\n')

    if plotFig:
        # Scatter plot
        fig, ax = plt.subplots(figsize=(9.5, 8))
        plt.scatter(x, y, color='#BF5700', marker='o')
        plt.title(f'\nProduct Standard Curve\n{inEnzymeName}\n{inSubstrate}',
                  fontsize=18, fontweight='bold')
        plt.xlabel(f'{psc.columns[0]}', fontsize=16)
        plt.ylabel(f'{psc.columns[1]}', fontsize=16)
        plt.grid(True)
        plt.plot(x, fitLine, color='#32D713', label=fit)
        plt.tight_layout()
        plt.legend(fontsize=12, prop={'weight': 'bold'})
        fig.canvas.mpl_connect('key_press_event', pressKey)
        plt.show()

    return slope



def processKinetics(slope, datasets, plotFig=False):
    print('=========================== Processing Kinetics Data '
          '============================')
    print(f'Slope: {red}{round(slope, 3)}{resetColor}\n')

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
        cutoff = conc / inConcCutoff
        print(f'Numeric Conc: {red}{conc} {inConcentrationUnit}{resetColor}\n'
              f'Cutoff: {cutoff}\n\n')

        # Filter datapoints
        x = pd.to_numeric(values.index, errors='coerce')
        y = pd.to_numeric(values.loc[:, '[Prod]'], errors='coerce')
        mask = y <= cutoff
        x = x[mask]
        y = y[mask]


        # Fit the datapoints
        if len(x) != 0:
            slope, intercept = np.polyfit(x, y, 1) # Degree 1 polynomial = linear
            fitLine = slope * x + intercept
            fit = f'y = {slope:.3f}x + {intercept:.3f}'
            print(f'Fit: {red}{fit}{resetColor}\n\n')
            reactionSlopes[conc] = slope


            if plotFig:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(9.5, 8))
                plt.scatter(x, y, color='#BF5700', marker='o')
                plt.title(f'\n{inEnzymeName}\n{inSubstrate}\n'
                          f'Substrate Concentration: {(conc)} {inConcentrationUnit}\n'
                          f'Maximum Product Concentration: {cutoff}',
                          fontsize=18, fontweight='bold')
                plt.xlabel(f'Time', fontsize=16)
                plt.ylabel(f'Product Concentration', fontsize=16)
                plt.grid(True)
                plt.plot(x, fitLine, label=fit, color='#F8971F')
                plt.tight_layout()
                plt.legend(fontsize=12, prop={'weight': 'bold'})
                fig.canvas.mpl_connect('key_press_event', pressKey)
                plt.show()

    return reactionSlopes



def MichaelisMenten(velocity):
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

    print(f'Dataset: {purple}{inEnzymeName} {inSubstrate}{resetColor}\n'
          f'Vmax: {red}{vMax:.3f}{resetColor}\n'
          f'Km = {red}{Km:.3f}{resetColor}\n\n')

    # Fit the data
    xFit = np.linspace(0, max(substrateConc) * 1.2, 100)
    yFit = MM(xFit, vMax, Km)


    # Plot the data
    fig, ax = plt.subplots(figsize=(9.5, 8))
    plt.scatter(substrateConc, v, color='#BF5700')
    plt.plot(xFit, yFit, color='#32D713', label='Michaelis-Menten fit')
    plt.title(f'Michaelis-Menten\n{inEnzymeName}\n{inSubstrate}\n'
              f'Km = {round(Km, dec)}\nVmax = {round(vMax, dec)}',
              fontsize=18, fontweight='bold')
    plt.xlabel(f'[Substrate]', fontsize=16)
    plt.ylabel(f'Velocity', fontsize=16)
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()


def processKineticsBurst(slope, datasets, plotFig):
    print('======================== Processing Burst Kinetics Data '
          '=========================')
    print(f'Slope: {red}{round(slope, 3)}{resetColor}\n')

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
        # Get datapoints
        x = np.array(pd.to_numeric(values.index, errors='coerce'))
        y = np.array(pd.to_numeric(values.loc[:, '[Prod]'], errors='coerce'))
        print(f'X, Y: {len(x)}, {len(y)}\n')

        # Fit the datapoints
        smoothedProduct = savgol_filter(y, window_length=15, polyorder=2)
        print(f'Len(y): {len(y)}, Len(smoothedProduct): {len(smoothedProduct)}\n')
        rate = np.gradient(smoothedProduct, x)
        threshold = 0.5 * np.max(rate)
        burstEndIndex = np.where(rate < threshold)[0][0]
        print(f'Split Index: {burstEndIndex}\n')

        # Evaluate splitting
        if burstEndIndex == 0:
            # cutoff = conc / inConcCutoff
            # print(f'Numeric Conc: {red}{conc} {inConcentrationUnit}{resetColor}\n'
            #       f'Cutoff: {cutoff}\n\n')
            #
            # # Filter datapoints
            # mask = y <= cutoff
            # x = x[mask]
            # y = y[mask]
            # print(f'X, Y: {len(x)}, {len(y)}\n')

            slope, intercept = np.polyfit(x, y, 1)  # Degree 1 polynomial = linear
            fitLine = slope * x + intercept
            fit = f'y = {slope:.3f}x + {intercept:.3f}'
            print(f'Fit: {red}{fit}{resetColor}\n\n')
            reactionSlopes[conc] = slope

            if plotFig:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(9.5, 8))
                plt.scatter(x, y, color='#BF5700', marker='o')
                plt.title(f'\n{inEnzymeName}\n{inSubstrate}\n'
                          f'Substrate Concentration: {conc} {inConcentrationUnit}',
                          fontsize=18, fontweight='bold')
                plt.xlabel(f'Time', fontsize=16)
                plt.ylabel(f'Product Concentration', fontsize=16)
                plt.grid(True)
                plt.plot(x, fitLine, label=fit, color='black')
                plt.tight_layout()
                plt.legend(fontsize=12, prop={'weight': 'bold'})
                fig.canvas.mpl_connect('key_press_event', pressKey)
                plt.show()
        else:
            # Phase 1: Burst
            timeBurst = x[:burstEndIndex]
            productBurst = y[:burstEndIndex]

            # Phase 2: Steady-State
            timeSteady = x[burstEndIndex:]
            productSteady = y[burstEndIndex:]
            print(f'Time 1: {len(timeBurst)}, {len(productBurst)}\n'
                  f'Time 2: {len(timeSteady)}, {len(productSteady)}\n\n')

            # Fit burst phase
            slope1, intercept1, *_ = linregress(timeBurst, productBurst)

            # Fit steady-state phase
            slope2, intercept2, *_ = linregress(timeSteady, productSteady)

            reactionSlopes[conc] = slope1


            if plotFig:
                # Scatter plot
                fig, ax = plt.subplots(figsize=(9.5, 8))
                plt.scatter(x, y, color='#BF5700', marker='o')
                plt.plot(timeBurst, slope1 * timeBurst + intercept1,
                         label='Burst Fit', color='blue') # Burst fit
                plt.plot(timeSteady, slope2 * timeSteady + intercept2,
                         label='Steady-State Fit', color='red') # Steady-state fit
                plt.title(f'\n{inEnzymeName}\n{inSubstrate}\n'
                          f'Substrate Concentration: {conc} {inConcentrationUnit}',
                          fontsize=18, fontweight='bold')
                plt.xlabel(f'Time', fontsize=16)
                plt.ylabel(f'Product Concentration', fontsize=16)
                plt.grid(True)
                plt.tight_layout()
                plt.legend(fontsize=12, prop={'weight': 'bold'})
                fig.canvas.mpl_connect('key_press_event', pressKey)
                plt.show()

    return reactionSlopes




# ====================================== Load data =======================================
# Load: Product Standard Curve
prodStdCurve = loadExcel(sheetName=inSheetName[0], loadStandard=True)
m = processStandardCurve(psc=prodStdCurve, plotFig=inPlotAllFigs)

# Load: Kinetics Data
fluorescence = loadExcel(sheetName=inSheetName[1])

# Process kinetics
if inBurstKinetics:
    v = processKineticsBurst(slope=m, datasets=fluorescence, plotFig=inPlotAllFigs)
else:
    v = processKinetics(slope=m, datasets=fluorescence, plotFig=inPlotAllFigs)
MichaelisMenten(velocity=v)
