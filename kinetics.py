from tokenize import group

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
import sys


# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inFileName = 'Kinetics template-SVTFQSAVK'
inPathFolder = f'{inEnzymeName}/Kinetics'
inSheetName = ['Product standard', 'SVTFQSAVK_Raw data']

# Input 2: Process Data
inConcentrationUnit = 'uM'
inPlotAllFigs = False
inStdCurveStartIndex = 0
inConcCutoff = 10 # Max percentage of substrate concentration in rxn plot



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
    print(f'Loading Excel Sheet: {pink}{sheetName}{resetColor}')
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
        df = pd.DataFrame(0.0, index=[], columns=[])
        for conc, value in indexConc.items():
            # Get only the data rows, remove the first 2 rows (NaN and Time)
            filteredIndex = data.index[2:]
            triplicate = data.loc[filteredIndex, [data.columns[i] for i in value]].copy()

            # 2. Redefine column headers
            newHeaders = data.iloc[1, value].tolist() # row 1 = 'G1', 'H1', 'I1'
            triplicate.columns = newHeaders
            triplicate.index.name = indexHeader
            dataFiltered[f'{conc} {inConcentrationUnit}'] = triplicate

            # Statistical analysis
            triplicate.loc[:, 'Avg'] = round(triplicate.loc[:, :].mean(axis=1), 2)
            triplicate.loc[:, 'StDev'] = round(triplicate.std(axis=1), 2)

        print(f'Loaded Data: {pink}{sheetName}{resetColor}')
        for conc, value in dataFiltered.items():
            print(f'Concentration: {purple}{conc}{resetColor}\n{value}\n\n')


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
        plt.title('\nProduct Standard Curve', fontsize=18, fontweight='bold')
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
    print(f'Slope: {red}{round(slope, 3)}{resetColor}')
    print(f'Slope: {red}{round(slope, 3)}{resetColor}')
    print(datasets)
    print('\n')


    # Process florescence
    for conc, values in datasets:
        # group = dataset[conc].copy()
        # group.loc[:, 'Avg'] = round(group.loc[:, :].mean(axis=1), 2)
        # group.loc[:, 'StDev'] = round(group.std(axis=1), 2)
        group.loc[:, '[Prod]'] = round((group.loc[:, 'Avg'] / slope), 2)

        print(f'Values: {purple}{conc}{resetColor}\n{group}\n\n')

        # Extract product concentrations

        datasets[conc] = group
    print(f'Reactions:\n{reactions}\n\n')


    # # Plot all reaction progress
    # x = reactions.index
    # fig, ax = plt.subplots(figsize=(9.5, 8))
    #
    # # Loop through each concentration and plot its [product] values
    # for conc in reactions.columns:
    #     y = reactions[conc]
    #     ax.plot(x, y, marker='o', label=conc)  # or use ax.scatter for dots only
    #
    # ax.set_title('Kinetics!', fontsize=18, fontweight='bold')
    # ax.set_xlabel('Time', fontsize=16)
    # ax.set_ylabel('Product Concentration', fontsize=16)
    # ax.grid(True)
    # ax.legend(title='[Substrate]')
    #
    # fig.tight_layout()
    # fig.canvas.mpl_connect('key_press_event', pressKey)
    # plt.show()



    # Plot individual reactions
    reactionSlopes = {}
    subConc = []
    for conc, values in datasets.items():
        if 'uM' in conc:
            numericConc = float(conc.replace('uM', ''))
        else:
            numericConc = float(conc.replace('mM', ''))
        subConc.append(numericConc)
        cutoff = numericConc / inConcCutoff

        print(f'Concentration: {red}{conc}{resetColor}\n'
              f'Maximum product concentration: {red}{cutoff}{resetColor}')

        # Filter datapoints
        x = values.index
        y = values.loc[:, '[Prod]']
        mask = y <= cutoff
        x = x[mask]
        y = y[mask]

        # Fit the datapoints
        slope, intercept = np.polyfit(x, y, 1)  # Degree 1 polynomial = linear
        fitLine = slope * x + intercept
        fit = f'y = {slope:.3f}x + {intercept:.3f}'
        print(f'Fit: {red}{fit}{resetColor}\n')
        reactionSlopes[conc] = slope


        if plotFig:
            # Scatter plot
            fig, ax = plt.subplots(figsize=(9.5, 8))
            plt.scatter(x, y, color='#BF5700', marker='o')
            plt.title(f'Substrate Concentration: {conc}\n'
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
    print()

    return subConc, reactionSlopes



def MichaelisMenten(substrateConc, velocity):
    print('=============================== Michaelis-Menten '
          '================================')
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

    print(f'Vmax: {red}{vMax:.3f}{resetColor}\n'
          f'Km = {red}{Km:.3f}{resetColor}\n\n')

    # Fit the data
    xFit = np.linspace(0, max(substrateConc) * 1.2, 100)
    yFit = MM(xFit, vMax, Km)


    # Plot the data
    fig, ax = plt.subplots(figsize=(9.5, 8))
    plt.scatter(substrateConc, v, color='#BF5700')
    plt.plot(xFit, yFit, color='#32D713', label='Michaelis-Menten fit')
    plt.title(f'Michaelis-Menten\n{inEnzymeName}\nVmax = {vMax}\nKm = {Km}',
              fontsize=18, fontweight='bold')
    plt.xlabel(f'[Substrate]', fontsize=16)
    plt.ylabel(f'Velocity', fontsize=16)
    plt.tight_layout()
    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()



# ====================================== Load data =======================================
# Load: Product Standard Curve
prodStdCurve = loadExcel(sheetName=inSheetName[0], loadStandard=True)
m = processStandardCurve(psc=prodStdCurve, plotFig=inPlotAllFigs)

# Load: Kinetics Data
fluorescence = loadExcel(sheetName=inSheetName[1])

# Process kinetics
subConc, v = processKinetics(slope=m, dataset=fluorescence, plotFig=inPlotAllFigs)
MichaelisMenten(substrateConc=subConc, velocity=v)
