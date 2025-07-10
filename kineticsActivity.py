import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys



# Input 1: Select Dataset
inEnzymeName = 'Mpro2'
inFileName = f'Rxn Kinetics - {inEnzymeName}'
inPathFolder = f'{inEnzymeName}/Kinetics'
inSheetName = ['']

# Input 2: Reaction Data
inPlotRxnDataIndex = 2



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



def loadExcel(sheetName=''):
    print('=================================== Load Data '
          '===================================')
    print(f'Loading excel sheet: {pink}{sheetName}{resetColor}')
    path = os.path.join(inPathFolder, inFileName)
    if '.xlsx' not in path:
        path += '.xlsx'
    print(f'Loading file at path:\n     {greenDark}{path}{resetColor}\n\n')

    # Load data
    if sheetName == '':
        data = pd.read_excel(path, header=[0])
    else:
        data = pd.read_excel(path, sheet_name=sheetName, header=[0])

    # Remove NaN
    dataFiltered = data.dropna(how='all')
    dataFiltered = dataFiltered.dropna(axis=1, how='all')

    # Set Headers
    dataFiltered.columns = list(dataFiltered.iloc[0])  # Set the first row as header
    dataFiltered = dataFiltered[1:]
    dataFiltered = dataFiltered.reset_index(drop=True)

    # Set index
    dataFiltered.index = dataFiltered.iloc[:, 0]
    dataFiltered.set_index(dataFiltered.columns[0], inplace=True)
    dataFiltered.index.name = None

    for column in dataFiltered.columns:
        if dataFiltered[column].isna().any():
            print(f'Dropping:\n{dataFiltered[column]}\n\n')
            dataFiltered = dataFiltered.drop(column, axis=1)

    print(f'Raw Data: {pink}{sheetName}{resetColor}\n{data}\n\n\n'
          f'Loaded Data: {pink}{sheetName}{resetColor}\n{dataFiltered}\n\n')

    return dataFiltered



def plotActivity(data):
    print('=============================== Reaction Kinetics '
          '===============================')
    index = data.index[inPlotRxnDataIndex]
    activity = data.loc[index, :].astype(float)
    if  'km' in index.lower():
        activity = activity.sort_values(ascending=False)
    else:
        activity = activity.sort_values(ascending=True)
    print(f'Plotting reaction data:\n{greenLight}{activity}{resetColor}\n\n')


    # Plotting
    fig, ax = plt.subplots(figsize=(9.5, 8))
    plt.barh(activity.index, activity.values, color='#BF5700', edgecolor='black')
    plt.title(f'{inEnzymeName}\nSubstrate Activity', fontsize=18, fontweight='bold')
    plt.xlabel(f'{index}', fontsize=16)
    # plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', length=4, labelsize=13)
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    fig.canvas.mpl_connect('key_press_event', pressKey)

    plt.show()



# ====================================== Load data =======================================
# Load: Kinetics run
rxn = loadExcel(sheetName=inSheetName[0])

# Plot reaction data
plotActivity(data=rxn)
