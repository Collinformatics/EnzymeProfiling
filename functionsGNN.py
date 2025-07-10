import math
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import platform
from rdkit import Chem
from rdkit.Chem import Draw
import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



# Colors: Console
greyDark = '\033[38;2;144;144;144m'
purple = '\033[38;2;189;22;255m'
magenta = '\033[38;2;255;0;128m'
pink = '\033[38;2;255;0;242m'
cyan = '\033[38;2;22;255;212m'
blue = '\033[38;5;51m'
green = '\033[38;2;5;232;49m'
greenLight = '\033[38;2;204;255;188m'
greenLightB = '\033[38;2;204;255;188m'
greenDark = '\033[38;2;30;121;13m'
yellow = '\033[38;2;255;217;24m'
orange = '\033[38;2;247;151;31m'
red = '\033[91m'
resetColor = '\033[0m'



inEnzymeName = 'Mpro2'
inPathFolder = f'{inEnzymeName}'
inSaveFigures = True

inMinimumSubstrateCount = 10
inUseEnrichmentFactor = True

inPlotSub = False
inBatchSize = 10



substrates = {
    'AVLQSGFR': 21945,
    'VKLQTNAV': 21581,
    'GGLQNSMK': 21217,
    'TILQADGR': 20853,
    'SPLQKMCH': 20489,
    'NCLQGRPA': 20125,
    'HYLQSTAT': 19761,
    'EKLQYAGL': 19397,
    'QRLQDVAY': 19033,
    'DVLQRMNT': 18669,
    'WWLQGQLE': 18305,
    'PYLQTTFS': 17941,
    'MKLQNAVT': 17577,
    'YGLQLGKI': 17213,
    'RKLQSTDR': 16849,
    'CKLQNGSR': 16485,
    'LVLQAAFK': 16121,
    'FALQKVLS': 15757,
    'TSLQGGAE': 15393,
    'KRLQSPMN': 15137,
}

maxValue = max(substrates.values())
for substrate, score in substrates.items():
    substrates[substrate] = score / maxValue



# =================================== Define Functions ===================================
def pressKey(event):
    if event.key == 'escape':
        plt.close()
    elif event.key == 'e':
        sys.exit()
    elif event.key == 'r':
        python = sys.executable # Doesnt seem to work on windows?
        os.execl(python, python, *sys.argv)



# ================================= Define ML Algorithms =================================
class graphNN(torch.nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super().__init__()
        self.conv1 = GCNConv(inputDim, hiddenDim)
        self.conv2 = GCNConv(hiddenDim, outputDim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(outputDim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1) # final output dimension (e.g., regression)
        )

    def forward(self, x, edgeIndex, batch):
        x = self.conv1(x, edgeIndex)
        x = torch.relu(x)
        x = self.conv2(x, edgeIndex)
        x = global_mean_pool(x, batch)  # aggregates node embeddings per graph
        out = self.mlp(x).view(-1)
        return out



class GNN:
    def __init__(self, folderPath, substrates, enzymeName, datasetTag, minSubCount, useEF,
                 epochs, batchSize, saveFigures):
        # Parameters: Files
        self.pathFolder = folderPath
        self.pathData = os.path.join(self.pathFolder, 'Data')
        self.pathModels = os.path.join(self.pathFolder, 'Models')
        self.pathFigures = os.path.join(self.pathFolder, 'Figures')
        self.pathFiguresTraining = os.path.join(self.pathFolder, 'FiguresModelTraining')
        os.makedirs(self.pathData, exist_ok=True)
        os.makedirs(self.pathModels, exist_ok=True)
        os.makedirs(self.pathFigures, exist_ok=True)
        os.makedirs(self.pathFiguresTraining, exist_ok=True)

        # Parameters: Data
        self.substrates = substrates
        self.substratesTest = None
        self.substratesNum = len(substrates.keys())
        self.subsLen = len(next(iter(self.substrates)))
        self.enzymeName = enzymeName
        self.datasetTag = datasetTag
        self.minSubCount = minSubCount
        self.useEF = useEF
        self.mol = []
        self.plotSubstrate()

        # Parameters: Model
        self.model = None
        self.device = self.getDevice()
        self.testingSetSize = 0.2
        self.epochs = epochs
        self.batchSize = batchSize

        # Parameters: Figures
        self.saveFigures = saveFigures
        self.figSize = (9.5, 8)
        self.labelSizeTitle = 18
        self.labelSizeAxis = 16
        self.labelSizeTicks = 13
        self.lineThickness = 1.5
        self.tickLength = 4
        self.figureResolution = 300

        # Parameters: Misc
        self.roundVal = 3

        # Parameters: Model Path
        if self.useEF:
            scoreType = 'EF'
        else:
            scoreType = 'Counts'
        self.tagExperiment = \
            (f'{self.enzymeName} - {self.datasetTag} - Test Size {self.testingSetSize} - '
             f'Batch {self.batchSize} - {scoreType} - MinCounts {self.minSubCount} - '
             f'N {self.substratesNum} - {self.subsLen} AA')
        self.tagGNN = f'GNN - {self.tagExperiment}'
        self.pathGNN  = os.path.join(self.pathModels, f'{self.tagGNN}.pt')


        # Generate: Molecular graph
        self.graph = self.molToGraph()

        # Train model
        self.trainGNN()



    def trainGNN(self):
        print('================================= Training GNN '
              '==================================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n'
              f'Loss Function: {purple}Mean Squared Error{resetColor}\n'
              f'Testing Size: {red}{self.testingSetSize}{resetColor}\n'
              f'Batch Size: {red}{self.batchSize}{resetColor}\n')
        self.model = graphNN(inputDim=1, hiddenDim=32, outputDim=1)
        self.model.to(self.device)

        # Randomly split the dataset
        trainGraphs, testGraphs = train_test_split(self.graph, test_size=0.2,
                                                   random_state=42)
        self.substratesTest = [data.substrate for data in testGraphs]


        # Create dataloaders
        loaderTrain = DataLoader(trainGraphs, batch_size=self.batchSize, shuffle=True)
        loaderTest = DataLoader(testGraphs, batch_size=self.batchSize, shuffle=False)


        if os.path.exists(self.pathGNN):
            print(f'Loading model:\n'
                  f'     {greenDark}{self.pathGNN}{resetColor}\n\n')
            self.model.load_state_dict(torch.load(self.pathGNN))
        else:
            # Set: Optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            # Train the model
            for epoch in range(1, self.epochs+1):
                self.model.train()
                totalLoss = 0
                for batch in loaderTrain:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = F.mse_loss(out, batch.y.float()) # depends on your task
                    loss.backward()
                    optimizer.step()
                    totalLoss += loss.item()
                if epoch % 20 == 0:
                    print(f'Epoch: {red}{epoch}{resetColor} / {red}{self.epochs}'
                          f'{resetColor}\n'
                          f' Loss: {red}{round(totalLoss, self.roundVal):,}'
                          f'{resetColor}\n')
                    # print(f'Predicted: {greenLight}{out}{resetColor}\n\n'
                    #       f'Activity: {greenLight}{batch.y.float()}{resetColor}\n')

            # Save the model
            print(f'Saving model at:\n'
                  f'     {greenDark}{self.pathGNN}{resetColor}\n\n')
            torch.save(self.model.state_dict(), self.pathGNN)

        self.testModel(testingLoader=loaderTest, modelType='GNN')



    def testModel(self, testingLoader, modelType):
        print('============================ Testing Model Accuracy '
              '=============================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}\n'
              f'Loss Function: {purple}Mean Squared Error{resetColor}\n'
              f'Model Type: {purple}{modelType}{resetColor}')
        yTest = []
        yPred = []
        testLoss = 0

        # Test the model
        self.model.eval()
        with torch.no_grad():
            for batch in testingLoader:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = F.mse_loss(out.view(-1), batch.y.float())
                testLoss += loss.item()

                # Collect values
                yTest.extend(batch.y.cpu().numpy())
                yPred.extend(out.view(-1).cpu().numpy())

        # Collect datapoints
        results = pd.DataFrame({
            'yTest': yTest,
            'yPred': yPred
        })
        results.index = self.substratesTest
        results = results.sort_values(by='yTest', ascending=False)
        R2 = r2_score(results.loc[:, 'yTest'], results.loc[:, 'yPred'])

        print(f'Model Accuracy:\n'
              f'     MSE: {red}{(testLoss / len(testingLoader)):.3e}{resetColor}\n'
              f'     R²: {red}{R2:.3e}{resetColor}\n\n')

        self.plotTestingPredictions(data=results, modelType=modelType, R2=R2)



    def plotTestingPredictions(self, data, modelType, R2):
        print(f'====================== Scatter Plot - Prediction Accuracy '
              f'======================')
        print(f'Dataset: {purple}{self.datasetTag}{resetColor}')
        x = data.loc[:, 'yTest']
        y = data.loc[:, 'yPred']

        # Evaluate data
        maxValue = max(max(x), max(y))
        magnitude = math.floor(math.log10(maxValue))
        unit = 10 ** (magnitude - 1)
        maxValue = math.ceil(maxValue / unit) * unit
        maxValue += unit
        axisLimits = [0, maxValue]
        print(f'R²: {red}{R2:.3e}{resetColor}\n\n'
              f'Data: {purple}{modelType}{resetColor}\n'
              f'{data}\n\n')

        useSciNotation = True
        title = f'\n{self.datasetTag}\nR² = {round(R2, self.roundVal)}\nGNN'
        if useSciNotation:
            title = f'\n{self.datasetTag}\nR² = {R2:.3e}\nGNN'



        # Plot the data
        fig, ax = plt.subplots(figsize=self.figSize)
        plt.scatter(x, y, alpha=0.7, color='#BF5700', edgecolors='#F8971F', s=50)
        plt.plot(axisLimits, axisLimits, color='#101010', lw=self.lineThickness)
        plt.xlabel('Experimental Activity', fontsize=self.labelSizeAxis)
        plt.ylabel('Predicted Activity', fontsize=self.labelSizeAxis)
        plt.title(title,
                  fontsize=self.labelSizeTitle, fontweight='bold')
        plt.subplots_adjust(top=0.852, bottom=0.075, left=0.162, right=0.935)

        # Set axes
        ax.set_xlim(0, maxValue)
        ax.set_ylim(0, maxValue)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', length=self.tickLength,
                       labelsize=self.labelSizeTicks, width=self.lineThickness)

        # Set the thickness of the figure border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        plt.grid(True)
        fig.canvas.mpl_connect('key_press_event', pressKey)
        plt.show()


        # Define: Save location
        figLabel = f'Prediction Accuracy - {self.tagGNN}' + '.png'
        saveLocation = os.path.join(self.pathFiguresTraining, figLabel)

        # Save figure
        if os.path.exists(saveLocation):
            print(f'{yellow}WARNING{resetColor}: '
                  f'{yellow}The figure already exists at the path\n'
                  f'     {saveLocation}\n\n'
                  f'We will not overwrite the figure{resetColor}\n\n')
        else:
            print(f'Saving figure at path:\n'
                  f'     {greenDark}{saveLocation}{resetColor}\n\n')
            fig.savefig(saveLocation, dpi=self.figureResolution)



    def molToGraph(self):
        print('===================== Convert Substrates To Molecular Graph '
              '=====================')
        print(f'Total Substrates: {red}{self.substratesNum}{resetColor}\n')

        # Evaluate: Peptides
        for substrate in self.substrates.keys():
            self.mol.append(Chem.MolFromSequence(substrate))

        graphList = []
        substrates = list(self.substrates.keys())

        # Generate graph
        for index, mol in enumerate(self.mol):
            substrate = substrates[index]
            atomFeatures = []
            edgeIndex = []
            edgeAttr = []

            # Atom features
            for atom in mol.GetAtoms():
                atomFeatures.append([atom.GetAtomicNum()])

            # Bond features
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edgeIndex += [[i, j], [j, i]]  # Undirected edges
                edgeAttr += [[bond.GetBondTypeAsDouble()]] * 2

            # Convert to tensors
            x = torch.tensor(atomFeatures, dtype=torch.float)
            y = torch.tensor(self.substrates[substrate], dtype=torch.float64)
            edgeIndex = torch.tensor(edgeIndex, dtype=torch.long).T
            edgeAttr = torch.tensor(edgeAttr, dtype=torch.float)

            # Collect data
            data = Data(x=x, edge_index=edgeIndex, edge_attr=edgeAttr, y=y)
            data.substrate = substrate # Record substrates
            if graphList == []:
                N = 10
                print(f'Evaluating Substrates:'
                      f'Substrate: {pink}{next(iter(self.substrates))}{resetColor}\n'
                      f'Activity: {red}{y}{resetColor}\n\n'
                      f'X (First {red}{N}{resetColor} atoms):\n'
                      f'{greenLight}{x[:N]}{resetColor}\n\n'
                      f'Edge Indices:\n'
                      f'{greenLight}{edgeIndex}{resetColor}\n\n'
                      f'Edge Attributes (First {red}{N}{resetColor} Bond Types):\n'
                      f'{greenLight}{edgeAttr[:N]}{resetColor}\n\n'
                      f'Graph:\n'
                      f'     {greenLight}{data}{resetColor}\n\n')
            graphList.append(data)

        return graphList



    def plotSubstrate(self):
        # Plot molecule
        if inPlotSub:
            substrate = next(iter(self.substrates))
            print(f'Plotting substrate: {pink}{substrate}{resetColor}')
            mol = Chem.MolFromSequence(substrate)
            saveLocation = f'{self.enzymeName}/Figures/molecule - {substrate}.png'
            img = Draw.MolToImage(mol, size=(1000, 800))
            if not os.path.exists(saveLocation):
                print(f'Saving molecule at path:\n'
                      f'     {greenDark}{saveLocation}{resetColor}\n\n')
                img.save(saveLocation, dpi=(300, 300))
            img.show()  # Opens in default viewer



    def getDevice(self):
        # Set device
        print('============================== Set Training Device '
              '==============================')
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f'Train with Device:{magenta} {device}{resetColor}\n'
                  f'Device Name:{magenta} {torch.cuda.get_device_name(device)}'
                  f'{resetColor}\n\n')
        else:
            device = 'cpu'
            print(f'Train with Device:{magenta} {device}{resetColor}\n'
                  f'Device Name:{magenta} {platform.processor()}{resetColor}\n\n')

        return device


# graph = GNN(folderPath=inPathFolder, substrates=substrates, enzymeName=inEnzymeName,
#             datasetTag='Testing Substrates', minSubCount=inMinimumSubstrateCount,
#             useEF=inUseEnrichmentFactor, batchSize=inBatchSize, saveFigures=inSaveFigures)
