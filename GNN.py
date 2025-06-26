import os.path
import sys
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
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



class GNN(torch.nn.Module):
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



class graphSubstrates:
    def __init__(self, substrates, enzymeName, datasetTag, batchSize):
        # Parameters: Data
        self.substrates = substrates
        self.substratesNum = len(substrates.keys())
        self.enzymeName = enzymeName
        self.datasetTag = datasetTag
        self.mol = []
        self.plotSubstrate()

        # Parameters: Misc
        self.roundVal = 3
        self.testSize = 0.2

        # Generate: Molecular graph
        self.graph = self.molToGraph()

        # Train model
        self.batchSize = batchSize
        self.trainGNN()



    def trainGNN(self):
        print('================================= Training GNN '
              '==================================')
        print(f'Loss Function: {purple}Mean Squared Error{resetColor}\n'
              f'Testing Size: {red}{self.testSize}{resetColor}\n'
              f'Batch Size: {red}{self.batchSize}{resetColor}\n')

        # Randomly split the dataset
        trainGraphs, testGraphs = train_test_split(self.graph, test_size=0.2,
                                                   random_state=42)

        # Create separate loaders
        loaderTrain = DataLoader(trainGraphs, batch_size=self.batchSize, shuffle=True)
        loaderTest = DataLoader(testGraphs, batch_size=self.batchSize, shuffle=False)
        model = GNN(inputDim=1, hiddenDim=32, outputDim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        epochs = 100
        for epoch in range(1, epochs+1):
            model.train()
            totalLoss = 0
            for batch in loaderTrain:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.mse_loss(out, batch.y.float()) # depends on your task
                loss.backward()
                optimizer.step()
                totalLoss += loss.item()
            if epoch % 25 == 0:
                print(f'Epoch: {red}{epoch}{resetColor} / {red}{epochs}{resetColor}\n'
                      f' Loss: {red}{round(totalLoss, self.roundVal):,}{resetColor}\n')
                # print(f'Predicted: {greenLight}{out}{resetColor}\n\n'
                #       f'Activity: {greenLight}{batch.y.float()}{resetColor}\n')

        # After training loop
        model.eval()
        testLoss = 0
        with torch.no_grad():
            for batch in loaderTest:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.mse_loss(out.view(-1), batch.y.float())
                testLoss += loss.item()
        print(f'Testing Model Accuracy:\n'
              f'     MSE: {red}{round(testLoss / len(loaderTest), self.roundVal)}'
              f'{resetColor}\n\n')



    def molToGraph(self):
        print('===================== Convert Substrates To Molecular Graph '
              '=====================')
        print(f'N Substrates: {red}{self.substratesNum}{resetColor}\n')

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
            if graphList == []:
                N = 10
                print(f'Substrate: {pink}{next(iter(self.substrates))}{resetColor}\n'
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


graph = graphSubstrates(substrates=substrates, enzymeName='Mpro2',
                        datasetTag='Testing Substrates', batchSize=inBatchSize)
