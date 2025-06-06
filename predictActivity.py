import esm
import numpy as np
import pandas as pd
import sys
import torch


inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
substrates = {
    'VVLQSAFE': 357412,
    'VALQAAFA': 311191,
    'VILQAGNS': 309441,
    'LILQSAFK': 291445,
    'IVLQSAFH': 267741,
    'VLLQAAFT': 243158,
    'AVLQAAFN': 218612
}



# ===================================== Set Options ======================================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.5f}'.format)

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

# Set device
print('\n================================== Set Training Device '
      '==================================')
if torch.cuda.is_available():
    device = 'cuda:0'
    print(f'Train with Device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {torch.cuda.get_device_name(device)}{resetColor}\n\n')
else:
    import platform
    device = 'cpu'
    print(f'Train with Device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {platform.processor()}{resetColor}\n\n')



# =================================== Define Functions ===================================
class CNN:
    def __init__(self, substrates):
        self.substrates = substrates



class GradBoostingRegressor:
    def __init__(self, df):
        from sklearn.ensemble import GradientBoostingRegressor


        X = df.drop(columns='count').values
        y = np.log1p(df['count'].values)

        model = GradientBoostingRegressor()
        model = model.to(device)
        model.fit(X, y)



def ESM(substrates, subLabel):
    print('=========================== Convert To Numerical: ESM '
          '===========================')
    print(f'Dataset: {purple}Template{resetColor}\n\n'
          f'Total unique substrates: {red}{len(substrates):,}{resetColor}\n')

    # Extract: Datapoints
    iteration = 0
    collectedTotalValues = 0
    evaluateSubs = {}
    for substrate, value in substrates.items():
        evaluateSubs[str(substrate)] = value
        iteration += 1
        collectedTotalValues += value
        # if iteration >= self.NSubsPCA:
        #     break
    sampleSize = len(evaluateSubs)
    print(f'Collected substrates:{red} {sampleSize:,}{resetColor}')
    if isinstance(collectedTotalValues, float):
          print(f'Total Values:{red} {round(collectedTotalValues, 1):,}'
                f'{resetColor}\n\n')
    else:
        print(f'Total Values:{red} {collectedTotalValues:,}{resetColor}\n\n')

    # Step 1: Convert substrates to ESM model format and generate embeddings
    subs = []
    values = []
    for index, (seq, value) in enumerate(evaluateSubs.items()):
        subs.append((f'Sub{index}', seq))
        values.append(value)



    # Step 2: Load the ESM model and batch converter
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    numLayersESM = 36
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # numLayersESM = 33
    model = model.to(device)


    batchConverter = alphabet.get_batch_converter()


    # Step 3: Convert substrates to ESM model format and generate embeddings
    try:
        batchLabels, batchSubs, batchTokens = batchConverter(subs)
        batchTokensCPU = batchTokens
        batchTokens = batchTokens.to(device)
    except Exception as exc:
        print(f'{orange}ERROR: The ESM has failed to evaluate your substrates\n\n'
              f'Exception:\n{exc}\n\n'
              f'Suggestion:'
              f'     Try replacing: {cyan}esm.pretrained.esm2_t36_3B_UR50D()'
              f'{orange}\n'
              f'     With: {cyan}esm.pretrained.esm2_t33_650M_UR50D()'
              f'{resetColor}\n')
        sys.exit(1)

    print(f'Batch Tokens:{greenLight} {batchTokens.shape}{resetColor}\n'
          f'{greenLight}{batchTokens}{resetColor}\n\n')
    slicedTokens = pd.DataFrame(batchTokensCPU[:, 1:-1],
                                index=batchSubs,
                                columns=subLabel)
    slicedTokens['Values'] = values
    print(f'Sliced Tokens:\n'
          f'{greenLight}{slicedTokens}{resetColor}\n\n')

    with torch.no_grad():
        results = model(batchTokens, repr_layers=[numLayersESM], return_contacts=False)

    # Step 4: Extract per-sequence embeddings
    tokenReps = results["representations"][numLayersESM]  # (N, seq_len, hidden_dim)
    # esm2_t36_3B_UR50D has 36 layers
    # esm2_t33_650M_UR50D has 33 layers
    # esm2_t12_35M_UR50D has 12 layers
    sequenceEmbeddings = tokenReps[:, 0, :]  # [CLS] token embedding: (N, hidden_dim)


    print(f'{greenLight}Extracted embeddings shape: '
          f'{sequenceEmbeddings.shape}{resetColor}\n')

    # Convert to numpy + store with counts
    embeddings = sequenceEmbeddings.cpu().numpy()
    values = np.array(values).reshape(-1, 1)
    data = np.hstack([embeddings, values])

    columns = [f'feat_{i}' for i in range(embeddings.shape[1])] + ['count']
    subEmbeddings = pd.DataFrame(data, index=batchSubs, columns=columns)

    return slicedTokens, batchSubs, sampleSize, subEmbeddings



# ===================================== Run The Code =====================================
output = ESM(substrates=substrates, subLabel=inAAPositions)



# ================================== Initialize Classes ==================================
# cnn = CNN(substrates=substrates)
regressor = GradBoostingRegressor(df=output[3])


