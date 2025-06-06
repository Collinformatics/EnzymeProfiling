import esm
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


class NN:
    def __init__(self, substrates):
        self.substrates = substrates





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
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    batch_converter = alphabet.get_batch_converter()


    # Step 3: Convert substrates to ESM model format and generate embeddings
    try:
        batchLabels, batchSubs, batchTokens = batch_converter(subs)
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
    slicedTokens = pd.DataFrame(batchTokens[:, 1:-1],
                                index=batchSubs,
                                columns=subLabel)
    slicedTokens['Values'] = values
    print(f'Sliced Tokens:\n'
          f'{greenLight}{slicedTokens}{resetColor}\n\n')

    return slicedTokens, batchSubs, sampleSize



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



# =================================== Initialize Class ===================================
nn = NN(substrates=substrates)



# ===================================== Run The Code =====================================
output = ESM(substrates=substrates, subLabel=inAAPositions)
