# Python Modules
- You will need to install the following modules:

        pip install biopython
        pip install fair-esm
        pip install logomaker
        pip install matplotlib
        pip install numpy
        pip install pandas
        pip install playsound
        pip install PyObjC
        pip install seaborn
        pip install scikit-learn
        pip install torch
        pip install wordcloud

# Keyboard Shortcuts

Esc: Close the current figure

F12: Rerun the script

Backspace: Terminate the script that is currently running

Note:

- For the shortcuts to work, the displayed figure must be selected.

- If nothing happens after pressing a key, click on the figure and try again.

# General Information

These scripts are designed to extact a protein substrate sequence from a longer DNA sequence. 

The extracted sequences are grouped into two catagories, the Unfilteded (or Initial) and Filtered (or Final) sorts.
- Unfiltered: contains the background data consisting of randomized substrate sequnces
- Filtered: contains the set of substrates were initially in the Unsorted set, and that were found to be active, then colleced with Fluorescence Activated Cell Sorting

The specificity of the enzyme is evaluated by calculating the Enrichment Score of each amino acid (AA).
- The enrichment score compairs the frequency of the AAs in the Filtered dataset to the Unfiltered set

# Express DNA

Run extractSubs.py to express DNA sequneces, and extract the protein substrates.

Input parameters you will need to know include:

Input 1:
- inFileName: select files you want to process.
  - Important: do not mix files from the intial and final sorts, these must be processed separatly. Files witin the same set can be processed in one batch, or individually, it is up to you how to do this, but I recommend at minimum combining the forward (R1) and reverse reads (R2).
- inPathFolder: define path to the folder for your 
- inPathDNASeqs: add the name of the folder with your fastq data
- inSaveFileName: define the name of your saved files (This should be related to the input files)

Input 2:
- inSaveFileName: define the name of your saved files (This should be related to the input files)

  - If:
  
          inFileName = ['Fyn-I_S6_L001_R1_001', 'Fyn-I_S6_L001_R2_001']

  - Recommend:
  
          inSaveFileName = 'Fyn-I_S6_L001'

Input 3:
- inAAPositions: name the positions in the substrate

  - Recommended: R1 to RN, where N = substrate length

        inAAPositions = ['R1','R2','R3','R4','R5','R6','R7','R8']


Input 4:
- inPrintNumber: how many expressed & extracted do you want to print to inspect how the script is performing with your input parameters
- inStartSeqR1, inStartSeqR2: what DNA sequences do you expect to see to the left of your substrate
- inEndSeqR1, inEndSeqR2: what DNA sequences do you expect to see to the right of your substrate

Input 5:
- inFixedLibrary: are any of the residues in your substrate not randomized
- inFixedResidue: make a list of what residue(s) should always be in the substrate
- inFixedPosition: where are these AAs expected to be

    If your substrate is NNNLQNNN, where Luecine and Glutamine at the 4th and 5th position then define the inputs as:

      inFixedLibrary = True
      inFixedResidue = ['L', 'Q']
      inFixedPosition = [4, 5]
    
    If you have multiple AAs at the same position such as NNN(L/M/F/Y)NNNN, put these residues in a list within the list:
  
        inFixedResidue = [['L', 'M', 'F', 'Y']]
        inFixedPosition = [4]

Input 6:
- inAlertPath: an optional input to play a sound then the script has finished processing the files


# Evaluate Substrates (Incomplete Section)

Run evaluateSubstrates.py to load all processed protein substrate files and evaluate enrichment.

- Don't run this script until you have processed all of your fastq/fasta files with extractSubstrates.py

If you are ready to process the extracted data, find the "filePaths" function in functions.py:
- Create a conditional for your enzyme that inclues the names you used to save the extracted substrates, and the lables for the AA positions:

        def filePaths(enzyme):
             if enzyme == 'enzymeName':
                  inFileNamesInitialSort = ['fileNameA', 'fileNameB']
                  inFileNamesFinalSort = ['fileName1', 'fileName2'] 
                  inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']

- Use else statments for the easy processing of multiple datasets:

          def filePaths(enzyme):
               if enzyme == 'enzymeName':
                    inFileNamesInitialSort = ['fileNameA', 'fileNameB']
                    inFileNamesFinalSort = ['fileName1', 'fileName2']
                    inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
               elif enzyme == 'enzymeName2':
                    inFileNamesInitialSort = ['fileNamC', 'fileNameD']
                    inFileNamesFinalSort = ['fileName3', 'fileName4']
                    inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']

Input parameters you will need to know include:

Input 1:
- inEnzymeName: select the enzyme you are working with
- inFilePath: define path to the folder with your extracted data
- inSavePathFigures: define path to the folder used to store your figures
- inSaveFigures: Set as True to automatically save your figures, Set as False to not automatically save the figures (this will not overwrite previously saved figures)

Input 2:
- Select which figure you want to plot

Input 3:
- inFilterSubstrates: do you want to select substrates with specific residue(s) in the substrate
- inFixedResidue: make a list of what residue(s) should always be in the substrate
- inFixedPosition: where are these AAs expected to be
- inExcludeResidues: do you want to select substrates without specific residue(s) in the substrate
- inExcludedResidue: define excluded AAs
- inExcludedPosition: define positions to exclude the AAs
- inMinimumSubstrateCount: exclude substrates with less than this value

# Motif Eval (Incomplete Section)

Input 2:

- inMotifPositions: Label the residues in the motif sequence

        inMotifPositions = ['P4', 'P3', 'P2', 'P1', 'P1\'', 'P2\'']

- inIndexNTerminus: Define the index of the first position in "inMotifPositions"

  - If the positions in the full substrate sequnece are:
 
        pos = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']

    - If: inIndexNTerminus = 0, the motif will start at 'R1', and extend to 'R6'
    
    - If: inIndexNTerminus = 1, the motif will start at 'R2', and extend to 'R7'

# Miscellaneous

Figures: Word Cloud

- The orientations of the words, and their colors will be randomized with each figure. 

- To remake the figure with a new arrangemnt, press the "F12" key (when the figure is selected) to rerun the script.

  - Recommendation: Turn off the switches that plot the Enrichment Maps, and Logos to avoid spending time plotting unnecessary figures before plotting the Word Cloud


