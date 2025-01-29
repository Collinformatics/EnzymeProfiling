Express DNA
-

Run extractSubstrates.py to express DNA sequneces, and extract the protein substrates.

Input parameters you will need to know include:

Input #1:
- inFileName: select files you want to process.
Important: do not mix files from the intial and final sorts, these must be processed separatly. Files witin the same set can be processed in one batch, or individually, it is up to you how to do this, but I recommend at minimum combining the forward (R1) and reverse reads (R2).

- inBaseFilePath: define the path to the folder for the enzyme you intend on working with.

        Recommend: make a folder for each unique enzyme that contains the folders "Fastq", "Extracted Data", and "Figures"
- inFilePath: define path to the folder with your fastq data
- inSavePath: define path to the folder used to store the extracted data
- inSaveFileName: define the name of your saved files
- inAlertPath: an optional input to play a sound then the script has finished processing the files

Input #2:
- inAAPositions: define the residues of your substrate
    - For a 8 residue substrate, it is recommened to name positions R1 to R8:

          inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
  
- inCodonSequence: define the type of degenerate codon you are working with

Input #3:
- inFixedLibrary: are any of the residues in your substrate not randomized
- inFixedResidue: make a list of what residue(s) should always be in the substrate
- inFixedPosition: where are these AAs expected to be

    If your substrate is NNNLQNNN, where Luecine and Glutamine at the 4th and 5th position then define the inputs as:

      inFixedResidue = True
      inFixedResidue = ['L', 'Q']
      inFixedPosition = [4, 5]
    
    If you have multiple AAs at the same position such as NNN(L/M/F/Y)NNNN, put these residues in a list within the list:
  
        inFixedResidue = [['L', 'M', 'F', 'Y']]
        inFixedPosition = [4]
- inPrintNumber: how many expressed & extracted do you want to print to inspect how the script is performing with your input parameters
- inStartSeqR1, inStartSeqR2: what DNA sequences do you expect to see to the left of your substrate
- inEndSeqR1, inEndSeqR2: what DNA sequences do you expect to see to the right of your substrate

Input #4:
- inPrintCounts: do you want to print the AA counts
- inPlotCounts: do you want to plot the AA counts
- inCountMapYLabel: select how the residue names are displayed in the figure

        inCountMapYLabel = 0 # for Alanine
        inCountMapYLabel = 1 # for Ala
        inCountMapYLabel = 2 # for A


Evaluate Substrates
-

Run evaluateSubstrates.py to load all processed protein substrate files and evaluate enrichment.

- Don't run this script until you have processed all of your fastq/fasta files with extractSubstrates.py

If you are ready to process the extracted data, find the "filePaths" function in functions.py:
- Create a conditional for your enzyme that inclues the names you used to save the extracted substrates, and the lables for the AA positions:

        def filePaths(enzyme):
             if enzyme == 'enzymeName':
                  inFileNamesInitialSort = ['fileNameA', 'fileNameB']
                  inFileNamesFinalSort = ['fileName1', 'fileName2'] 
                  inAAPositions = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']



