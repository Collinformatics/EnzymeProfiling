Express DNA
-

Run extractSubstrates.py to express DNA sequneces, and extract the protein substrates.

Input #1:
- inBaseFilePath: define the path to the folder for the enzyme you intend on working with.
Recommend: make a folder for each unique enzyme that contains the folders "Fastq", "Extracted Data", and "Figures"

- inFileName: select files you want to process.
Important: do not mix files from the intial and final sorts, these must be processed separatly. Files witin the same set can be processed in one batch, or individually, it is up to you how to do this, but I recommend at minimum combining the forward (R1) and reverse reads (R2).

- inSaveFileName: define the name of your saved files
- inAlertPath: an optional input to play a sound then the script has finished processing the files

Input #2:
- inAAPositions: define the residues of your substrate
- inCodonSequence: define the type of degenerate codon you are working with

Input #3:
- inFixedLibrary: are any of the residues in your substrate not randomized
- inFixedResidue: 

Evaluate Substrates
-

Run evaluateSubstrates.py to load all processed protein substrate files and evaluate enrichment.

