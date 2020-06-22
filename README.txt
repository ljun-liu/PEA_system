update 22/06/2020
(1). correct some mistakes in Propre.py

-------------------------------------------------------------------------------
update 09/06/2020
(1). replace the old 'run.sh' by the modified one;
(2). change the structure of PEA system;
(3). add Confidence colunmn in classes_error table;
(4). align the numbers in the table at the centre of cells;
(5). distinguish the 'Overall' bar with a different colour

-------------------------------------------------------------------------------
update 08/06/2020
(1). create a figure directory for comparison graphs

-------------------------------------------------------------------------------
update 27/05/2020

(1). add a function for drawing the class_confidence graph
-------------------------------------------------------------------------------
update 25/05/2020

(1). add a flag to control whether showing the number in the graphs or not;
(2). sort the bars in the extra graph
-------------------------------------------------------------------------------
update 24/05/2020

(1). add 'Overall' row (Sub/Del/Ins/PER) in the class_errorinfo table;
(2). add 'Overall' (Sub/Del/Ins/PER) in the class_errorinfo bar graph;
(3). add '# Phonemes' column for count of reference phonemes per class in the class_errorinfo table;
(4). add a bar graph about phonemic classes and #tokens in the results directory;
(5). align the numbers in the table based on rightmost numbers;
(6). modify the width between xticklabels;
(7). add an extra detailed class_errorinfo figure with 4 subplots (Sub/Del/Ins/PER)
-------------------------------------------------------------------------------
update 22/05/2020

This PEA system is designed to evaluate analyze ASR systems with PER on the level of phonemic classes. Besides, comparison among different ASR systems is available in this system.

There are two packages in the 'src' directory: 'data' and 'main'.
The 'data' dir stores the raw results from ASR system to be tested (in 'score_5' dir) and the analysis output from this PEA system (in 'output' dir).
The 'main' dir contains the core functional files, which are 'Main.py', 'Prepro.py', 'PER.py' and 'Analysis.py'.

Here I will introduce the usage of this system.
1. assign values to input arguments "scoring_dir" and "results_dir" in the shell file 'run.sh'. It's important to note that the directories should end up with '/'.
2. ensure you are in this PEA system to execute the command script successfully; 
3. run 'run.sh'.

After running this system, you will obtain the detailed analysis files in 'YOUR_RESULTS_DIRECTORY'.
