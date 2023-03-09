

### To obtain the latex tables of results for the metrics present in the paper and average transfer recall results (recall @ 10), 
one runs the evaluation_image_privacy.py file with the following list of parameters: 

"--original_dir" --  Directory with original (unmodified images);

"--confounder_features_dir" -- Directory with confounders;

"--folder_paths" -- All the folders with modified 'private' versions of original images;

"--folder_names" -- Titles of each folder to be printed in latex code;

"--results_folder", Path to results directory;

"--num_images_per_person", number of images per each person present in the original dataset;

"--experiment_name", title of a current experiment, also used as a name of a log file;

'-m', '--method', first embedding method used in optimization process, if all embedding methods should be used for optimization process, specify 'all';

'-m1', '--method_1' - second embedding method used in optimization process, if it is not needed specify None.
