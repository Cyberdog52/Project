
Before you start with the project:

1. 	Download validation, test and train sets and put them in this directory
2. 	Run tf_record_to_numpy 3 times for each set (train, test, validation)
	This will create pkl files in the same directory
	You can delete the tf_record files now, don't need them anymore
3.	Merge the 18 validation pkl files into train folder. Rename them by hand as if they were train files (from 59 to 76)
	There should now be 76 dataTrain pkl files in the train directory
	You can delete the validation folder now, don't need them anymore
4. 	Run produce_masked_inputs.py twice for train and test
	You can delete dataTrain and dataTest, from now use newTrain and newTest


Important things:

Some videos are shorter than 50 frames, interpolate them to 50
It is better to delete the first and last images of a video than keeping all
Some segmentation images are blank, do not segment if they are blank
