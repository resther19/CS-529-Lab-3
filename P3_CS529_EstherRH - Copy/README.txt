++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++		CS 529:  Project 3 Music Genre Classification	    +	                +                                                               + +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Files:

CNNmodel.py 	 	     implementation of the CNN classifier 

cfg.py			     config file

predict.py			     gives predictions for the training set

predict_test.py 		     gives predictions for the testing ser

SVM_with_wavelets.m	     implementation of the SVM classifier

helperscatfeatures.m       helper for SVM_with_wavelets.m that      					extracts the features as wavelets 						coefficients

modecount.m                helper for SVM_with_wavelets.m that 						returns the modes of the class labels

DataRepFigures.m 		creates the figures for the 3 							different representations of the data
					(If used with Linux/Mac keep as it is, 					if used with Windows need to change 						the folder path for each song from 						train\*****.mp3 to train/*****.mp3)

train folder			folder that contains the .mp3 files 						for the training set

test folder			folder that contains the .mp3 files 						for the testing set


clean16k folder			folder that contains clean .wav files
					for the training ser

clean_test folder		folder that contains clean .wav files
					for the testing set

models folder		     to save the convolutional model

pickles folder			to save data 

train.csv				testing data songs id with labes

test_idx.csv			training data songs id

80_20_predictions.csv		the output file of the predictions 						given by the CNN for the 80-20 split 						of the training data (this file is 						used by SVM_with_wavelets.m to find 						the confusion matrix for the CNN 						model)

train_mat.csv			training data labels only (in order,  						used by SVM_with_wavelets.m to get the 					labels for training)

training_set.csv			training data song ids with .wav 						extension (used by clean_data.py)

test_set.csv			testing data song ids with .wav 							extension (used by clean_date.py)

pred_test_50.csv			predictions for the SVM submitted to 						Kaggle			

pred_test_t.csv			predictions for the CNN submitted to 						Kaggle


For the figures of the three different representations of the data, run DataRepFigures.m

For the CNN model, run CNNmodel.py to train the model then run either predict.py for predictions on the training set or predict_test.py for predictions on the testing set 

For the SVM model, run SVM_with_wavelets.m to train the model and find the accuracy 

For the confusion matrices, run SVM_with_wavelets.m 


**Sorry for using both Python and MATLAB again, I do not know how to work with wavelets on python. 