import initialize
import getInitData
import data_preprocess
import Propagation
import trainNN
import predict
import numpy

"""
Image classification bot, Provide bot with data sets to train, cross validate, and test classifications based on the data
provided. The bot will produce classification predictions based on a one vs. all classification.

By: Michael Salzarulo

For first time use the recommended initialization has been set as the default parameters: 
"""

print("Initializing Neural Network parameters ... \n")
initialize.initializeNN()
print("Retrieving initial parameters ... \n")
values = getInitData.getInitData()
print("Converting images to %dx1 matrix ..." % values["numel_max"])
data_preprocess.readInGreyscale(values)
print("Calculating initial prediction... \n")
output1, dk = Propagation.propagate(values)
predict1, threshold1 = predict.predict(output1, values)
error1 = predict.analyze(predict1, values, threshold1)
raw_input("READY TO TRAIN NETWORK \nThis might take a few moments\npress enter to continue ... \n")
trainNN.gradientDescent(values)
getInitData.getLearnedTheta(values)
raw_input("Press enter to continue ... \n")
output2, dk = Propagation.propagate(values)
predict2, threshold2 = predict.predict(output2, values)
error2 = predict.analyze(predict2, values, threshold2)
predict.visualizeWeights(values, dk)
print("\nBefore learning: ")
print("%2.12f%% accuracy\n with the predicted output of: " % (error1*100))
print(predict1)
print("\nAfter learning: ")
print("%2.12f%% accuracy\n with the predicted output of: " % (error2*100))
print(predict2)

xval = raw_input("\n\nRun Cross validation? [Y]")
if xval == "Y" or xval == "y":
    initialize.initializeTest()
    getInitData.getinitTest(values)
    data_preprocess.readInGreyscale(values)
    output3, dk = Propagation.propagate(values)
    predict3, threshold3 = predict.predict(output3, values)
    error3 = predict.analyze(predict3, values, threshold3)
    predict.visualizeWeights(values, dk)

test = raw_input("\n\nRun a test set? [T]")
if test == "T" or test == "t":
    initialize.initializeTest(xval=False)
    getInitData.getinitTest(values)
    data_preprocess.readInGreyscale(values)
    output4, dk = Propagation.propagate(values)
    predict4, threshold4 = predict.predict(output4, values)
    print(predict4)



