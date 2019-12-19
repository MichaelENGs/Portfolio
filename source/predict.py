import numpy
from data_preprocess import readInGreyscale as rig
import initialize as InitNew
import getInitData as Init
from Propagation import propagate as fp
from Propagation import forwardPropogate as f
import matplotlib.pyplot as mat
from skimage.transform import resize
from PIL import Image


def predict(output, values):
    """
    Predicts the binary output based on the training of the network.

    Arguments:
    output              -- output vector of forward propagation

    returns:
    Y                   -- vector containing bool predictions of outputs
    threshold           -- Vector of calculated threshold outputs
    """

    L_tot = values["hidden_layer_size"]

    print("Predicting classifications... \n")
    threshold = numpy.average(output["layer%d" % (L_tot-1)])/output["layer%d" % (L_tot-1)]
    print("the threshold value is %.12f" % numpy.average(output["layer%d" % (L_tot-1)]))
    Y = [int(x < 1) for x in numpy.nditer(threshold)]  # Create output matrix of predicted values

    if values["image_labels"] == "Human analysis needed":
        print(Y)

    return Y, threshold


def analyze(predicted_output, values, threshold):
    """
    Calculates analysis of network values.

    Arguments:
    predicted_output    -- vector of predicted values
    true_ouput          -- vector of actual output values
    training_data_sets  -- number of data sets propagated through the network

    returns:
    error               -- float percentage of calculated performance error
    """

    # Overall accuracy of the network
    error = float(sum(int(predicted_output[x] == values["image_labels"][0, x]) for x in range(0, len(predicted_output)))) / len(
        predicted_output)
    print("The network is %3.2f%% accurate based on %d sets of data" % (error*100,
         len(predicted_output)))  # Calculates and prints the accuracy of the network's prediction
    print("with an output of:")
    print(predicted_output)
    print("")
    print("with expected:")
    print(values["image_labels"])
    print("")

    # Confidence calculation
    count = 0
    confidence = numpy.multiply(numpy.abs(numpy.subtract(1, threshold)), 100/numpy.max(numpy.abs(numpy.subtract(1, threshold))))
    for x in numpy.nditer(confidence):
        print("there is %2.2f%% confidence for prediction %d" % (x, predicted_output[count]))
        count += 1
    return error


def visualizeWeights(values, gradients):
    """
    Display the mapped line wieghts as well as the gradient

    Arguments:
    values              -- dictionary of initialized network parameters

    returns:
    n/a
    """

    L_tot = values["hidden_layer_size"] - 1

    rows, cols = (values["matrix_shape"][values["min_matrix_index"]])
    theta = values["theta"]["theta0"]
    grads = gradients["delta0"]
    theta = numpy.reshape(theta[0, 1:], (rows, cols))
    grads = numpy.resize(gradients["delta%d" % L_tot][1:, 0], (rows, cols))
    if L_tot == 0:
        grads = numpy.resize(gradients["delta0"][0, 1:], (rows, cols))
    ax1 = mat.imshow(theta, cmap="gray")
    mat.title("Theta")
    mat.show(block=False)
    mat.figure(3)
    ax2 = mat.imshow(grads, cmap="gray")
    mat.title("Gradients")
    mat.show()
    mat.figure(4)



if __name__ == "__main__":

    # Preform prediction and analysis on initialized parameters
    values = Init.getInitData()
    output, dk = fp(values)
    predicted_output, threshold = predict(output, values)
    error = analyze(predicted_output, values, threshold)

    # Preform prediction and analysis on training data
    Init.getLearnedTheta(values)
    output, dk = fp(values)
    predicted_output, threshold = predict(output, values)
    error = analyze(predicted_output, values, threshold)
    visualizeWeights(values, dk)

    # Preform prediction and analysis on xval data
    InitNew.initializeTest()
    Init.getinitTest(values)
    rig(values)
    output, dk = fp(values)
    predicted_output, threshold = predict(output, values)
    error = analyze(predicted_output, values, threshold)
    visualizeWeights(values, dk)

    # Preform prediction on test data
    InitNew.initializeTest(xval=False)
    Init.getinitTest(values)
    rig(values)
    output = f(values)
    predicted_output = predict(output, values)
