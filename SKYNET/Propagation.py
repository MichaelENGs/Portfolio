import numpy
from numba import jit
from source import getInitData as getInit
from source import initialize as init
import matplotlib.pyplot as mat


def propagate(dictionary_of_values):
    """
    forward propagation of the neural network, and back propagation of network.

    Arguments:
    dictionary_of_values    -- dictionary containing initialized values
        data_directory      -- file path of stored data vectors
        training_set_size   -- integer number of training data sets
        theta               -- vector of line weights
        func                -- string containing function choice default is sigmoid
        theta1              -- matrix of line weights from input to hidden layer
        theta2              -- vector of line weights from hidden layer to output

    returns:
    fp_output               -- dictionary of single output, and activations from the forward pass through the network
    gradients               -- dictionary of calculated gradients after backwards propagation
    """

    m = dictionary_of_values["training_set_size"]
    fp_output = {"layer0": 0}

    # forward propagation of neural network
    # open images as a matrix
    image_vector = getInit.getImageVect(dictionary_of_values["image_vector_path"], m)
    # image_vector = numpy.load("%s\Vertical_vector_image15.npy" % dictionary_of_values["image_vector_path"])
    # m = 1

    # Sigmoid activation forward pass
    print("Preforming forward propagation with sigmoid activation... \n")
    for x in range(dictionary_of_values["hidden_layer_size"]):
        theta = numpy.asmatrix(dictionary_of_values["theta"]["theta%d" % x])
        if x == 0:
            output = sigmoidActive(theta, image_vector)  # calculate from the input to the first hidden layer
        else:
            output = sigmoidActive(theta, output)
        output = numpy.insert(output, 0, [1], axis=0)
        fp_output["layer%d" % x] = output
    fp_output["layer%d" % x] = output[1, :]

    # back propagation
    print("Preforming backward propagation with sigmoid activation... \n")
    gradients = backProp(dictionary_of_values, fp_output, image_vector)

    return fp_output, gradients


@jit
def sigmoidActive(theta, layer):
    """
    Returns a vector of sigmoid activated nodes.

    Arguments:
    theta                       -- vector of line weights
    layer                       -- vector of values

    Returns:
    activated_output            -- vector of predicted outputs
    """

    hypothesis = numpy.dot(theta, layer)  # multiple feature linear regression
    activated_output = (1 / (1 + numpy.exp(-hypothesis)))  # Calculate the predicted output

    return activated_output


# @jit
def backProp(dictionary_of_values, fp_output, image_vector):
    """
    preforms backward propagation of neural network, with sigmoid activations.

    Arguments:
    dictionary_of_values     -- dictionary of initialized Network parameters
    fp_output                -- dictionary of outputs from forward propagation
    image_vector             -- matrix of input images

    returns:
    delta                    -- dictionary of gradients
    """

    m = dictionary_of_values["training_set_size"]
    # m = 1
    true_output = dictionary_of_values["image_labels"]
    # true_output = 1
    L_total = dictionary_of_values["hidden_layer_size"]
    afinal = fp_output["layer%d" % (L_total-1)]
    dzfinal = afinal - true_output  # Calculate error of predicted versus true output
    # Check number of layers
    if L_total == 1:
        delta = {"delta0": numpy.dot(dzfinal, image_vector.T)}
        return delta

    afinal1 = fp_output["layer%d" % (L_total - 2)]
    delta = {"delta%d" % (L_total-L_total): numpy.dot(dzfinal, afinal1.T)}  # Calculate gradient of error
    for x in range(-L_total+2, 1):
        theta = dictionary_of_values["theta"]["theta%d" % (-x+1)]
        acurrent = fp_output["layer%d" % -x]
        #  check for hidden layer
        try:
            aprev = fp_output["layer%d" % (-x - 1)]
        except:
            aprev = image_vector
        if x == -L_total+2:
            dz = numpy.multiply(numpy.dot(dzfinal.T, numpy.asmatrix(theta[:, 1:])),
                         numpy.multiply(afinal1[1:, :], numpy.subtract(1, afinal1[1:, :])).T)  # Calculate error of hidden layer
            delta["delta%d" % (L_total-1+x)] = numpy.dot(aprev, dz)
        else:
            dz = numpy.multiply(numpy.dot(dz, numpy.asmatrix(theta[:, 1:])),
                         numpy.multiply(acurrent[1:, :], numpy.subtract(1, acurrent[1:, :])).T)
            delta["delta%d" % (L_total-1+x)] = numpy.dot(aprev, dz)  # Calculate gradient of error

    for x in delta:
        delta[x] = delta[x]/m


    return delta


def gradientChecks(values):
    """
    Preforms gradient checking on the algorithm with the definition of a derivative.

    Arguments:
    values                    -- Dictionary of initialized parameters

    returns:
    checked_grad              -- vector of values of gradient checking

    """

    # Set default params
    epsilon = 10**-4
    theta = values["theta"]["theta0"]
    zeros_vect = numpy.zeros((1, theta.shape[1]))
    checked_grad = numpy.zeros((26, theta.shape[1]))
    for x in range(len(theta[0, :])):
        zeros_vect[0, x] = epsilon
        values["theta"]["theta0"] = theta+zeros_vect
        add_term, dk = propagate(values)
        values["theta"]["theta0"] = theta
        values["theta"]["theta0"] = theta - zeros_vect
        sub_term, dk = propagate(values)
        zeros_vect[0, x] = 0
        checked_grad[:, x] = (add_term["layer1"] - sub_term["layer1"])/(2*epsilon)

    return checked_grad


def forwardPropogate(values):
    """
    Preforms forward propagation.

    Arguments:
    values              -- Dictionary of initialized parameters

    returns:
    output              -- Dictionary of predicted values on data
    """

    m = values["training_set_size"]
    fp_output = {"layer0": 0}

    # forward propagation of neural network
    # open images as a matrix
    image_vector = getInit.getImageVect(values["image_vector_path"], m)

    # Sigmoid activation forward pass
    print("Preforming forward propagation with sigmoid activation... \n")
    for x in range(values["hidden_layer_size"]):
        theta = numpy.asmatrix(values["theta"]["theta%d" % x])
        if x == 0:
            output = sigmoidActive(theta, image_vector)  # calculate from the input to the first hidden layer
        else:
            output = sigmoidActive(theta, output)
        output = numpy.insert(output, 0, [1], axis=0)
        fp_output["layer%d" % x] = output
    fp_output["layer%d" % x] = output[1, :]

    return fp_output

if __name__ == "__main__":
    values = getInit.getInitData()
    # output, grads = forwardPropagate(values)
    # grad_check = gradientChecks(values)
    init.initializeTest(xval=False)
    getInit.getinitTest(values)
    # output, grads = Propagate(values)
    output = forwardPropogate(values)
