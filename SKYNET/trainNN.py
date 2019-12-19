from source import Propagation as fp
import numpy
import matplotlib.pyplot as mat
import getInitData as getInit
import Propagation as fp
from numba import jit


def gradientDescent(dictionary_of_values):
    """
    trains the network using gradient descent

    Arguments:
    dictionary of values
        m               -- integer number of training images
        output          -- vector of expected output
        hypothesis      -- vector of hypothesis values for each node
        theta           -- vector of line weights
        input_vector    -- input values
        function_choice -- string containing choice of function default is sigmoid
        alpha           -- integer value representing step size defaults at 0.01

    returns:
        theta           -- vector of updated line weights
    """

    print("Network training in progress ...\n")

    # unpack initialized values
    # m = dictionary_of_values["training_set_size"]
    # true_output = dictionary_of_values["image_labels"]
    m = 1
    true_output = 1
    cost_primte = []
    try:
        iterations = int(raw_input("Insert number of iterations to train on"))
    except:
        print("The default number of iterations set to 1000")
        iterations = 1000

    L_tot = dictionary_of_values["hidden_layer_size"]
    output_history = []

    # Preform gradient descent
    for num_of_i in range(0, iterations):
        # propagate through the network
        fp_output, gradients = fp.propagate(dictionary_of_values)
        # update line weights
        dictionary_of_values["theta"] = updateWeights(dictionary_of_values, gradients)
        # calculate cost function
        afinal = fp_output["layer%d" % (L_tot-1)]
        costfunc = numpy.sum(numpy.sum(numpy.multiply(numpy.log(afinal), -true_output).T - (
            numpy.multiply(numpy.log(numpy.subtract(1, afinal)), (numpy.subtract(1, true_output)))).T))
        costfunc = costfunc / m
        output_history.append(costfunc)  # Record output of cost function

        assert (numpy.isnan(costfunc) == 0)  # Check to make sure cost function returns valid value

    print("Displaying network learning ... \nClose window to continue ...\n")

    mat.plot(range(0, num_of_i + 1), output_history)
    plot_title = ("Gradient Descent with %d iterations and alpha=%.04f using %s activation" % (
        num_of_i + 1, dictionary_of_values["descent_step_size"], "sigmoid"))
    mat.suptitle(plot_title)
    mat.xlabel("iterations")
    mat.ylabel("Cost value")
    mat.show()

    print("Network training complete!\n")
    print("Saving line weights...\n")
    saveTheta(dictionary_of_values)

    return


def updateWeights(values, grads):
    """
    Preforms the update rule on line weights.

    Arguments:
    values              -- dictionary of existing network parameters
    grads               -- dictionary of line gradients

    returns:
    values              -- dictionary of updated network parameters
    """

    alpha = values["descent_step_size"]
    L_tot = values["hidden_layer_size"]

    for x in range(-L_tot + 1, 1):
        if x == 0:
            values["theta"]["theta%d" % (x + L_tot - 1)] -= numpy.multiply(alpha, grads["delta%d" % (-x)])
        else:
            values["theta"]["theta%d" % (x + L_tot - 1)] -= numpy.multiply(alpha, grads["delta%d" % (-x)]).T

    return values["theta"]


def saveTheta(values):
    """
    Saves line weights to a file

    Arguments:
    values          -- dictionary of initialized network parameters

    returns:

    File I/O:
    creates or truncates a file named learned_theta. There will be an equal number of learned theta files as there are
    initialized theta values.
    """

    for x in values["theta"]:
        theta = values["theta"][x]
        path = "H:\\repos\\test\\another_test\\learned_%s" % x
        tempfile = open(path, "w+")
        numpy.save(path, theta)
        tempfile.close()


if __name__ == "__main__":
    values = getInit.getInitData()
    # getInit.getLearnedTheta(values)
    gradientDescent(values)
    # saveTheta(values)  # Saves the learned theta vector
