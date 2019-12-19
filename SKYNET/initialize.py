import os
import numpy
from PIL import Image
from skimage import color
from skimage.transform import resize
import data as data
import gdal


def initializeNN():
    """
    This function calculates initial parameters for the network and saves them to a file.

    Arguments:

    returns:

    file I/O:
    The following is saved to a txt file
        files               -- vector of image paths to be analyzed
        numel_max           -- integer value representing maximum number of elements
        numel_min           -- integer value representing minimum number of elements
        min_matrix          -- integer value representing index of minimum matrix shape
        matrix_shape        -- list of shapes of matrixes for input images
        file_sizes          -- vector containing number of element values per image
        theta               -- line weights initialized with a gaussian distribution about 0
        training_set_sizes  -- integer representing number of data sets
        alpha               -- integer representing the value of the step size
        function_choice     -- String containing choice of activation function
        im_folderpath       -- string containg path of relevent directories
        expected_output     -- array containing the image labels in binary bool values
    """

    # load image paths to vector
    im_folderpath = "H:\\repos\\SKYNET\\skynet\\data\\small_training_images"
    files = [(im_folderpath + '\\' + name) for name in os.listdir(im_folderpath) if
             name != "image_labels.txt"]  # Store full file path with images
    training_set_size = 0  # Initialize training set size
    file_sizes = []  # Initializing file_size vector
    matrix_shape = []  # Initializing matrix shape vector

    # load training labels and sort to match order of files
    tempfile = open(im_folderpath + "\\image_labels.txt", 'r')
    expected_output = tempfile.read()
    expected_output = numpy.array(expected_output.split())
    tempfile.close()

    # Open images as greyscale and save array sizes to file_size
    for filename in files:
        myFile = Image.open(filename)
        myArray = numpy.array(myFile)
        myArray = color.rgb2gray(myArray)
        myArray = resize(myArray, (myArray.shape[0] / 5, myArray.shape[1] / 5), anti_aliasing=True)
        matrix_shape.append(myArray.shape)
        file_sizes.append(myArray.shape[0] * myArray.shape[1])
        training_set_size += 1
    numel_min = min(file_sizes)  # Return minimum array size
    min_matrix = file_sizes.index(numel_min)  # Return index of minimum matrix
    numel_max = max(file_sizes)  # Return maximum array size

    try:
        hidden_layer_size = int(raw_input(
            "Choose number of hidden layers"))  # Prompts user to choose size of the hidden layer
    except ValueError:
        print "hidden layer size set to default of 1"
        hidden_layer_size = 1

    initTheta(hidden_layer_size, numel_min)

    try:
        alpha = float(raw_input("choose gradient descent step size"))  # Prompts user to initialize step size
    except:
        print "alpha set to default of 0.01"
        alpha = 0.01

    im_folderpath = "H:\\repos\\SKYNET\\skynet\\data\\small_image_vectors_training"

    # Save parameters to file
    outputs = [files, numel_max, file_sizes, training_set_size, alpha, im_folderpath, expected_output,
               hidden_layer_size, matrix_shape, numel_min, min_matrix]  # Saves outputs as list
    path = "H:\\repos\\SKYNET\\skynet\\data"  # Sets file path
    tempfile = open(path + "\\training_settings_initialized", "w+")  # opens file in path
    for x in outputs:
        try:
            for y in x:
                tempfile.write(str(y) + "\n")  # writes output to file
        except:
            tempfile.write(str(x) + "\n")  # Catches when the output is not iterable and writes to file
        tempfile.write("\n")  # Separates outputs with newline
    tempfile.close()

    return


def initializeTest(xval=True):
    """
    Initialized parameters of test set or cross validation set. Training of Neural net must be completed before this
    function can be fetched.

    Arguments:
    xval            -- determines weather setting are initialized for cross validation or testing

    returns:

    File I/O:
    The function will always return the same file independent of xval argument
    """

    if xval == True:
        # load image paths to vector
        im_folderpath = "E:\\cross_validation"
        # load training labels and sort to match order of files
        tempfile = open(im_folderpath + "\\image_labels.txt", 'r')
        expected_output = tempfile.read()
        expected_output = numpy.array(expected_output.split())
        tempfile.close()
    else:
        im_folderpath = "E:\\small_test_images"
        expected_output = "Human analysis needed"

    files = [(im_folderpath + '\\' + name) for name in os.listdir(im_folderpath) if
             name != "image_labels.txt"]  # Store full file path with images
    training_set_size = 0  # Initialize training set size
    file_sizes = []  # Initializing file_size vector
    matrix_shape = []  # Initializing matrix shape vector

    # Open images as greyscale and save array sizes to file_size
    for filename in files:
        myFile = Image.open(filename)
        myArray = numpy.array(myFile)
        myArray = color.rgb2gray(myArray)
        # vvvvvvvvvvvvvvvvv This step might not be necessary vvvvvvvvvvvvvvvv
        # ------------------VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV-----------------
        #                                 vvvv
        myArray = resize(myArray, (myArray.shape[0] / 5, myArray.shape[1] / 5), anti_aliasing=True)
        matrix_shape.append(myArray.shape)
        file_sizes.append(myArray.shape[0] * myArray.shape[1])
        training_set_size += 1
    numel_min = min(file_sizes)  # Return minimum array size
    min_matrix = file_sizes.index(numel_min)  # Return index of minimum matrix
    numel_max = max(file_sizes)  # Return maximum array size

    #  Set directory to appropriate folder
    im_folderpath = "E:\\xval_image_vectors"
    if xval == False:
        im_folderpath = "E:\\small_image_vectors_test"

    # Save parameters to file
    outputs = [files, numel_max, file_sizes, training_set_size, im_folderpath, expected_output,
               matrix_shape, numel_min, min_matrix]  # Saves outputs as list
    path = "H:\\repos\\test\\another_test"  # Sets file path
    tempfile = open(path + "\\test_settings_initialized", "w+")  # opens file in path
    for x in outputs:
        try:
            for y in x:
                tempfile.write(str(y) + "\n")  # writes output to file
        except:
            tempfile.write(str(x) + "\n")  # Catches when the output is not iterable and writes to file
        tempfile.write("\n")  # Separates outputs with newline
    tempfile.close()

    return


def initTheta(num_layers=1, numel_max=4328000):
    """
    Initializes theta values for all layers of the network will save each theta set to a unique file for the layer. This
    function will also prompt the user to enter the number of nodes per layer.

    Arguments:
    numel_max           -- long value containing the maximum number of elements in a training set image
    num_layer           -- number of layers to initialize

    returns:

    File I/O:
    save vector of values to a numpy file. Outputs number of files equivalent to num_layers
    """

    try:
        hidden_layer_node = int(raw_input("Insert number of nodes from the input layer"))
    except:
        print "number of hidden layers set to default of 1"
        hidden_layer_node = 1

    # Initialize wights from input layer to hidden layer
    # Will generate perceptron if hidden layer is set to 1
    # This will generate values with a gaussian distribution about 0 with a std of 0.01 and a shape of (hidden layer nods, numel_min+1)
    theta = numpy.asmatrix(
        numpy.random.normal(0, .01, (hidden_layer_node, (numel_max + 1))))  # Actually Numel_min
    theta[:, 0] = 1  # initialize bias weight
    path = "E:\\theta\\initialized_theta0"
    tempfile = open(path, "w+")
    numpy.save(path, theta)
    tempfile.close()

    #  Initialize the remaining layers
    if num_layers > 1:
        num_nodes = numpy.matrix(numpy.zeros((1, num_layers)))
        num_nodes[0, 0] = hidden_layer_node
        num_nodes[0, -1] = 1  # Set the output layer to one classification
    for x in range(1, num_layers):
        # Finds number of nodes for each layer
        if num_nodes[0, x] == 0:
            try:
                num_nodes[0, x] = int(raw_input("Insert number of hidden layer nodes for layer %d" % (x + 1)))
            except:
                print "number of hidden layers set to default of 1"
                num_nodes[0, x] = 1

        # Initialize weights from hidden layer to output layer
        path = "E:\\theta\\initialized_theta%d" % (x)
        if x != num_layers:
            theta = numpy.asmatrix(
                numpy.random.normal(0, .01, (int(num_nodes[0, x]), int(num_nodes[0, x - 1] + 1))))
            tempfile = open(path, "w+")
            theta[:, 0] = 1  # initialize bias weight
            theta_sum = numpy.sum(theta[0, 1:])
            numpy.save(path, theta)
        else:
            theta = numpy.asmatrix(
                numpy.random.normal(0, .01, (int(num_nodes[0, x - 1] + 1), 1)))
            theta[0, 0] = 1
            numpy.save(path, theta[:, 0])
        tempfile.close()



def printFiles():
    """
    Displays the data files as they are read into Skynet

    returns:
    n/a
    """
    im_folderpath = "H:\\repos\\SKYNET\\skynet\\data\\small_training_images"
    files = [(im_folderpath + '\\' + name) for name in os.listdir(im_folderpath) if
             name != "image_labels.txt"]
    print(files)


if __name__ == "__main__":
#     # initTheta()
     initializeNN()
# # image_vect = getImageVect(h, e)
#      initializeTest(xval=False)
