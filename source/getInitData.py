import numpy
from PIL import Image
from skimage.transform import resize
from skimage import color

def getInitData():
    """
    Returns the parameters initialized by initialize function

    Arguments:

    returns:
    files               -- list of images full path
    numel_max           -- long maximum number of elements
    File_sizes          -- list of number of elements in each image
    theta               -- list of line weights
    training_set_size   -- integer number of images
    alpha               -- float of step size for gradient descent
    function_choice     -- string defining activation function
    im_folderpath       -- path where image vectors are saved
    expected_output     -- list of labeled outputs
    """

    path = "H:\\repos\\test\\another_test"  # Sets path
    tempfile = open(path + "\\training_settings_initialized", "r")  # opens file where initialized data is stored
    all_vars = tempfile.read()  # reads all the data from the file
    tempfile.close()

    a, b, c, d, e, f, g, h, i, j, k, l = all_vars.split("\n\n")  # Unpacks individual outputs from data
    a = a.split("\n")
    c = c.split("\n")
    f = "".join(f.split("\n"))
    output = [int(x) for x in g.split("\n")]
    h = "".join(h.split("\n"))
    temp = []
    for x in i.split("(")[1:]:
        y = x.split("L")
        w = (int(y[0]))
        z = (int(y[1][1:]))
        temp.append([w,z])

    init_vals = {"file_list": a,
                 "numel_max": int(b),
                 "numel_min": int(j),
                 "min_matrix_index": int(k),
                 "numel_list": c,
                 "training_set_size": int(d),
                 "descent_step_size": float(e),
                 "image_vector_path": f,
                 "image_labels": numpy.asmatrix(output),
                 "hidden_layer_size": int(h),
                 "matrix_shape": temp,
                 "theta": getInitTheta(int(h))}

    return init_vals


def getInitTheta(hidden_layer_size):
    """
    Fetches the initialized theta values and returns a dictionary of single matrix of values

    Arguments:
    hidden_layer_size         -- integer representing the number of hidden layer nodes

    Returns:
    theta                     --dictionary of matrix of theta values
    """

    tempfile = "E:\\theta\\initialized_theta0.npy"
    theta = numpy.load(tempfile)
    theta = {"theta0": theta}

    if hidden_layer_size > 1:  # If there are hidden layers load the line weights for the layers into a matrix
        for x in range(1, hidden_layer_size):
            tempfile = ("E:\\theta\\initialized_theta%x.npy" % x)
            theta_temp = numpy.load(tempfile)
            theta["theta%d" %x] = theta_temp


    return theta


def getImageVect(directory, training_set_size):
    """
    Returns matrix of image vectors

    Arguments:
    directory               -- path where image vectors are stored
    training_set_size       -- integer representing number of imagaes to train

    returns:
    image_vector            -- rectangular array where rows represent fetures and columns represent training sets
    """

    image_vector = []
    for ind in range(0, training_set_size):
        if ind == 0:
            image_vector = numpy.load("%s\Vertical_vector_image%s.npy" % (directory, str(ind)))
        else:
            image_vector = numpy.append(image_vector, numpy.load("%s\Vertical_vector_image%s.npy" % (directory, str(ind))),
                     axis=1)  # Open vertical vector image and append array

    return image_vector


def getLearnedTheta(values):  # This function is out dated
    """
    fetches the trained theta values

    returns:
    theta           -- vector of learned theta values
    """

    m = values["hidden_layer_size"]

    for x in range(m):
        path = ("H:\\repos\\test\\another_test\\learned_theta%d.npy" % x)
        values["theta"]["theta%d" % x] = numpy.load(path)

    return


def getOriginalImages(x):
    """
    Returns an original image with scaled down resolution

    Arguments:
    x               -- File path of desired image

    returns:
    myArray         -- numpy array of grey scaled and resolution adjusted image
    """

    filename = x
    myFile = Image.open(filename)
    myArray = numpy.array(myFile)
    myArray = color.rgb2gray(myArray)
    myArray = resize(myArray, (myArray.shape[0]/5, myArray.shape[1]/5), anti_aliasing=True)

    return myArray


def getinitTest(values):
    """
    Updates the initialized parameters to match the test data set

    Arguments:
    values              -- dictionary of parameters

    returns:

    """

    path = "H:\\repos\\test\\another_test"  # Sets path
    tempfile = open(path + "\\test_settings_initialized", "r")  # opens file where initialized data is stored
    all_vars = tempfile.read()  # reads all the data from the file
    tempfile.close()

    a, b, c, d, e, f, g, h, i, j = all_vars.split("\n\n")
    a = a.split("\n")
    c = c.split("\n")
    e = "".join(e.split("\n"))
    output = []
    for x in f.split("\n"):
        try:
            output.append(int(x))
        except:
            output = "".join(x)
    try:
        output = numpy.asmatrix(output)
    except ValueError:
        pass
    temp = []
    for x in g.split("(")[1:]:
        y = x.split("L")
        w = (int(y[0]))
        z = (int(y[1][1:]))
        temp.append([w, z])

    values["file_list"] = a
    values["numel_max"] = int(b)
    values["numel_list"] = c
    values["training_set_size"] = int(d)
    values["image_vector_path"] = e
    values["image_labels"] = output

    return


if __name__ == "__main__":
    # getInitTheta(3)
    values = getInitData()
    getinitTest(values)
    # getImageVect(values["image_vector_path"],1)
    # e = values["training_set_size"]
    # g = values["image_vector_path"]
    # e = int(.01*e)
    # image_vect = getImageVect(g, e)
