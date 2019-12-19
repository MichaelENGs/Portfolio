import numpy
from source import getInitData as getInit
from PIL import Image
from skimage.transform import resize
from skimage import color
import logging


def readInGreyscale(values=getInit.getInitData()):
    """
    Reads training images in with grey scaling, resolution scaling and saves the vertical pixel vector to a file.
    The resulting vector is nx1 where n is the number of elements of the smallest image in the set. All images should
    be scaled to training set parameters.

    Arguments:
    values              -- Dictionary of initialized parameters

    returns:

    File I/0:
    creates or truncates existing vertical vector images
    """

    # unpack values
    files = values["file_list"]
    file_sizes = values["numel_list"]
    working_directory = values["image_vector_path"]
    min_matrix_dim = values["matrix_shape"][values["min_matrix_index"]]
    # Values specific to training set
    numel_min = values["numel_min"]
    numel_max = values["numel_max"]

    logging.basicConfig(level=logging.DEBUG)

    # Open file in grey scale and create a horizontal vector then pad image
    i = 0  # integer index
    numel_max = int(numel_max)
    for filename in files:
        size_of_file = int(file_sizes[i])
        myFile = Image.open(filename)
        myArray = color.rgb2grey(numpy.array(myFile))
        path = "%s\\vertical_vector_image%s" % (working_directory, str(i))
        Image_vector = open(path, 'w+')  # Open and truncate file to save vertical array
        cols, rows = myArray.shape  # find shape of current image

        if numel_min + size_of_file == (2 * numel_min):  # If max array size is equal to current array size
            myArray = resize(myArray, (myArray.shape[0] / 5, myArray.shape[1] / 5), anti_aliasing=True)  # Minimum scaling
        elif numel_max < size_of_file:
            raise AttributeError(
                "The maximumm file size calculated (%d) is less than the current file size (%d). Reinitialize data ..."
                    % (numel_min, size_of_file))
        if numel_min > size_of_file:
            myArray = resize(myArray, (min_matrix_dim[0] / float(cols) * cols, min_matrix_dim[1] / float(rows) * rows),
                             anti_aliasing=True)
        else:  # Default number of elements is not equal to max number of elements
            #  resize image to same shape as smallest image
            myArray = resize(myArray, (cols / (float(cols) / float(min_matrix_dim[0])),
                                       (rows / (float(rows) / float(min_matrix_dim[1])))), anti_aliasing=True)

        myArray = myArray.reshape(numel_min, 1)
        myArray = numpy.insert(myArray, [0], [1], axis=0)  # Insert biased feature
        numpy.save(path, myArray)  # Save vector to file

        Image_vector.close()
        i += 1

    # logging.debug(myArray)
    # logging.warn(myArray.shape)


if __name__ == "__main__":
    values = getInit.getInitData()
    # readInGreyscale()  # Calls the function rigc

    # Calls function on test data
    # getInit.getinitTest(values)
    readInGreyscale(values)
