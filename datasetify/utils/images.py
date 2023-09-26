import cv2
import os
import imutils

from datasetify.utils.fs import extract_basename


def image_resize(image, width=None, height=None, max=None, inter=cv2.INTER_CUBIC):
    '''Resize image to given width and height
    Keyword arguments:
    image -- cv2 image
    width -- image width
    height -- image height
    max —- image dimensions limit. If max > w/h then borders are drawn
    '''
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    if max is not None:
        if w > h:
            # produce
            r = max / float(w)
            dim = (max, int(h * r))
        elif h > w:
            r = max / float(h)
            dim = (int(w * r), max)
        else:
            dim = (max, max)

    else:
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def image_scale(image, scalar=1.0, inter=cv2.INTER_CUBIC):
    '''scale image with scalar coef
    Keyword arguments:
    image -- cv2 image
    scalar -- resize coef
    '''
    (h, w) = image.shape[:2]
    dim = (int(w * scalar), int(h * scalar))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def rotate(image, save=False, filename="image", path=""):
    '''rotate image on angle 90 and save to path (optionally)
    Keyword arguments:
    image -- cv2 image
    save -- save file to directory?
    filename — image base filename
    path — save dir
    '''
    r = image.copy()
    r = imutils.rotate_bound(r, 90)
    r_file = os.path.splitext(filename)[0] + "-rot90.png"
    if save:
        cv2.imwrite(os.path.join(path, r_file), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return r


def flip_image(image, save=False, filename="image", path=""):
    '''flip image and save to path (optionally)
    Keyword arguments:
    image -- cv2 image
    save -- save file to directory?
    filename — image base filename
    path — save dir
    '''
    flip_img = cv2.flip(image, 1)
    flip_file = extract_basename(filename) + "-flipped.png"
    if save:
        cv2.imwrite(
            os.path.join(path, flip_file), flip_img, [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )

    return flip_img
