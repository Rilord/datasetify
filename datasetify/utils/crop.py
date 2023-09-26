def crop(img, data):
    """Crop image with square
    Keyword arguments:
    img -- cv2 image
    data -- crop data: list[0, 1, 2, left_x, bottom_y, right_x, top_y]
    """
    (h, w) = img.shape[:2]
    top = max(int(h * float(data[4])), 0)
    bottom = min(int(h * float(data[5])), h)
    left = max(int(h * float(data[3])), 0)
    right = min(int(h * float(data[6])), w)

    return img[top:bottom, left:right]

def crop_square(img, data):
    """Crop image with square
    Keyword arguments:
    img -- cv2 image
    data -- crop data: list[0, 1, 2, left_x, bottom_y, right_x, top_y]
    """
    (h, w) = img.shape[:2]
    top = max(int(h * float(data[4])), 0)
    bottom = min(int(h * float(data[5])), h)
    left = max(int(h * float(data[3])), 0)
    right = min(int(h * float(data[6])), w)

    raw_w = right - left
    raw_h = bottom - top

    if raw_w > raw_h:
        diff = raw_w - raw_h
        if top - (diff / 2) < 0:
            diff = diff - top
            top2 = 0
            bottom2 = bottom + diff
        elif (diff % 2) == 0:  # even
            diff = int(diff / 2)
            top2 = top - diff
            bottom2 = bottom + diff
        else:  # odd
            diff = int(diff / 2)
            top2 = top - diff
            bottom2 = bottom + diff + 1

        cropped = img[top2:bottom2, left:right]

    elif raw_h > raw_w:
        diff = raw_h - raw_w
        if (left - (diff / 2)) < 0:
            diff = diff - left
            left2 = 0
            right2 = right + diff
        elif (diff % 2) == 0:  # even
            diff = int(diff / 2)
            left2 = left - diff
            right2 = right + diff
        else:  # odd
            diff = int(diff / 2)
            left2 = left - diff
            right2 = right + diff + 1

        cropped = img[top:bottom, left2:right2]
    else:
        cropped = img[top:bottom, left:right]

    return cropped
