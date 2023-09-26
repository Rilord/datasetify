import cv2
import os

M, N = 512, 512

images_dir = 'images'
out_dir = 'splits'

for filename in os.listdir(images_dir):
    im = cv2.imread(os.path.join(images_dir,filename))
    if im is not None:
        img_id = 0
        tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
        for tile in tiles:
            basename = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(out_dir, f'{basename}_{img_id}.png'), tile)
            img_id += 1

