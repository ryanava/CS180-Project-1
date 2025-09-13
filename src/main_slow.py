# CS180 Project 1

import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
import os

# name of the input file
imname = input("Enter the path to the image file (e.g., emir.tif or cathedral.jpg): ").strip()
imname = os.path.join("images", imname)
search_area = int(input("Enter search area size (e.g., 50 or 100): ").strip())
naive_search_algorithm = input("Enter the type of naive search (euclidean or ncc): ").strip()

# read in the image
im = cv2.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = im.astype(np.float32) / 255.0

# compute the height of each part B G R (just 1/3 of total)
height = int(np.floor(im.shape[0] / 3.0))

# separate color channels
b = im[:height, :, 0]
g = im[height: 2*height, :, 0]
r = im[2*height: 3*height, :, 0]


ag = utils.naive_align(naive_search_algorithm, g, b, search_area)
ar = utils.naive_align(naive_search_algorithm, r, b, search_area)
# create a color image
im_out_color = np.dstack([ar, ag, b])

im_out_color_int = (im_out_color * 255).astype(np.uint8)

im_out_grayscale = utils.color_to_grayscale(im_out_color_int)

im_out_color_black_cropped = utils.automatic_cropping_black(im_out_color_int, im_out_grayscale)

im_out_color_cropped = utils.automatic_cropping_color(im_out_color_black_cropped)

# save the image
cv2.imwrite('out_fname.jpg', cv2.cvtColor(im_out_color_cropped, cv2.COLOR_RGB2BGR))

# display the image
# display color and grayscale images side by side using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(im_out_color_cropped)
plt.title('Color Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(im_out_grayscale, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.show()