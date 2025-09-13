# CS180 Project 1

import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
import os

#please enter something valid
imname = input("Enter the path to the image file (e.g., emir.tif or cathedral.jpg): ").strip()
imname = os.path.join("images", imname)
search_area = int(input("Enter search area size (e.g., 50 or 100): ").strip())
naive_search_algorithm = input("Enter the type of naive search (euclidean or ncc): ").strip()
naive_or_pyramid = input("Enter the implementation you want (e.g., naive or pyramid): ")

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

# aligned green (it was the best for my images. i had an implementation for choosing which plate to align to but some just looked awful)
if (naive_or_pyramid.lower() == "naive"):
    ab, first_displacement = utils.naive_align(b, g, search_area, naive_search_algorithm)
    first_displacement = (first_displacement[1], first_displacement[0])
    ar, second_displacement = utils.naive_align(r, g, search_area, naive_search_algorithm)
    second_displacement = (second_displacement[1], second_displacement[0])
elif (naive_or_pyramid.lower() == "pyramid"):
    ab, first_displacement = utils.pyramid_align(b, g, search_area, naive_search_algorithm)
    first_displacement = (first_displacement[1], first_displacement[0])
    ar, second_displacement = utils.pyramid_align(r, g, search_area, naive_search_algorithm)
    second_displacement = (second_displacement[1], second_displacement[0])
im_out_color = np.dstack([ar, g, ab])

#convert to int
im_out_color = (im_out_color * 255).astype(np.uint8)

#convert to grayscale
im_out_grayscale = utils.color_to_grayscale(im_out_color)

#crop black borders
color_cropped_black_borders = utils.automatic_cropping_black(im_out_color, im_out_grayscale)

#crop color borders
im_out_color_cropped = utils.automatic_cropping_color(color_cropped_black_borders)

# display the image
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 3)
plt.imshow(im_out_color_cropped)
plt.title('Color Image Colors Cropped')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(color_cropped_black_borders)
plt.title('Color Image Black Cropped')
plt.axis('off')

plt.subplot(1, 3, 1)
plt.imshow(im_out_color, cmap='gray')
plt.title('Aligned Image Uncropped')
plt.figtext(0.5, 0.02, f"Displacements: {first_displacement}, {second_displacement}", wrap=True, horizontalalignment='center', fontsize=10)
plt.axis('off')

plt.show()