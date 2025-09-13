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
naive_or_pyramid = input("Enter the implementation you want (e.g., naive or pyramid): ").strip()
alignment_color = input("Enter the plate color you want to align your 3 plates to (e.g., red or green or blue: ").strip()

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

#order of input tuple doesnt matter
if (alignment_color.lower() =="green"):
    input = (b, r)
    anchor = g
elif (alignment_color.lower() =="red"):
    input = (b, g)
    anchor = r
elif (alignment_color.lower() =="blue"):
    input = (g, r)
    anchor = b

# aligned green
if (naive_or_pyramid.lower() == "naive"):
    first_aligned = utils.naive_align(input[0], anchor, search_area, naive_search_algorithm)
    second_aligned = utils.naive_align(input[1], anchor, search_area, naive_search_algorithm)
elif (naive_or_pyramid.lower() == "pyramid"):
    first_aligned = utils.pyramid_align(input[0], anchor, search_area, naive_search_algorithm)
    second_aligned = utils.pyramid_align(input[1], anchor, search_area, naive_search_algorithm)

#no real way to make this look sensical
if (alignment_color.lower() =="green"):
    im_out_color = np.dstack([second_aligned, anchor, first_aligned])
elif (alignment_color.lower() =="red"):
    im_out_color = np.dstack([anchor, second_aligned, first_aligned])
elif (alignment_color.lower() =="blue"):
    im_out_color = np.dstack([second_aligned, first_aligned, anchor])



im_out_color = (im_out_color * 255).astype(np.uint8)

im_out_grayscale = utils.color_to_grayscale(im_out_color)

#crops grayscale bars only
color_cropped_black_borders = utils.automatic_cropping_black(im_out_color, im_out_grayscale)

im_out_color_cropped = utils.automatic_cropping_color(color_cropped_black_borders)

# save the image
cv2.imwrite('out_fname.jpg', cv2.cvtColor(im_out_color_cropped, cv2.COLOR_RGB2BGR))

# display the image
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 4)
plt.imshow(im_out_color_cropped)
plt.title('Color Image no color border')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(color_cropped_black_borders)
plt.title('Color Image no black border')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(im_out_grayscale, cmap='gray')
plt.title('Aligned Grayscale Image')
plt.axis('off')

plt.subplot(1, 4, 1)
plt.imshow(im, cmap='gray')
plt.title('Red, Blue, Green Channels')
plt.axis('off')

plt.show()