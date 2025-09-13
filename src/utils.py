
import numpy as np
from skimage.transform import rescale

def naive_align(input, anchor, search_area, type_of_alignment):
    #dont accept non strings
    if not isinstance(type_of_alignment, str):
        return TypeError
    if search_area > max(input.shape):
        return IndexError
    temp_crop_input = pre_align_crop(input, .1)
    temp_crop_anchor = pre_align_crop(anchor, .1)
    if type_of_alignment.lower() == "euclidean":
        best_shift = euclidean_shift(temp_crop_input, temp_crop_anchor, search_area)
    elif type_of_alignment.lower() == "ncc":
        best_shift = ncc_shift(temp_crop_input, temp_crop_anchor,search_area)
    return np.roll(input, shift=best_shift, axis=(0, 1)), best_shift
    
    

def pre_align_crop(img, margin=0.1):
    #margin is a percentage so i dont magic number it. crop before alignment to avoid borders influencing ncc/euclidean
    y, x = img.shape
    top = int(y * margin)
    bottom = int(y * (1 - margin))
    left = int(x * margin)
    right = int(x * (1 - margin))
    return img[top:bottom, left:right]


def pyramid_align(input, anchor, smallest_img_size, naive_search_algorithm):
    #this way we can pass it in cropped, but we dont accidentally roll the cropped ones (they are only temporary)
    temp_crop_input = pre_align_crop(input, .1) #10% each side crop
    temp_crop_anchor = pre_align_crop(anchor, .1)
    #align does the aligning, shift finds the shift to shift it by
    best_shift = pyramid_shift(temp_crop_input, temp_crop_anchor, smallest_img_size, naive_search_algorithm)
    return np.roll(input, shift=best_shift, axis=(0, 1)), best_shift

def pyramid_shift(input, anchor_array, smallest_img_size, naive_search_algorithm="ncc"):
    #hardcoded
    scale = 4

    # base step
    if min(input.shape) < smallest_img_size:
        if naive_search_algorithm.lower() == "euclidean":
            return euclidean_shift(input, anchor_array, smallest_img_size // scale)
        elif naive_search_algorithm.lower() == "ncc":
            return ncc_shift(input, anchor_array, smallest_img_size // scale)
        else:
            return ValueError
        
    #rescale making it smaller before each recurssive call
    input_rescaled = rescale(input, 1/scale, anti_aliasing=True)
    anchor_rescaled = rescale(anchor_array, 1/scale, anti_aliasing=True)

    # recursive call (returns a displacement/shift vector) -- the thing i need to print
    smaller_shift = pyramid_shift(input_rescaled, anchor_rescaled, smallest_img_size, naive_search_algorithm="ncc")

    # scale up the shift for the next recurssion layer (after we make it over the hump)
    shift_vector_guess = (scale * smaller_shift[0], scale * smaller_shift[1])
    
    window_guess = search_radius(input.shape)
    #max(window_scale // scale, 5)  
    if naive_search_algorithm.lower() == "euclidean":
        better_shift = euclidean_shift(input, anchor_array, window_guess, shift_vector_guess)
    elif naive_search_algorithm.lower() == "ncc":
        better_shift = ncc_shift(input, anchor_array, window_guess, shift_vector_guess)
    else:
         return ValueError
    return better_shift

#this function was my best solution to gradually reducing the search_area at higher and higher
#resolution images, without being too large or too small, and without it reacthing poorly to images of varying sizes
def search_radius(img_shape, A=5000, min_size=5, max_size=50):
        img_size = max(img_shape)
        r = int(round(A / img_size))
        #this way we dont search 1x1 at the top but also dont search 100x100 on the next coarsest level
        return max(min_size, min(max_size, r))

#returns the best shift
def euclidean_shift(input, anchor, window, shift_guess=(0,0)):
    best_score = np.inf
    best_shift = shift_guess

    for dy in range(shift_guess[0] - window, shift_guess[0] + window + 1):
        for dx in range(shift_guess[1] - window, shift_guess[1] + window + 1):
            shifted_input = np.roll(input, shift=(dy, dx), axis=(0, 1))
            score = np.sum((shifted_input - anchor) ** 2)
            if score < best_score:
                best_score = score
                best_shift = (dy, dx)
    
    return best_shift

#returns the best shift
def ncc_shift(input, anchor_array, window, shift_guess=(0,0)):
    best_score = -1
    best_shift = shift_guess

    vec_anchor = anchor_array.flatten()
    vec_anchor_norm = vec_anchor / np.linalg.norm(vec_anchor)

    for dy in range(shift_guess[0] - window, shift_guess[0] + window + 1):
        for dx in range(shift_guess[1] - window, shift_guess[1] + window + 1):
            shifted_input = np.roll(input, shift=(dy, dx), axis=(0, 1)) # dy vertical dx horizontal
            
            #turn into vectors
            vec_input = shifted_input.flatten()
            #does euclidean norm
            vec_input_norm = vec_input / np.linalg.norm(vec_input)
            #compute dot product
            curr_score = np.dot(vec_input_norm, vec_anchor_norm)
            if curr_score > best_score:
                best_score = curr_score
                best_shift = (dy, dx)

    return best_shift


def color_to_grayscale(color_image):
    blue, green, red = color_image[:,:,0], color_image[:,:,1], color_image[:,:,2]
    grayscale_image = ((0.299 * red) + (0.587 * green) + (0.114 * blue)).astype(np.uint8)
    return grayscale_image

def automatic_cropping_black(original_image, manipulated_image):
    ### FIRST CROP THE BLACK BORDERS
    #both these values are hardcoded
    #the difference between means in rows/columns that is significant (from my findings) 
    # to indicate a shift from black line to non black line
    threshold = 4
    #so that the cropping doesnt crop too far into the image 
    edges = 16
    # top check(rows)
    top_candidates = []
    for dy in range((manipulated_image.shape[0] // edges) - 1):
        #mean because its rows, no longer pixels for better accuracy
        curr_row_intensity = np.mean(manipulated_image[dy, :])
        next_row_intensity = np.mean(manipulated_image[dy + 1, :])
        if (next_row_intensity - curr_row_intensity) > threshold:
            #only checks positive black to white changes
            top_candidates.append(dy + 1)
    #only returns 
    top = top_candidates[-1] if top_candidates else 0

    # bottom check(rows)
    bottom_candidates = []
    for dy in range((manipulated_image.shape[0] // edges) - 1):
        curr_row_intensity = np.mean(manipulated_image[-dy - 1, :])
        next_row_intensity = np.mean(manipulated_image[-dy - 2, :])
        if (next_row_intensity - curr_row_intensity) > threshold:
            bottom_candidates.append(manipulated_image.shape[0] - dy - 1)
    bottom = bottom_candidates[-1] if bottom_candidates else manipulated_image.shape[0] - 1

    # left check (columns)
    left_candidates = []
    for dx in range((manipulated_image.shape[1] // edges) - 1):
        curr_col_intensity = np.mean(manipulated_image[:, dx])
        next_col_intensity = np.mean(manipulated_image[:, dx + 1])
        if (next_col_intensity - curr_col_intensity) > threshold:
            left_candidates.append(dx + 1)
    left = left_candidates[-1] if left_candidates else 0

    # right check(columns)
    right_candidates = []
    for dx in range((manipulated_image.shape[1] // edges) - 1):
        curr_col_intensity = np.mean(manipulated_image[:, -dx - 1])
        next_col_intensity = np.mean(manipulated_image[:, -dx - 2])
        if (next_col_intensity - curr_col_intensity) > threshold:
            right_candidates.append(manipulated_image.shape[1] -dx -1)
    right = right_candidates[-1] if right_candidates else manipulated_image.shape[1] - 1
    
    cropped_image = original_image[top:bottom, left:right, :]
    return cropped_image


def automatic_cropping_color(manipulated_image):
    #NOW CROP THE COLOR (color bars)
    edges = 20

    x_length = manipulated_image.shape[1]
    y_length = manipulated_image.shape[0]


    def check_condition(pixels):

        #if any pixel is greater than 230 it might be a color border
        first_condition = np.any(pixels > 230)
        #if any dif between channels for a single pixel is this large it might be significant
        diffs = np.max(pixels, axis=-1) - np.min(pixels, axis=-1)
        second_condition = np.any(diffs > 210)
        #peaks at brightest point (spot)
        third_condition = np.sum(pixels == 255)
        #peaks at darkest point (spot)
        fourth_condition = np.sum(pixels == 1)
        # not just one high difference or bright pixel, but many major differences between channels
        fifth_condition = np.sum(diffs > 120) > 40
        # not just differences, but also pixels that are super low compared to piuxels around it, but arent necessarily dark
        sixth_condition = (np.sum(pixels < 50) > 800) & (np.sum(diffs > 70) > 800)
        if ((first_condition & second_condition & fifth_condition) | (third_condition > 10) | (fourth_condition > 10) | (sixth_condition)):
            return True
        return False
    
    # top check (rows)
    top_candidates = []
    for dy in range((y_length // edges) - 1):
        row = manipulated_image[dy, :, :]
        if check_condition(row):
            top_candidates.append(dy + 1)
    top = top_candidates[-1] if top_candidates else 0

    # bottom check (rows)
    bottom_candidates = []
    for dy in range((y_length // edges) - 1):
        row = manipulated_image[-(dy+1), :, :]
        if check_condition(row):
            bottom_candidates.append(y_length - dy - 1)
    bottom = bottom_candidates[-1] if bottom_candidates else y_length

    # left check (columns)
    left_candidates = []
    for dx in range((x_length // edges) - 1):
        col = manipulated_image[(y_length // edges):-(y_length // edges), dx, :] 
        if check_condition(col):
            left_candidates.append(dx + 1)
    left = left_candidates[-1] if left_candidates else 0

    # right check (columns)
    right_candidates = []
    for dx in range((x_length // edges) - 1):
        col = manipulated_image[(y_length // edges):-(y_length // edges), -(dx+1), :] 
        if check_condition(col):
            right_candidates.append(x_length - dx - 1)
    right = right_candidates[-1] if right_candidates else x_length

    cropped_image = manipulated_image[top:bottom, left:right, :]
    return cropped_image