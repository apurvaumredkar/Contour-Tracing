import numpy as np
import cv2

def harris_points(input_image):
    #noise removal
    input_img = cv2.imread(input_image)
    img_shape_x = input_img.shape[0]
    img_shape_y = input_img.shape[1]
    numerator = input_img ** 1.5
    denominator = input_img ** 0.5
    kernel = np.ones([3, 3], dtype = int)
    noise_filtered_img = cv2.filter2D(numerator, -1, kernel) / cv2.filter2D(denominator, -1, kernel)
    noise_filtered_img = noise_filtered_img.astype(np.uint8)
    #saliency detection
    salient_func = cv2.saliency.StaticSaliencyFineGrained_create()
    (stat, salient_img) = salient_func.computeSaliency(noise_filtered_img)
    salient_img = (salient_img*255).astype(np.uint8)
    #adding gaussain blur and thresholding
    blurred_img = cv2.GaussianBlur(salient_img,(3,3),0)
    thresh_img = cv2.adaptiveThreshold(blurred_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,2)
    #harris points detection algorithm
    grid_size = np.ceil(img_shape_x/100).astype(np.uint8)
    if grid_size<3: grid_size = 3
    k = 0.04
    harris_response = np.zeros((img_shape_x, img_shape_y))
    point_map = np.zeros_like(input_img)
    for i in range(0, img_shape_x, grid_size):
        for j in range(0, img_shape_y, grid_size):
            img_block = thresh_img[i:i+grid_size, j:j+grid_size]
            block_x, block_y = np.gradient(img_block) - np.mean(img_block)
            ixx = block_x**2
            iyy = block_y**2
            ixy = block_x*block_y
            # calculating the covariance matrix values
            sxx = np.sum(ixx)
            syy = np.sum(iyy)
            sxy = np.sum(ixy)
            covariance_matrix = np.array([[sxx,sxy],[sxy,syy]])
            # calculating the eigenvalues
            eig_vals = np.linalg.eigvals(covariance_matrix)
            # calculating the Harris Response
            harris_response[i,j] = min(eig_vals)
    # plotting the Harris Points
    point_map[harris_response>0] = [0,0,255]
    point_map = point_map.astype(np.uint8)
    return point_map
