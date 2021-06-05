#Importing the essential libraries
import numpy as np
import cv2

# Defining the function that will trace the contour around the input image
def contour_trace(input_image):
  input_img = cv2.imread(input_image)
  #cv2.imshow("Input Image", input_img)
  #Finding the height and width of the input image
  img_shape_x = input_img.shape[0]
  img_shape_y = input_img.shape[1]
  
  # Step 1: Noise Filtering (to enhance the image)
  # We have to strike a balance between retaining the edges of an image and filtering noise.
  # Hence we will discard salt and pepper noise only.
  numerator = input_img ** 1.5
  denominator = input_img ** 0.5
  kernel = np.ones([3, 3], dtype = int)
  noise_filtered_img = cv2.filter2D(numerator, -1, kernel) / cv2.filter2D(denominator, -1, kernel)
  noise_filtered_img = noise_filtered_img.astype(np.uint8)
  #cv2.imshow("Noise Filtered Input Image")

  # Step 2: Static Saliency Detection: Finding the dominant objects in the image
  # Using the Fine Grained Static Saliency function
  salient_func = cv2.saliency.StaticSaliencyFineGrained_create()
  (stat, salient_img) = salient_func.computeSaliency(noise_filtered_img)
  salient_img = (salient_img*255).astype(np.uint8)
  #cv2.imshow("Fine Grained Saliency", salient_img)

  # Adding the Gaussian blur and then thresholding the image
  blurred_img = cv2.GaussianBlur(salient_img,(3,3),0)
  thresh_img = cv2.adaptiveThreshold(blurred_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,2)
  #cv2.imshow("Thresholded Image", thresh_img)

  # Applying Harris Corner Detection Algorithm
  # Defining a grid size that is variable according to the input image such that no ValueError is thrown on while calculating the gradient
  grid_size = np.ceil(img_shape_x/100).astype(np.uint8)
  if grid_size<3: grid_size = 3
  # Harris detector free parameter in the equation.
  k = 0.04  
  # Initializing the array that will contain the Harris Responses
  harris_response = np.zeros_like(thresh_img)
  # Initializing the array that will contain the Harris Points
  point_map = np.zeros_like(input_img)

  # To display harris points over the original image (optional)
  #harris_points = np.copy(input_img)
  for i in range(0, img_shape_x, grid_size):
    for j in range(0, img_shape_y, grid_size):
      img_block = thresh_img[i:i+grid_size, j:j+grid_size]
      block_x, block_y = np.gradient(img_block) - np.mean(img_block)
      ixx = block_x**2
      iyy = block_y**2
      ixy = block_x*block_y
      #Calculating the covariance matrix values
      sxx = np.sum(ixx)
      syy = np.sum(iyy)
      sxy = np.sum(ixy)
      covariance_matrix = np.array([[sxx,sxy],[sxy,syy]])
      #Calculating the eigenvalues
      eig_vals = np.linalg.eigvals(covariance_matrix)
      #Calculating the Harris Response
      harris_response[i,j] = min(eig_vals)
  # Plotting the Harris Points
  point_map[harris_response>0] = [0,0,255]
  #harris_points[harris_response>0] = [0,0,255]
  point_map = point_map.astype(np.uint8)
  #cv2.imshow(point_map)
  #cv2.imshow(harris_points)

  # Tracing the contour based on the Harris points
  # 1. Adjusting the Harris points by dilation, erosion and closing morphological transform

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*grid_size,2*grid_size))
  point_map_dilated = cv2.dilate(point_map,kernel)
  point_map_eroded = cv2.erode(point_map_dilated,kernel)
  point_map_closing = cv2.morphologyEx(point_map_eroded, cv2.MORPH_CLOSE,kernel,iterations=7)
  adjusted_points = np.copy(point_map_closing[:,:,2]) 
  #cv2.imshow(adjusted_points)

  # 2. Extracting the boundary contours
  # Initializing the list of contours
  contours = []
  # A function that will check if the pixel is a boundary pixel
  # Working: The N8 neighbourhood points are checked, if there is a black pixel present it means the pixel is present at the edge and hence a boundary pixel
  def checkIfBoundaryPixel(x,y):
    block = (adjusted_points[x-1:x+2, y-1:y+2]==255).flatten()
    return block.all()==False and adjusted_points[x,y]==255

  # Top edge boundary points
  for j in range(img_shape_y):
    if adjusted_points[0,j] == 255: contours.append([0,j])
  # Bottom edge boundary points
  for j in range(img_shape_y):
    if adjusted_points[img_shape_x-1,j] == 255: contours.append([img_shape_x-1,j])

  # Left edge boundary points
  for i in range(img_shape_x):
    if adjusted_points[i, 0] == 255: contours.append([i,0])

  # Right edge boundary points
  for i in range(img_shape_x):
    if adjusted_points[i, img_shape_y-1] == 255: contours.append([i,img_shape_y-1])

  # Extracting the coordinates of the remaining boundary pixels
  for i in range(img_shape_x):
    for j in range(img_shape_y):
      if checkIfBoundaryPixel(i,j): contours.append([i,j])

  # Initializing the contour map
  contour_map = np.zeros_like(input_img)
  # Mapping the points
  for i in contours:
    contour_map[i[0],i[1],:] = [0,0,255]
  # Dilating the contour for better results
  contour_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
  dilated_contours = cv2.dilate(contour_map,contour_dilation_kernel)
  #cv2.imshow(dilated_contours)

  # Mapping the contours over the input image
  output_img = np.copy(input_img)
  for i in range(img_shape_x):
    for j in range(img_shape_y):
      if dilated_contours[i,j,2] == 255:
        output_img[i,j,:] = [0,0,255]
  #cv2.imshow(output_img)
  # Returning the contour traced image
  return output_img

input_img = input('Enter file path here: ') 
output_img = contour_trace(input_img)
cv2.imshow("Input Image", cv2.imread(input_img))
cv2.imshow("Output Image", output_img)
cv2.waitKey()

