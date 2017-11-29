import cv2       # opencv
import numpy as np 

#-----------obsFilter---------------------------------------------
# Description: the function takes the observation and processes it
#              with various filters.
# params: observation_n: A list
# pre: the list needs to be non-empty
# post: the processed image is returned: A 442 x 800 grey scale ndarray
#-----------------------------------------------------------------
def obsFilter(observation_n):
  if(observation_n != None):
    #the keyword for accessing the pixel info is 'vision'
    vision = observation_n[0]['vision']    # this is a 768 x 1024 x 3 ndarray
    #crop to 200 x 200
    output = vision[330:530,318:518,:]
    #grey-scale conversion
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    #gaussian downsizing
    output = cv2.pyrDown(output)
    #de-noise using adaptive gaussian
    output = cv2.adaptiveThreshold(output,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    #morphology
    kernel = np.ones((2,2),np.uint8)
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    return output

    
