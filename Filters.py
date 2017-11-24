import cv2       # opencv

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
    #resize to 442 x 800
    output = vision[88:530,18:818,:]
    #to grey
    return cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
