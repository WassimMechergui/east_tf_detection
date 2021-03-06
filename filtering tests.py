from imutils.object_detection import non_max_suppression
from imutils import rotate
import numpy as np
import cv2
import time
def average_color(image):
    x,y, c = image.shape[0], image.shape[1], image.shape[2]
    r , g , b = 0, 0 , 0
    for i in range(x):
        for j in range(y):
            r += image[i,j,0]
            g += image[i,j,1]
            b += image[i,j,2]
    return r /(x*y+1) , g/(x*y+1), b/(x*y+1) 

net = cv2.dnn.readNet('frozen_east_text_detection.pb')


def EAST_text_detector(original, image, confidence=0.1):

    reference = time.time()

    # Set the new width and height and determine the changed ratio
    (h, W) = image.shape[:2]
    (newW, newH) = (480, 480)
    rW = W / float(newW)
    rH = h / float(newH)



    # Resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (h, W) = image.shape[:2]

    # Define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (480,480), (123.68, 116.78, 103.94), swapRB=True, crop=True)
    net.setInput(blob)
    
    scores, geometry = net.forward(layerNames)
    
    # Grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores

    
    return np.array([1]), scores[0,0]

# Convert to grayscale and Otsu's threshold
image = cv2.imread('data/2.jpg')

# Perform EAST text detection

do_image = False

if(do_image):

    final = image[:,:,:]/255
    for angle in [30,0,-30]:
        cop = rotate(image,angle)
        result, mask = EAST_text_detector(cop,cop)
        mask = cv2.cvtColor(cv2.resize(mask,(result.shape[1],result.shape[0])),cv2.COLOR_GRAY2RGB)
        mask[:,:,1] = 0
        final += rotate(result/255 + ((1.5*mask)), -angle)
    cv2.imshow('result', final/3)
    cv2.waitKey(0)


def filter_p(src):

    kernel_size = 3

#    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    laplacian3 = cv2.Laplacian(src,ddepth= cv2.CV_8UC3, ksize = 3)


#    gray_bgr3 =  cv2.cvtColor(laplacian3, cv2.COLOR_GRAY2BGR)



    dst3 = cv2.fastNlMeansDenoisingColored(laplacian3,None,30,15,21,7)



    #t = cv2.GaussianBlur(dst3 , (5,5) ,cv2.BORDER_DEFAULT)


    kernel = np.ones((5,5),np.float32)/25
    
    dst = cv2.filter2D(dst3,-1,kernel)

    return dst




videoMode = False
if(videoMode):
    cap = cv2.VideoCapture(0)
    i = 0
    while(True) and i == 0:
        ret, frame = cap.read()
        print(frame.shape)
        final = frame[:,:,:]/255

        
        for angle in [-30,0,30]:
            cop = rotate(frame,angle)

            result , mask = EAST_text_detector(cop,cop)

            mask = cv2.cvtColor(cv2.resize(mask,(result.shape[1],result.shape[0])),cv2.COLOR_GRAY2RGB)

            mask[:,:,1] = 0

            final += rotate(result/255 + ((0.5*mask-0.5)), -angle)
        
        filter_s = filter_p(frame)

        cv2.imshow('result', (final/3) )
        cv2.imshow('result', filter_s )
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




src = cv2.imread('./data/pee.jpg')

s =  filter_p(src) 

result, mask = EAST_text_detector(src,src)
mask = cv2.cvtColor(cv2.resize(mask,(result.shape[1],result.shape[0])),cv2.COLOR_GRAY2RGB)
mask[:,:,1] = 0

cv2.imshow('result', s/255 + mask)

cv2.waitKey(0)