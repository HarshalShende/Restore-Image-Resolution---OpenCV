import cv2
from cv2 import dnn_superres

# initialize super resolution object
sr = dnn_superres.DnnSuperResImpl_create()

# read the model
path = 'EDSR_x4.pb'
sr.readModel(path)

# set the model and scale
sr.setModel('edsr', 4)

# # if you have cuda support
# sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# load the image
image = cv2.imread('hello.png')

# upsample the image
upscaled = sr.upsample(image)
# save the upscaled image
cv2.imwrite('upscaled_hello.png', upscaled)

# traditional method - bicubic
bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv2.INTER_CUBIC)
# save the image
cv2.imwrite('bicubic_hello.png', bicubic)

