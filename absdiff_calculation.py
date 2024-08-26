import cv2

image1 = cv2.imread('dalle_image.png')
image2 = cv2.imread('image_with_hidden_data_efficiency_test.png')

abs_diff = cv2.absdiff(image1, image2)

cv2.imshow('Absolute Difference', abs_diff)

cv2.waitKey(0)
cv2.destroyAllWindows()
