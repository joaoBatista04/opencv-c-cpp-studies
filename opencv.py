import cv2
import matplotlib.pyplot as plt
import numpy as np

#Flag 1 - color image
#Flag 0 - Grayscale
#Flag -1 - Unchanged image
img = cv2.imread("./mandril.jpg", 1)

h, w, c = img.shape
print("Dimensions of the image is: nnHeight:", h, "pixelsnWidth:", w, "pixelsnNumber of Channels:", c)

#print(type(img))
#print(img.dtype)
#print(img)

#cv2.imshow('Mandrill', img)
#k = cv2.waitKey(0)
#if k == 27 or k == ord('q'):
#   cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Mandrill_grey.jpg', gray)

def extract_bit_plane(cd):
    #  extracting all bit one by one 
    # from 1st to 8th in variable 
    # from c1 to c8 respectively 
    c1 = np.mod(cd, 2)
    c2 = np.mod(np.floor(cd/2), 2)
    c3 = np.mod(np.floor(cd/4), 2)
    c4 = np.mod(np.floor(cd/8), 2)
    c5 = np.mod(np.floor(cd/16), 2)
    c6 = np.mod(np.floor(cd/32), 2)
    c7 = np.mod(np.floor(cd/64), 2)
    c8 = np.mod(np.floor(cd/128), 2)
    # combining image again to form equivalent to original grayscale image 
    cc = 2 * (2 * (2 * c8 + c7) + c6) # reconstructing image  with 3 most significant bit planes
    to_plot = [cd, c1, c2, c3, c4, c5, c6, c7, c8, cc]
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(10, 8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, i in zip(axes.flat, to_plot):
        ax.imshow(i, cmap='gray')
    plt.tight_layout()
    plt.show()
    return cc

reconstructed_image = extract_bit_plane(gray)