import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

c = np.array([(50,50), (50, 100), (150,255), (150, 150)])
t1 = np.linspace(0,c[0,1],c[0,0]+1-0).astype('uint8')
print(len(t1))
t2 = np.linspace(c[0,1]+1,c[1,1],c[1,0]-c[0,0]).astype('uint8')
# print(len(t2))
print(t1)
print(t2)

t3 = np.linspace(c[1,1]+1,c[2,1],c[2,0]-c[1,0]).astype('uint8')
t4 = np.linspace(c[2,1]+1,c[3,0],c[2,0]-c[3,0]).astype('uint8')
t5 = np.linspace(c[2,0]+1,255,255-c[2,0]).astype('uint8')
# print(len(t3))
# print(t2)
# print(t3)

transform = np.concatenate((t1,t2),axis=0).astype('uint8')
transform = np.concatenate((transform,t3),axis=0).astype('uint8')
transform = np.concatenate((transform,t4),axis=0).astype('uint8')
transform = np.concatenate((transform,t5),axis=0).astype('uint8')
print(len(transform))

fig , ax =plt.subplots()
ax.plot(transform)

ax.set_xlabel(r'input,$f(\mathbf{x})$')
ax.set_ylabel('Output,$\mathrm{T}[f(\mathbf{x})]$')
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_aspect('equal')
plt.savefig('transform3.png')
plt.show()

img_orig = cv.imread('5e31db5bfd3aa32ba2a5a2443a175711.jpeg',cv.IMREAD_GRAYSCALE)
cv.namedWindow("image",cv.WINDOW_AUTOSIZE)
cv.imshow("image",img_orig)
cv.waitKey(0)

image_transformed_new = cv.LUT(img_orig,transform)
cv.imshow("Image",image_transformed_new)
cv.waitKey(0)
cv.destroyAllWindows()