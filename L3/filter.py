import cv2
import sys
import numpy             as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	img = sys.argv[1]


	img = cv2.imread(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


	img = img.astype(np.float32)
	img = (img - img.min()) / (img.max() - img.min())


	dx_filter = 0.5 * np.array(
		[[ 0, 0, 0],
		 [-1, 0, 1],
		 [ 0, 0, 0]]
	)

	dx_img = cv2.filter2D(img, -1, dx_filter) * 2

	dy_filter = 0.5 * np.array(
		[[ 0,  1, 0],
		 [ 0,  0, 0],
		 [ 0, -1, 0]]
	)

	dy_img = cv2.filter2D(img, -1, dy_filter) * 2

	filter3 = dx_filter + dy_filter

	img3 = cv2.filter2D(img, -1, filter3)


	minimum = min(img.min(), dx_img.min(), dy_img.min(), img3.min())
	maximum = max(img.max(), dx_img.max(), dy_img.max(), img3.max())

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	im1 = ax1.imshow(img,      vmin=minimum, vmax=maximum)
	im2 = ax2.imshow(dx_img,   vmin=minimum, vmax=maximum)
	im3 = ax3.imshow(dy_img,   vmin=minimum, vmax=maximum)
	im4 = ax4.imshow(img3, vmin=minimum, vmax=maximum)

	ax1.set_title("Original Image")
	ax2.set_title("Filter 1")

	ax3.set_title("Filter 2")
	ax4.set_title("Filter 3")




	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im1, cax=cbar_ax)

	plt.show()