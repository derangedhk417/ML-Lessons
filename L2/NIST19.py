import json
import cv2
import os
import code
import sys

import numpy as np

from Progress import ProgressBar


# Designed to load and clean up images from NIST special database 19.
class SDB19Loader:
	def __init__(self, path='../nist_19/', size=24, out='../nist_19_24'):
		self.path = path
		self.size = size
		self.out  = out
		self.load()

	def load(self):
		subdirs  = []
		dirclass = []
		names    = []
		for entry in os.listdir(self.path):
			inner = os.path.join(self.path, entry)
			if os.path.isdir(inner):
				if len(entry) == 2:
					subdirs.append(inner)
					dirclass.append(chr(int(entry, 16)))
					names.append('train_' + entry)

		self.data = []

		pb = ProgressBar("Loading", 15, len(subdirs), update_every=1, ea=15)

		# We now have a list of directories and their corresponding 
		# characters. We need to load the images with opencv, convert
		# them to BGR, crop them and resize them.
		for idx, (c, d, n) in enumerate(zip(dirclass, subdirs, names)):
			subdir = os.path.join(d, n)
			for entry in os.listdir(subdir):
				ext = entry.split('.')[-1].lower()
				if ext == 'png':
					self.data.append(HandwrittenCharacter(
						os.path.join(subdir, entry), c, self.size
					))

			pb.update(idx + 1)

		pb.finish()

		pb = ProgressBar("Saving", 15, len(self.data), update_every=2000, ea=15)

		self.meta = []

		for i, d in enumerate(self.data):
			outpath = os.path.join(self.out, "%06d.np"%i)
			self.meta.append(d.char)
			np.save(outpath, d.image)
			pb.update(i + 1)

		pb.finish()

		with open(os.path.join(self.out, 'meta.json'), 'w') as file:
			file.write(json.dumps({'classes': self.meta}))



	def __str__(self):
		return "SDB19Loader(path=\'%s\')"%self.path

	def __repr__(self):
		return str(self)

class HandwrittenCharacter:
	def __init__(self, path, char, size):
		self.char = char
		self.path = path

		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = self.crop(img)
		self.image = cv2.resize(img, (size, size))
		self.image = (255 - self.image) / 255

	def crop(self, img):
		img[img > 200] = 255

		left_adjust = 0
		while left_adjust < img.shape[1] and (img[:, left_adjust] == 255).all():
			left_adjust += 1


		right_adjust = img.shape[1] - 1
		while right_adjust >= 0 and (img[:, right_adjust - 1] == 255).all():
			right_adjust -= 1

		top_adjust = 0
		while top_adjust < img.shape[0] and (img[top_adjust, :] == 255).all():
			top_adjust += 1

		bottom_adjust = img.shape[0] - 1
		while bottom_adjust >= 0 and (img[bottom_adjust - 1, :] == 255).all():
			bottom_adjust -= 1

		return img[top_adjust:bottom_adjust, left_adjust:right_adjust]

	def __str__(self):
		return "HandwrittenCharacter(path=\'%s\', char=\'%s\')"%(self.path, self.char)

	def __repr__(self):
		return str(self)

if __name__ == '__main__':
	path = sys.argv[1]
	out  = sys.argv[2]

	loader = SDB19Loader(path=path, out=out)