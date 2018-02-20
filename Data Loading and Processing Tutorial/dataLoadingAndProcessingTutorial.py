from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as numpy
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()

# Plot
def show_landmarks(image, landmarks):
	""" Show image with landmarks """
	plt.imshow(image)
	plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')	
	plt.pause(2.0)

landmarks_frame = pd.read_csv('./faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n,0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1,2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

plt.figure()
show_landmarks(io.imread(os.path.join('./faces/', img_name)), landmarks)
plt.show()

class FaceLandMarksDataset(Dataset):
	""" Face Landmarks dataset. """

	def __init__(self, csv_file, root_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform(callable, optional): Optional transform to be applied 
			on sample
		"""
		self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.landmarks_frame)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
		image = io.imread(img_name)
		landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
		landmarks = landmarks.astype('float').reshape(-1,2)
		sample = {'image': image, 'landmarks': landmarks}

		if self.transform:
			sample = self.transform(sample)
		return sample

face_dataset = FaceLandMarksDataset(csv_file='faces/face_landmarks.csv', root_dir='faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
	sample = face_dataset[i]

	print(i, sample['image'].shape, sample['landmarks'].shape)

	ax = plt.subplot(1,4,i+1)
	plt.tight_layout()
	ax.set_title('Sample #{}'.format(i))
	ax.axis('off')
	show_landmarks(**sample)

	if i == 3:
		plt.show()
		break