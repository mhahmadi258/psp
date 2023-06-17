from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from utils import data_utils


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im


class MHImagesDataset(Dataset):

	def __init__(self, source_root, train, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		if train:
			self.source_paths =self.source_paths[:int(len(self.source_paths)*0.9)]
		else:
			self.source_paths = self.source_paths[int(len(self.source_paths)*0.9):]
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		idx = 0
		while True:
			path = self.source_paths[index-idx]
			try:
				img = Image.open(path)
				break
			except :
				idx +=1
   
		img = img.convert('RGB') if self.opts.label_nc == 0 else img.convert('L')
  
		from_im = img.crop((0,0,512,512))
		to_im = img.crop((512,0,1024,512))


		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im
