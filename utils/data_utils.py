"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def make_babapour_dataset(dir):
    paths = list()
    for video in os.listdir(dir):
        video_path = os.path.join(dir, video)
        for person in os.listdir(video_path):
            person_path = os.path.join(video_path,person)
            target_dir = os.path.join(person_path,'best_faces')
            target = os.path.join(target_dir,os.listdir(target_dir)[0])
            for seq in os.listdir(person_path):
                if seq != 'best_faces':
                    seq_path = os.path.join(person_path,seq)
                    frames = list()
                    for frame in os.listdir(seq_path):
                        frames.append(os.path.join(seq_path,frame))
                    paths.append((frames, target))
    return paths