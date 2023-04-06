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

def make_babapour_dataset(dir, random_seq=None, random_frame=None):
    paths = list()
    for video in os.listdir(dir):
        video_path = os.path.join(dir, video)
        for person in os.listdir(video_path):
            person_path = os.path.join(video_path,person)
            target_dir = os.path.join(person_path,'best_faces')
            target = os.path.join(target_dir,os.listdir(target_dir)[0])
            if random_seq:
                selected_seqs = os.listdir(person_path)[::random_seq]
            else:
                selected_seqs = os.listdir(person_path)
            for seq in selected_seqs:
                if seq != 'best_faces':
                    seq_path = os.path.join(person_path,seq)
                    if random:
                        selected_frames = random.sample(os.listdir(seq_path),random_frame)
                    else:
                        selected_frames = os.listdir(seq_path)
                    for frame in selected_frames:
                        paths.append((os.path.join(seq_path,frame), target))
    return paths