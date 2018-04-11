import os
import sys
import csv
import math
import random
import shutil
import tarfile
import glob

# how we split train / validation / test set.
PERCENTAGE_FOR_TRAIN = 0.75
PERCENTAGE_FOR_VALIDATION = 0.2
PERCENTAGE_FOR_TEST = 0.05

# the cat categories in oxford iiit pet dataset
OXFORD_CAT_CATEGORIES = [
    'Egyptian_Mau', 'Maine_Coon', 'Bengal', 'Sphynx', 'Ragdoll', 'Siamese', 'Abyssinian', 'Birman', 'British_Shorthair',
    'Persian',  'Russian_Blue',  'Bombay']

# the dog categories in oxford iiit pet dataset
OXFORD_DOG_CATEGORIES = [
    'american_bulldog', 'wheaten_terrier', 'english_setter', 'shiba_inu', 'saint_bernard', 'leonberger', 'samoyed',
    'scottish_terrier', 'miniature_pinscher', 'pomeranian', 'english_cocker_spaniel','pug', 'basset_hound', 'chihuahua',
    'german_shorthaired', 'yorkshire_terrier', 'havanese', 'japanese_chin', 'boxer', 'keeshond', 'newfoundland',
    'american_pit_bull_terrier', 'staffordshire_bull_terrier', 'great_pyrenees', 'beagle']

# the broken image files in oxford iiit pet dataset, see:
# https://gist.github.com/haoxi911/c75b65a421620b7f8cb0523c3cd069f1
BROKEN_IMAGE_FILES = [
    'Egyptian_Mau_167.jpg', 'Egyptian_Mau_191.jpg', 'Egyptian_Mau_177.jpg', 'Egyptian_Mau_139.jpg',
    'Egyptian_Mau_14.jpg', 'Egyptian_Mau_129.jpg', 'Egyptian_Mau_186.jpg', 'Egyptian_Mau_145.jpg',
    'staffordshire_bull_terrier_2.jpg', 'staffordshire_bull_terrier_22.jpg', 'Abyssinian_5.jpg', 'Abyssinian_34.jpg']


def _import_oxford_dataset(cls, breeds, input_folder, output_folder):
    for breed in breeds:
        file_glob = os.path.join(input_folder, breed + '*.jpg')
        files = glob.glob(file_glob)
        random.shuffle(files)

        avg_total = len(files)
        avg_train = max(int(math.ceil(float(avg_total) * PERCENTAGE_FOR_TRAIN)), 1)
        avg_val = max(int(math.ceil(float(avg_total) * PERCENTAGE_FOR_VALIDATION)), 1)
        avg_test = max(avg_total - avg_train - avg_val, 1)
        avg_total = avg_train + avg_val + avg_test
        print('  Pick up {:4d} images from {:30s}, {:4d} for training, {:4d} for validation, {:4d} for testing'
              .format(avg_total, breed, avg_train, avg_val, avg_test))

        index = 0
        dst_folder = ''
        for item in files:
            if index == 0:
                dst_folder = os.path.join(output_folder, cls, 'training')
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
            elif index == avg_train:
                dst_folder = os.path.join(output_folder, cls, 'validation')
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
            elif index == avg_train + avg_val:
                dst_folder = os.path.join(output_folder, cls, 'testing')
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
            elif index >= avg_total:
                break
            #shutil.copyfile(item, dst_folder)
            print('  Copied file: %s' % item)
            index += 1


if __name__ == '__main__':
    if not (len(sys.argv) == 4):
        print("Usage: enhance_pet_dataset.py input_folder input_type output_folder")
    else:
        if sys.argv[2] == 'oxford':
            _import_oxford_dataset('C', OXFORD_CAT_CATEGORIES, sys.argv[1], sys.argv[3])
            _import_oxford_dataset('D', OXFORD_DOG_CATEGORIES, sys.argv[1], sys.argv[3])
