import os
import sys
import csv
import math
import random
import shutil
import tarfile

# those we try to identify from a large set of images
WANTED_IMAGES_CATEGORIES = ['B', 'C', 'D', 'F', 'H', 'R', 'SA']
WANTED_IMAGES_FOR_TRAIN = 2000

# those we try to filter out of a large set of images
UNWANTED_IMAGES_CATEGORY = 'N'
UNWANTED_IMAGES_FOR_TRAIN = 30000


def _read_wnid_list(imagenet_folder):
    return [name[:9] for name in os.listdir(imagenet_folder)]


def _read_wanted_list(input_file):
    with open(input_file, mode='r') as infile:
        reader = csv.reader(infile)
        return [rows[0] for rows in reader if rows[3] == '' or rows[3] != UNWANTED_IMAGES_CATEGORY]


def _read_unwanted_list(imagenet_folder, input_file):
    return list(set(_read_wnid_list(imagenet_folder)) - set(_read_wanted_list(input_file)))


def _read_input_summary(imagenet_folder, input_file):
    summary = {}
    with open(input_file, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[3] in WANTED_IMAGES_CATEGORIES:
                summary.setdefault(row[3], {}).update({row[0]: row[1]})
    for row in _read_unwanted_list(imagenet_folder, input_file):
        summary.setdefault(UNWANTED_IMAGES_CATEGORY, {}).update({row: 1})
    return summary


def _copy_images_for_class(cls, dic, train, imagenet_folder, output_folder):
    if cls != UNWANTED_IMAGES_CATEGORY:
        total = 0
        for wnid, imgs in dic.iteritems():
            total += int(imgs)
        print('- Class: {:5s} Subclasses: {:5d}   Images: {:8d}'.format(cls, len(dic), total))
    else:
        print('- Class: {:5s} Subclasses: {:5d} '.format(cls, len(dic)))

    avg_train = int(math.ceil(float(train) / len(dic)))
    print('  Pick up {:4d} images from each subclass'.format(avg_train))
    dst_folder = os.path.join(output_folder, cls)
    os.makedirs(dst_folder)

    for wnid in dic:
        tar = os.path.join(imagenet_folder, wnid + '.tar')
        handle = tarfile.open(tar)
        files = handle.getmembers()
        random.shuffle(files)
        index = 0
        for item in files:
            if index == avg_train:
                break
            handle.extract(item, dst_folder)
            print('  Extracted file: %s' % item.name)
            index += 1
        handle.close()


def _copy_images_for_wanted_class(cls, dic, imagenet_folder, output_folder):
    _copy_images_for_class(cls, dic, WANTED_IMAGES_FOR_TRAIN, imagenet_folder, output_folder)


def _copy_images_for_unwanted_class(cls, dic, imagenet_folder, output_folder):
    _copy_images_for_class(cls, dic, UNWANTED_IMAGES_FOR_TRAIN, imagenet_folder, output_folder)


if __name__ == '__main__':
    if not (len(sys.argv) == 4):
        print("Usage: create_simplified_dataset.py summary_csv imagenet_folder output_folder")
    else:
        classes = _read_input_summary(sys.argv[2], sys.argv[1])
        if os.path.exists(sys.argv[3]):
            shutil.rmtree(sys.argv[3])
        for key in classes:
            if key != UNWANTED_IMAGES_CATEGORY:
                _copy_images_for_wanted_class(key, classes[key], sys.argv[2], sys.argv[3])
            else:
                _copy_images_for_unwanted_class(key, classes[key], sys.argv[2], sys.argv[3])
