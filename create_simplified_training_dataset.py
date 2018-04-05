import os
import sys
import csv
import math
import random
import shutil

# those we try to identify from a large set of images
WANTED_IMAGES_EACH_CLASS = 2000

# those we try to filter out of a large set of images
UNWANTED_IMAGES_EACH_CLASS = 20000


def _read_input_summary(input_file):
    summary = {}
    with open(input_file, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if len(rows) >= 4 and len(rows[3]) > 0 and rows[3] != 'M':
                summary.setdefault(rows[3], {}).update({rows[0]: rows[1]})
    return summary


def _copy_images_for_class(cls, dic, copy_total, input_folder, output_folder):
    total = 0
    for wnid, imgs in dic.iteritems():
        total += int(imgs)
    print('- Class: {:5s} Subclasses: {:5d}   Images: {:8d}'.format(cls, len(dic), total))

    subfolders = [name for name in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, name))]
    avg_total = int(math.ceil(float(copy_total) / len(dic)))
    print('  Pick up {:4d} images from each subclass'.format(avg_total))
    dst_folder = os.path.join(output_folder, cls)
    os.makedirs(dst_folder)

    for wnid in dic:
        folder = next(val for val in subfolders if wnid in val)
        folder = os.path.join(input_folder, folder)
        files = [os.path.join(folder, name) for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
        random.shuffle(files)
        index = 0
        for item in files:
            if index == avg_total:
                break
            shutil.copy(item, dst_folder)
            print('  Copied file: %s' % item)
            index += 1


def _copy_images_for_wanted_class(cls, dic, input_folder, output_folder):
    _copy_images_for_class(cls, dic, WANTED_IMAGES_EACH_CLASS, input_folder, output_folder)


def _copy_images_for_unwanted_class(cls, dic, input_folder, output_folder):
    _copy_images_for_class(cls, dic, UNWANTED_IMAGES_EACH_CLASS, input_folder, output_folder)


if __name__ == '__main__':
    if not (len(sys.argv) == 4):
        print("Usage: create_simplified_training_dataset.py input_csv input_folder output_folder")

    if len(sys.argv) == 4:
        classes = _read_input_summary(sys.argv[1])
        if os.path.exists(sys.argv[3]):
            shutil.rmtree(sys.argv[3])
        for key in classes:
            if key != 'N':
                _copy_images_for_wanted_class(key, classes[key], sys.argv[2], sys.argv[3])
            else:
                _copy_images_for_unwanted_class(key, classes[key], sys.argv[2], sys.argv[3])
