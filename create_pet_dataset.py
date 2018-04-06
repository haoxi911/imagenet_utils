import os
import sys
import csv
import math
import random
import shutil
import tarfile


# number of images for each pet category (e.g. cats, dogs, etc.)
PET_IMAGES_PER_CLASS = 4000

# number of images for general 'not pet' category
NOT_PET_IMAGES_TOTAL = 60000


def _read_wnid_list(imagenet_folder):
    return [name[:9] for name in os.listdir(imagenet_folder)]


def _read_pet_list():
    with open('imagenet-animals.csv', mode='r') as infile:
        reader = csv.reader(infile)
        return [rows[0] for rows in reader if rows[3] == '' or rows[3] != 'N']


def _read_not_pet_list(imagenet_folder):
    return list(set(_read_wnid_list(imagenet_folder)) - set(_read_pet_list()))


def _read_pet_summary(imagenet_folder):
    summary = {}
    with open('imagenet-animals.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[3] != '' and row[3] != 'M' and row[3] != 'N':
                summary.setdefault(row[3], {}).update({row[0]: row[1]})
    for row in _read_not_pet_list(imagenet_folder):
        summary.setdefault('N', {}).update({row: 1})
    return summary


def _copy_images_for_class(cls, dic, copy_total, imagenet_folder, output_folder):
    if cls != 'N':
        total = 0
        for wnid, imgs in dic.iteritems():
            total += int(imgs)
        print('- Class: {:5s} Subclasses: {:5d}   Images: {:8d}'.format(cls, len(dic), total))
    else:
        print('- Class: {:5s} Subclasses: {:5d} '.format(cls, len(dic)))

    avg = int(math.ceil(float(copy_total) / len(dic)))
    print('  Pick up {:4d} images from each subclass'.format(avg))
    dst_folder = os.path.join(output_folder, cls)
    os.makedirs(dst_folder)

    for wnid in dic:
        tar = os.path.join(imagenet_folder, wnid + '.tar')
        handle = tarfile.open(tar)
        files = handle.getmembers()
        random.shuffle(files)
        index = 0
        for item in files:
            if index == avg:
                break
            handle.extract(item, dst_folder)
            print('  Extracted file: %s' % item.name)
            index += 1
        handle.close()


def _copy_images(cls, dic, imagenet_folder, output_folder):
    if key != 'N':
        _copy_images_for_class(cls, dic, PET_IMAGES_PER_CLASS, imagenet_folder, output_folder)
    else:
        _copy_images_for_class(cls, dic, NOT_PET_IMAGES_TOTAL, imagenet_folder, output_folder)


if __name__ == '__main__':
    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        print("Usage: create_pet_dataset.py imagenet_folder output_folder")
    else:
        summary = _read_pet_summary(sys.argv[1])
        if len(sys.argv) == 3:
            if os.path.exists(sys.argv[2]):
                shutil.rmtree(sys.argv[2])
            for key in summary:
                _copy_images(key, summary[key], sys.argv[1], sys.argv[2])
        else:
            for key in summary:
                if key == sys.argv[3]:
                    if os.path.exists(os.path.join(sys.argv[2], key)):
                        shutil.rmtree(os.path.join(sys.argv[2], key))
                    _copy_images(key, summary[key], sys.argv[1], sys.argv[2])
                    break
