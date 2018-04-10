import os
import sys
import csv
import math
import random
import shutil
import tarfile


# number of images for each pet category (e.g. cats, dogs, etc.)
PET_IMAGES_PER_CLASS = 8000

# number of images for general 'not pet' category
NOT_PET_IMAGES_TOTAL = 360000

# how we split train / validation / test set.
PERCENTAGE_FOR_TRAIN = 0.75
PERCENTAGE_FOR_VALIDATION = 0.2
PERCENTAGE_FOR_TEST = 0.05

# the wnids in 'not pet' category but contains pet images
PET_IMAGES_IN_NOT_PET = [
    'n00005787', 'n00288000', 'n00288190', 'n00288384', 'n00440218', 'n00449977', 'n00450070', 'n00450335', 'n00450700',
    'n00450866', 'n00450998', 'n00451186', 'n00452864', 'n00453126', 'n00453478', 'n00453935', 'n00454237', 'n00454395',
    'n00454493', 'n01322508', 'n01519873', 'n01696633', 'n01697178', 'n01697457', 'n01698434', 'n01698640', 'n02010272',
    'n02062430', 'n02062744', 'n02064816', 'n02065026', 'n02066245', 'n02068541', 'n02068974', 'n02069412', 'n02069701',
    'n02069974', 'n02071294', 'n02071636', 'n02391234', 'n02391373', 'n02469248', 'n02900160', 'n02900459', 'n02900594',
    'n02910864', 'n02912557', 'n02937958', 'n03217739', 'n03218198', 'n03352232', 'n03388711', 'n03410740', 'n03480719',
    'n03480973', 'n03610524', 'n03638014', 'n03644378', 'n03651843', 'n03652932', 'n03745146', 'n03803284', 'n03831203',
    'n03831382', 'n03959014', 'n03981924', 'n03993703', 'n04100519', 'n04215153', 'n04295353', 'n04353573', 'n04368840',
    'n04486616', 'n04577567', 'n04593629', 'n04979002', 'n04979307', 'n07805594', 'n07805731', 'n07805966', 'n07806043',
    'n07806120', 'n08560295', 'n08616050', 'n08614632', 'n09290350', 'n09902353', 'n09967555', 'n10062594', 'n10185793',
    'n10186068', 'n10186143', 'n10186216', 'n10342893', 'n10530383', 'n10538733', 'n10538853', 'n10540252', 'n10722029',
    'n10802507', 'n12118414']


def _read_wnid_list(imagenet_folder):
    return [name[:9] for name in os.listdir(imagenet_folder)]


def _read_pet_list():
    with open('imagenet-animals.csv', mode='r') as infile:
        reader = csv.reader(infile)
        return [rows[0] for rows in reader]  # if rows[3] == '' or rows[3] != 'N'


def _read_not_pet_list(imagenet_folder):
    return list(set(_read_wnid_list(imagenet_folder)) - set(_read_pet_list()) - set(PET_IMAGES_IN_NOT_PET))


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

    avg_total = int(math.ceil(float(copy_total) / len(dic)))
    avg_train = int(math.ceil(float(avg_total) * PERCENTAGE_FOR_TRAIN))
    avg_val = int(math.ceil(float(avg_total) * PERCENTAGE_FOR_VALIDATION))
    avg_test = avg_total - avg_train - avg_val
    print('  Pick up {:4d} images from each subclass, {:4d} for training, {:4d} for validation, {:4d} for testing'
          .format(avg_total, avg_train, avg_val, avg_test))

    for wnid in dic:
        tar = os.path.join(imagenet_folder, wnid + '.tar')
        handle = tarfile.open(tar)
        files = handle.getmembers()
        random.shuffle(files)
        index = 0
        dst_folder = ''
        for item in files:
            if index == 0:
                dst_folder = os.path.join(output_folder, 'train', cls)
                os.makedirs(dst_folder)
            elif index == avg_train:
                dst_folder = os.path.join(output_folder, 'val', cls)
                os.makedirs(dst_folder)
            elif index == avg_train + avg_val:
                dst_folder = os.path.join(output_folder, 'test', cls)
                os.makedirs(dst_folder)
            elif index == avg_total:
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


def _clean_up(output_folder):
    dic = {
        'n01640846_9466.JPEG': 'R',
        'n04155068_537.JPEG': 'N',
        'n12757303_3302.JPEG': 'N'
    }
    # delete some broken images
    for name, folder in dic.iteritems():
        path = os.path.join(output_folder, folder, name)
        if os.path.exists(path):
            os.remove(path)


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
        _clean_up(sys.argv[2])