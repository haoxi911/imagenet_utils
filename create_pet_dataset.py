import os
import sys
import csv
import math
import random
import shutil
import tarfile
import glob

# This script builds a Mobilenet `pet` dataset using ImageNet data.
#
# 1) the following pet categories will be built from listed sources:
#
# *bird*
# n01503061 bird
# n01613615 young bird
#
# *cat*
# n02121808 domestic cat, house cat, Felis domesticus, Felis catus
# n02121620 cat, true cat
# n02122948 kitten, kitty
#
# *dog*
# n02084071 dog, domestic dog, Canis familiaris
# n02115335 wild dog
# n02083672 bitch
# n01322343 pup, whelp
#
# *fish*
# n02512053 fish

# *horse*
# n02374451 horse, Equus caballus
# n02376542 foal
#
# *reptile*
# n01661091 reptile, reptilian
#
# *small animals*
# n02342885 hamster
# n02364520 cavy
# n02367492 chinchilla, Chinchilla laniger
#
# 2) the following not-pet categories will be built from listed sources:
#
# *bed linen*
# n02821030 bed linen
#
# *blanket*
# n02849154 blanket, cover
#
# *car*
# n02958343 car, auto, automobile, machine, motorcar
# n02959942 car, railcar, railway car, railroad car
#
# *carpet*
# n04118021 rug, carpet, carpeting
#
# *dirt*
# n14844693 soil, dirt
#
# *fence*
# n03327234 fence, fencing
#
# *field*
# n08569998 field
#
# *goat*
# n02418064 goat antelope
# n02419796 antelope
# n02411705 sheep
# n02428842 forest goat, spindle horn, Pseudoryx nghetinhensis
#
# *grass*
# n12102133 grass
#
# *hay*
# n07802026 hay
#
# *human hair*
# n05256862 hairdo, hairstyle, hair style, coiffure, coif
# n03476083 hairpiece, false hair, postiche
#
# *mulch*
# n03797896 mulch
#
# *person*
# n00007846  person, individual, someone, somebody, mortal, soul
# n07942152  people
# n04976952  complexion, skin color, skin colour
# n13895262  belly
# n06892775  concert
# n08249459  concert band, military band
# n08079613  baseball club, ball club, club, nine
#
# *scarf*
# n04143897 scarf
#
# *snow*
# n11508382 snow, snowfall
#
# *toy animals*
# n03964744 plaything, toy
# n02085374 toy dog, toy
#
# *truck*
# n04490091 truck, motortruck
#
# 3) the following categories will always be excluded (TBD):
#
# n00523513 sport, athletics
# n07805594 bird feed, bird food, birdseed
# n07805731 petfood, pet-food, pet food
# n08555333 stockbroker belt
# n08182379 crowd
# n05538625 head, caput
# n03430959 gear, paraphernalia, appurtenance
# n03538634 horse-drawn vehicle
# n03217739 dogcart
# n03538406 horse cart, horse-cart
# n03351434 fishing gear, tackle, fishing tackle, fishing rig, rig
# n04124202 saddle blanket, saddlecloth, horse blanket
# n03539678 horse-trail
# n00450070 horse racing
# n04294879 stable, stalls, horse barn
# n03538037 horse, gymnastic horse
# n03218198 dogsled, dog sled, dog sleigh
# n03610524 kennel, doghouse, dog house
# n03745146 menagerie, zoo, zoological garden
# n03981924 pony cart, ponycart, donkey cart, tub-cart
# n04123740 saddle
# n03993703 pound, dog pound
# n08616050 pasture, pastureland, grazing land, lea, ley
# n00015388 animal, animate being, beast, brute, creature, fauna
# n01318894 animal, animate being, beast, brute, creature, fauna / pet
# n02982515 cat box
# n05217859 body, dead body
# n02843553 bird feeder, birdfeeder, feeder
# n07805966 dog food
# n03993703 pound, dog pound
# n02936714 enclosure
# n03610524 kennel, doghouse, dog house
# n09972661 cowboy, cowpuncher, puncher, cowman, cattleman, cowpoke, cowhand, cowherd
# n10186068 horse trader
# n03920641 pet shop
# n10171567 herder, herdsman, drover
# n03376159 fold, sheepfold, sheep pen, sheepcote
#

# the positive categories
POS_IMAGE_CATEGORIES = [
    'bird', 'cat', 'dog', 'fish', 'horse', 'reptile', 'small animals'
]

# the broken / corrupted images
CORRUPTED_IMAGE_FILES = [
    "n09620794_5529.JPEG",
    "n01640846_9466.JPEG",
    "n04155068_537.JPEG",
    "n12757303_3302.JPEG"
]

# number of images for positive category (e.g. cats, dogs, etc.)
POS_IMAGES_PER_CLASS = 5000

# number of images for negative category (i.e. others)
NEG_IMAGES_PER_CLASS = 3000

# how we split train / validation / test set.
PERCENTAGE_FOR_TRAIN = 0.65
PERCENTAGE_FOR_VALIDATION = 0.2
PERCENTAGE_FOR_TEST = 0.15

# minimum number of images for each subclass in dataset
SUBCLASS_IMAGES_MIN = 1


def _read_wnid_full(imagenet_folder):
    return [name[:9] for name in os.listdir(imagenet_folder)]


def _read_wnid_folder(path, d=''):
    wnids = {}
    files = glob.glob(os.path.join(path, '*.csv'))
    for f in files:
        with open(f, mode='r') as infile:
            reader = csv.reader(infile)
            wnids.update({rows[0]: rows[1] for rows in reader if len(d) <= 0 or (len(rows) > 3 and rows[3] == d)})
    return wnids


def _read_dataset_summary(imagenet_folder):
    summary = {}

    # the pet categories we're interested in.
    for folder in POS_IMAGE_CATEGORIES:
        mask = ""
        for i in folder.upper().split():
            mask += i[0]
        summary.setdefault(folder, _read_wnid_folder(os.path.join('./synsets', folder), mask))

    # the not-pet categories.
    for folder in os.listdir('./synsets/others'):
        if os.path.isdir(os.path.join('./synsets/others', folder)):
            summary.setdefault(folder, _read_wnid_folder(os.path.join('./synsets/others', folder), 'X'))

    return summary


def _copy_images_for_class(cls, rows, copy_total, imagenet_folder, output_folder):
    total = 0
    for wnid, count in rows.iteritems():
        total += int(float(count))
    print('- Class: {:5s} Subclasses: {:5d} Total: {:8d}'.format(cls, len(rows), total))

    if total < copy_total:
        copy_total = total
    avg_total = int(math.floor(float(copy_total) / len(rows)))
    avg_train = int(math.floor(float(avg_total) * PERCENTAGE_FOR_TRAIN))
    avg_val = int(math.floor(float(avg_total) * PERCENTAGE_FOR_VALIDATION))
    avg_test = int(math.floor(float(avg_total) * PERCENTAGE_FOR_TEST))
    avg_total = avg_train + avg_val + avg_test
    assert avg_train >= SUBCLASS_IMAGES_MIN and avg_val >= SUBCLASS_IMAGES_MIN and avg_test >= SUBCLASS_IMAGES_MIN
    print('  Pick up {:4d} images from each subclass, {:4d} for training, {:4d} for validation, {:4d} for testing'
          .format(avg_total, avg_train, avg_val, avg_test))

    for wnid, unused in rows.iteritems():
        tar = os.path.join(imagenet_folder, wnid + '.tar')
        handle = tarfile.open(tar)
        files = handle.getmembers()
        if len(files) >= SUBCLASS_IMAGES_MIN:
            random.shuffle(files)
            copy_total = min(len(files), avg_total)
            copy_train = int(math.floor(float(copy_total) * PERCENTAGE_FOR_TRAIN))
            copy_val = int(math.floor(float(copy_total) * PERCENTAGE_FOR_VALIDATION))
            copy_test = int(math.floor(float(copy_total) * PERCENTAGE_FOR_TEST))
            # print ('%d : %d : %d' % (copy_train, copy_val, copy_test))

            index = 0
            dst_folder = ''
            for item in files:
                if index == 0:
                    dst_folder = os.path.join(output_folder, cls, 'training')
                    if not os.path.exists(dst_folder):
                        os.makedirs(dst_folder)
                elif index == copy_train:
                    dst_folder = os.path.join(output_folder, cls, 'validation')
                    if not os.path.exists(dst_folder):
                        os.makedirs(dst_folder)
                elif index == copy_train + copy_val:
                    dst_folder = os.path.join(output_folder, cls, 'testing')
                    if not os.path.exists(dst_folder):
                        os.makedirs(dst_folder)
                elif index >= copy_train + copy_val + copy_test:
                    break
                if item.name not in CORRUPTED_IMAGE_FILES:
                    handle.extract(item, dst_folder)
                    print('  Extracted file: %s' % item.name)
                index += 1
        handle.close()


def _copy_images(cls, dic, imagenet_folder, output_folder):
    if key in POS_IMAGE_CATEGORIES:
        _copy_images_for_class(cls, dic, POS_IMAGES_PER_CLASS, imagenet_folder, output_folder)
    else:
        _copy_images_for_class(cls, dic, NEG_IMAGES_PER_CLASS, imagenet_folder, output_folder)


if __name__ == '__main__':
    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        print("Usage: create_pet_dataset.py imagenet_folder output_folder")
    else:
        summary = _read_dataset_summary(sys.argv[1])
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
