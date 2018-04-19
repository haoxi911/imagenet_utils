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
#
# 1) the following wnids (and child nodes) will be excluded:
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
#
# 2) the following wnids (and child nodes) will be enhanced:
#
# n02127808 big cat, cat
# n12102133 grass
# n07802026 hay
# n09282208 floor
# n02849154 blanket, cover
# n02821030 bed linen
# n03797896 mulch
# n11508382 snow, snowfall
# n02416519 goat, caprine animal
# n14844693 soil, dirt
# n02430045 deer, cervid
# n04118021 rug, carpet, carpeting
# n03476083 hairpiece, false hair, postiche
# n02818832 bed
# n04143897 scarf
#
#
# 3) the following categories will be created:
#
# *birds*
# n01503061 bird
# n01613615 young bird
#
# *cats*
# n02121808 domestic cat, house cat, Felis domesticus, Felis catus
# n02121620 cat, true cat
# n02122948 kitten, kitty
#
# *dogs*
# n02084071 dog, domestic dog, Canis familiaris
# n02115335 wild dog
# n02083672 bitch
# n01322343 pup, whelp
#
# *fish*
# n02512053 fish
#
# *goats*
# n02418064 goat antelope
# n02411206 musk ox, musk sheep, Ovibos moschatus
# n02419796 antelope
# n02411705 sheep
# n02428842 forest goat, spindle horn, Pseudoryx nghetinhensis
#
# *horses*
# n02374451 horse, Equus caballus
# n02376542 foal
#
# *reptiles*
# n01661091 reptile, reptilian
#
# *small animals*
# n02342885 hamster
# n02364520 cavy
# n02367492 chinchilla, Chinchilla laniger
#
# *persons*
# n00007846  person, individual, someone, somebody, mortal, soul
# n07942152  people
# n04976952  complexion, skin color, skin colour
# n13895262  belly
# n06892775  concert
# n08249459  concert band, military band
# n08079613  baseball club, ball club, club, nine
#


# number of images for positive category (e.g. cats, dogs, persons, etc.)
POS_IMAGES_PER_CLASS = 10000

# number of images for negative category (i.e. not pet)
NEG_IMAGES_PER_CLASS = 500000

# how we split train / validation / test set.
PERCENTAGE_FOR_TRAIN = 0.6
PERCENTAGE_FOR_VALIDATION = 0.2
PERCENTAGE_FOR_TEST = 0.2

# number of images for each enhanced subclass in negative category
NEG_IMAGES_ENHANCED_MIN = 1000

# minimum number of images for each subclass in dataset
SUBCLASS_IMAGES_MIN = 6

# the wnids in negative category which contain images we want to enhance for training
# https://docs.google.com/spreadsheets/d/1m3ODqTe-qwutwwYhFmekQnuXN3WajOXrfAzIH_c19Ec/edit#gid=1618404554
NEG_IMAGES_ENHANCED = [
    'n12102133', 'n07802026', 'n00007846', 'n02472987', 'n09918248', 'n09282208', 'n02849154', 'n03797896', 'n11508382',
    'n02416880', 'n15019030', 'n02430045', 'n04183217']


def _read_wnid_full(imagenet_folder):
    return [name[:9] for name in os.listdir(imagenet_folder)]


def _read_wnid_folder(path, d=''):
    wnids = []
    files = glob.glob(os.path.join(path, '*.csv'))
    for f in files:
        with open(f, mode='r') as infile:
            reader = csv.reader(infile)
            wnids.extend([rows[0] for rows in reader if len(d) <= 0 or (len(rows) > 3 and rows[3] == d)])
    return wnids


def _read_wnid_animals():
    with open('./synsets/imagenet-animals.csv', mode='r') as infile:
        reader = csv.reader(infile)
        return [rows[0] for rows in reader]


def _read_negative_list(imagenet_folder):
    return list(set(_read_wnid_full(imagenet_folder)) - set(_read_wnid_folder('./synsets/mixed'))
                - set(_read_wnid_folder('./synsets/person')) - set(_read_wnid_animals()))


def _read_dataset_summary(imagenet_folder):
    summary = {}
    summary.setdefault('B', _read_wnid_folder('./synsets/bird', 'B'))
    summary.setdefault('C', _read_wnid_folder('./synsets/cat', 'C'))
    summary.setdefault('D', _read_wnid_folder('./synsets/dog', 'D'))
    summary.setdefault('P', _read_wnid_folder('./synsets/person', 'P'))
    with open('./synsets/imagenet-animals.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) > 3 and row[3] in ['F', 'H', 'R', 'SA']:
                summary.setdefault(row[3], []).append(row[0])
    summary.setdefault('N', _read_negative_list(imagenet_folder))
    return summary


def _copy_images_for_class(cls, ids, copy_total, imagenet_folder, output_folder):
    print('- Class: {:5s} Subclasses: {:5d} '.format(cls, len(ids)))

    avg_total = int(math.floor(float(copy_total) / len(ids)))
    avg_train = int(math.floor(float(avg_total) * PERCENTAGE_FOR_TRAIN))
    avg_val = int(math.floor(float(avg_total) * PERCENTAGE_FOR_VALIDATION))
    avg_test = int(math.floor(float(avg_total) * PERCENTAGE_FOR_TEST))
    avg_total = avg_train + avg_val + avg_test
    assert avg_train >= SUBCLASS_IMAGES_MIN and avg_val >= SUBCLASS_IMAGES_MIN and avg_test >= SUBCLASS_IMAGES_MIN
    print('  Pick up {:4d} images from each subclass, {:4d} for training, {:4d} for validation, {:4d} for testing'
          .format(avg_total, avg_train, avg_val, avg_test))

    for wnid in ids:
        tar = os.path.join(imagenet_folder, wnid + '.tar')
        handle = tarfile.open(tar)
        files = handle.getmembers()
        if len(files) >= SUBCLASS_IMAGES_MIN:
            random.shuffle(files)
            copy_total = min(len(files), avg_total)
            if cls == 'N' and wnid in NEG_IMAGES_ENHANCED:
                copy_total = min(NEG_IMAGES_ENHANCED_MIN, len(files))
            copy_train = int(math.floor(float(copy_total) * PERCENTAGE_FOR_TRAIN))
            copy_val = int(math.floor(float(copy_total) * PERCENTAGE_FOR_VALIDATION))
            copy_test = int(math.floor(float(copy_total) * PERCENTAGE_FOR_TEST))

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
                handle.extract(item, dst_folder)
                print('  Extracted file: %s' % item.name)
                index += 1
        handle.close()


def _copy_images(cls, ids, imagenet_folder, output_folder):
    if key != 'N':
        _copy_images_for_class(cls, ids, POS_IMAGES_PER_CLASS, imagenet_folder, output_folder)
    else:
        _copy_images_for_class(cls, ids, NEG_IMAGES_PER_CLASS, imagenet_folder, output_folder)


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
