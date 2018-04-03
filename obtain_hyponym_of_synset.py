import os
import tarfile
import datetime
import shutil
import sys
import urllib2
import csv

BASE_URL = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=%s"

def _read_synset_csv():
    with open('synsets.csv', mode='r') as infile:
        reader = csv.reader(infile)
        return {rows[0]: rows[1] for rows in reader}


def _list_hyponym(dict, wnid, output={}):
    output[wnid] = dict[wnid]
    contents = urllib2.urlopen(BASE_URL % wnid ).read()
    lines = contents.splitlines()
    for line in lines:
        if line.startswith('-'):
            wnid = line[1:]
            output = _list_hyponym(dict, wnid, output)
    return output


def _extract_tars(dataset, wnid, dict):
    dir = os.path.join(os.getcwd(), wnid)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    for key, value in dict.iteritems():
        tar = os.path.join(dataset, key + '.tar')
        if not os.path.exists(tar):
            print("Tarball wasn't found at path: %s" % tar)
        else:
            dir = os.path.join(dir, key + " " + value)
            os.makedirs(dir)
            tar = tarfile.open(tar)
            tar.extractall(dir)


if __name__ == '__main__':
    if not (len(sys.argv) == 3):
        print("Usage: obtain_hyponym_of_synset.py dataset_folder wnid")

    if len(sys.argv) == 3:
        dict = _read_synset_csv()
        dict = _list_hyponym(dict, sys.argv[2])
        _extract_tars(sys.argv[1], sys.argv[2], dict)
