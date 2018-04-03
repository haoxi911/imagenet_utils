import os
import tarfile
import sys
import urllib2
import csv

BASE_URL = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=%s"

def _read_synset_csv():
    with open('synsets.csv', mode='r') as infile:
        reader = csv.reader(infile)
        return {rows[0]: rows[1] for rows in reader}


def _list_hyponym(_dict, wnid):
    print wnid + ' ' + _dict[wnid]

    contents = urllib2.urlopen(BASE_URL % wnid ).read()
    lines = contents.splitlines()
    for line in lines:
        if line.startswith('-'):
            wnid = line[1:]
            _list_hyponym(_dict, wnid)


if __name__ == '__main__':
    if not (len(sys.argv) == 3):
        print("Usage: obtain_hyponym_of_synset.py dataset_folder wnid")

    if len(sys.argv) == 3:
        _dict = _read_synset_csv()
        _list_hyponym(_dict, sys.argv[1])

