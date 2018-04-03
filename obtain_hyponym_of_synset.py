import os
import tarfile
import shutil
import sys
import csv


def _read_synset_words():
    with open('words.txt', mode='r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        return {rows[0]: rows[1] for rows in reader}


def _read_synset_nodes():
    dic = {}
    with open('wordnet.is_a.txt', mode='r') as infile:
        reader = csv.reader(infile, delimiter=' ')
        for rows in reader:
            dic.setdefault(rows[0], []).append(rows[1])
    return dic


def _list_hyponym(wnid):
    output = {}
    _list_hyponym_sub(
        wnid, _read_synset_words(), _read_synset_nodes(), output
    )
    return output


def _list_hyponym_sub(wnid, words, nodes, output):
    output[wnid] = words[wnid]
    if wnid in nodes:
        for node in nodes[wnid]:
            _list_hyponym_sub(node, words, nodes, output)


def _extract_tars(dataset, wnid, nodes):
    path = os.path.join(os.getcwd(), wnid)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for key, value in nodes.iteritems():
        tar = os.path.join(dataset, key + '.tar')
        if not os.path.exists(tar):
            print("Tarball wasn't found at path: %s" % tar)
        else:
            dest = os.path.join(path, key + " " + value)
            os.makedirs(dest)
            print("Extract tarball to folder: %s" % dest)
            handle = tarfile.open(tar)
            handle.extractall(dest)


if __name__ == '__main__':
    if not (len(sys.argv) == 3):
        print("Usage: obtain_hyponym_of_synset.py dataset_folder wnid")

    if len(sys.argv) == 3:
        nodes = _list_hyponym(sys.argv[2])
        _extract_tars(sys.argv[1], sys.argv[2], nodes)
