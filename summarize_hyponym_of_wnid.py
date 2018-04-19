import os
import tarfile
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


def _list_hyponym_sub(wnid, words, nodes, output):
    output[wnid] = words[wnid]
    if wnid in nodes:
        for node in nodes[wnid]:
            _list_hyponym_sub(node, words, nodes, output)


def _export_summary(imagenet_folder, wnid):
    items = {}
    words = _read_synset_words()
    nodes = _read_synset_nodes()
    _list_hyponym_sub(wnid, words, nodes, items)

    with open(('summary-%s.csv' % wnid), 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in items.iteritems():
            filepath = os.path.join(imagenet_folder, key + '.tar')
            if not os.path.exists(filepath):
                print("Tarball wasn't found at path: %s" % filepath)
            else:
                tar = tarfile.open(filepath)
                count = len(tar.getmembers())
                tar.close()
                writer.writerow([key, count, value])
                print("Found {:5d} images in tarball {:s}".format(count, filepath))


if __name__ == '__main__':
    if not (len(sys.argv) >= 3):
        print("Usage: summarize_hyponym_of_wnid.py imagenet_folder wnid")
    else:
        for wnid in sys.argv[1:]:
            _export_summary(sys.argv[1], wnid)
