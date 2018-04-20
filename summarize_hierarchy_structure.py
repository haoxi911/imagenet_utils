# Code to print the full hierarchy of all wnids in the ImageNet tree

import os
import csv
import shutil


def _read_synset_words():
    with open('words.txt', mode='r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        return {rows[0]: rows[1] for rows in reader}


def _read_synset_structure():
    nodes = {}
    topnodes = []  # the top level nodes
    children = []  # the scanned child nodes
    with open('wordnet.is_a.txt', mode='r') as infile:
        reader = csv.reader(infile, delimiter=' ')
        for rows in reader:
            nodes.setdefault(rows[0], []).append(rows[1])
            if rows[0] not in topnodes and rows[0] not in children:
                topnodes.append(rows[0])
            children.append(rows[1])
            if rows[1] in topnodes:
                topnodes.remove(rows[1])
    return nodes, topnodes


def _print_synset_node(node, nodes, level):
    if node in nodes:
        for wnid in nodes[node]:
            word = wnid + ' ' + words[wnid]
            for x in xrange(0, level):
                word = '\t%s' % word
            with open('synsets.txt', 'a') as the_file:
                the_file.write(word + '\n')
            _print_synset_node(wnid, nodes, level + 1)


words = _read_synset_words()
nodes, roots = _read_synset_structure()
print roots
for wnid in roots:
    _print_synset_node(wnid, nodes, 0)
