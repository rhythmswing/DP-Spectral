#!/usr/bin/python3 -W ignore
import sys
import getopt
from laplacian import *

if __name__ == '__main__':

    longs = ['input=', 'output=', 'beta=', 'dim=']
    shorts = ':'

    beta = 0
    input = ''
    algorithm = ''
    dimension = 50
    output = ''

    opts, args = getopt.getopt(sys.argv[1:], shorts, longs)

    for o, a in opts:
        if o == '--input':
            input = a
        if o == '--beta':
            beta = float(a)
        if o == '--dim':
            dimension = int(a)
        if o == '--output':
            output = a

    if input == '':
        print('graph input unspecified')
        exit()
    if output == '':
        output = 'embedding.txt'
    if beta == '':
        print('beta unspecified; default: beta=0')
        beta = 0

    heading, edgelist = load_edgelist(input)
    adj = edgelist_to_adjacency_sparse(edgelist, heading)
    print('beta = {0}'.format(beta))
    embed, W = laplacian_eigenmap_sparse(adj, dimension, beta)

    heading = [heading[0], dimension]
    write_embedding(embed, heading, output)
