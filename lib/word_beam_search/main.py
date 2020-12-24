from __future__ import division
from __future__ import print_function

import editdistance

from word_beam_search import wordBeamSearch

# Settings
useNGrams = True

# main
if __name__ == '__main__':
    # decode matrix
    res = wordBeamSearch(data.mat, 10, loader.lm, useNGrams)
    print('Result:       "' + res + '"')
    print('Ground Truth: "' + data.gt + '"')
    strEditDist = str(editdistance.eval(res, data.gt))
    print('Editdistance: ' + strEditDist)
