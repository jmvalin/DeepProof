#!/usr/bin/python3

import random
import re
import numpy as np
from mistakes import *
import encoding
import h5py

maxlen = 300
minlen = 80
frac = .4

text = []

print("Computing lines")
for line in sys.stdin:
    if random.random() > frac:
        continue
    line = line.rstrip()
    if len(line) < minlen:
        continue
    if line.find('ISBN') >= 0:
        continue
    if line.find('University Press') >= 0:
        continue
    if re.match("^[0-9][0-9][0-9][0-9]", line):
        continue
    if re.match("\([0-9][0-9][0-9][0-9]\)", line):
        continue
    line = " ".join(line.split())
    line = line.replace("`", "'")
    orig_len = strlen = len(line)
    #print(line)
    chop_begin = chop_end = False
    if strlen > maxlen - 2:
        c = random.randrange(3)
        if c == 0:
            pos = 0
        elif c == 1:
            pos = strlen - maxlen + 2
        else:
            pos = random.randrange(strlen - maxlen + 2)
        if pos > 0 and line[pos-1] != ' ':
            pos = pos + line[pos:].find(' ') + 1
        chop_begin = True if pos > 0 else False
        line = line[pos:]
        strlen = len(line)
        if strlen > maxlen - 2:
            chop_end = True
            end = maxlen - 2
            while line[end] != ' ' and end > 0:
                end -= 1
            line = line[:end]
    if len(line) < minlen:
        continue

    orig = line;
    #print (orig)
    #continue
    line = word_substitute(line, verbs_rules, 0.15)
    line = word_substitute(line, homonyms_rules, 0.15)
    line = word_substitute(line, prepositions_rules, 0.15)
    line = word_substitute(line, misc_rules, 0.15)
    line = strip_plural(line, 0.15)
    line = add_plural(line, 0.005)
    line = letter_deletion(line, 0.005)
    line = letter_doubling(line, 0.005)
    line = letter_swap(line, 0.005)
    line = letter_subst(line, 0.002)
    if len(text) % 1000000 == 0:
        print(len(text))
    
    text.append((line, orig, chop_begin, chop_end, orig_len))
    #if len(text) > 1000:
    #    break
    print (line, '\t', orig)

print("Encoding lines")
input_text = np.zeros((len(text), maxlen), dtype='uint8')
output_text = np.zeros((len(text), maxlen), dtype='uint8')
for i, entry in enumerate(text):
    line, orig, chop_begin, chop_end, orig_len = entry
    byte_line = encoding.encode_string(line, maxlen, 0)
    byte_orig = encoding.encode_string(orig, maxlen, 0)
    if chop_begin:
        byte_line[0] = 1
        byte_orig[0] = 1
    if chop_end:
        byte_orig[len(orig)+1] = 1
        if len(line)+1 < maxlen:
            byte_line[len(line)+1] = 1
    input_text[i,:] = byte_line
    output_text[i,:] = byte_orig
    if i % 1000000 == 0:
        print(i)
    #print (orig_len, encoding.decode_string(byte_line), '\t', encoding.decode_string(byte_orig))
    #print()

h5f = h5py.File(sys.argv[1], 'w');
h5f.create_dataset('input', data=input_text)
h5f.create_dataset('output', data=output_text)
h5f.close()
