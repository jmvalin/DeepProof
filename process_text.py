#!/usr/bin/python3

from mistakes import *

for line in sys.stdin:
    line = line.rstrip()
    if len(line) > 90:
        continue
    orig = line;
    line = word_substitute(line, verbs_rules, 0.2)
    line = word_substitute(line, homonyms_rules, 0.2)
    line = word_substitute(line, prepositions_rules, 0.2)
    line = word_substitute(line, misc_rules, 0.2)
    line = letter_deletion(line, 0.01)
    line = letter_doubling(line, 0.01)
    line = letter_swap(line, 0.01)
    
    print (line, '\t', orig)
