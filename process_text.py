#!/usr/bin/python3

import sys
import random
import math

verbs_rules = [["has", "have", "had"],
               ["was", "were", "are", "is"]
              ]

homonyms_rules = [["there", "their"],
                  ["to", "too", "two"],
                 ]

def word_substitute(line, rules, prob):
    for group in rules:
        for word in group:
            word_len = len(word)
            while True:
                where = line.find(" " + word + " ")
                if where < 0:
                    break
                where = where + 1
                if random.random() < prob:
                    subst = random.choice(group)
                    line = line[:where] + subst + line[(where+word_len):]
                #fixme: keep iterating
                break
    return line

def letter_deletion(line, prob):
    line_len = len(line)
    pos = 0
    prob_1 = 1./prob
    while pos < line_len:
        uni = random.random()
        pos = pos - int(prob_1*math.log(.00001 + uni))
        if pos >= line_len:
            break
        line = line[:pos] + line[(pos+1):]
        line_len = len(line)
    return line
        

def letter_doubling(line, prob):
    line_len = len(line)
    pos = 0
    prob_1 = 1./prob
    while pos < line_len:
        uni = random.random()
        pos = pos - int(prob_1*math.log(.00001 + uni))
        if pos >= line_len:
            break
        line = line[:pos] + line[pos] + line[pos:]
        line_len = len(line)
    return line

def letter_swap(line, prob):
    line_len = len(line)
    pos = 1
    prob_1 = 1./prob
    while pos < line_len:
        uni = random.random()
        pos = pos - int(prob_1*math.log(.00001 + uni))
        if pos >= line_len:
            break
        line = line[:(pos-1)] + line[pos] + line[pos-1] + line[(pos+1):]
        line_len = len(line)
    return line

for line in sys.stdin:
    line = line.rstrip()
    if len(line) > 90:
        continue
    orig = line;
    line = word_substitute(line, verbs_rules, 0.2)
    line = word_substitute(line, homonyms_rules, 0.2)
    line = letter_deletion(line, 0.01)
    line = letter_doubling(line, 0.01)
    line = letter_swap(line, 0.01)
    
    print (line, '\t', orig)
