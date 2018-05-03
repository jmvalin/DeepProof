# -*- coding: utf-8 -*-
import sys
import random
import math
import re

verbs_rules = [["has", "have", "had"],
               ["was", "were", "are", "is"],
               ["go", "gone", "went", "goes"],
               ["be", "been"]
              ]

homonyms_rules = [["there", "their", "they're"],
                  ["to", "too", "two"],
                  ["break", "brake"],
                  ["its", "it's"],
                  ["then", "than"]
                 ]

prepositions_rules = [["to", "at", "in"],
                      ["out", "off"],
                      ["on", "over"]
                     ]

misc_rules = [["the", "a"],
              ["you", "your"],
              ["I", "me"],
              ["this", "that"],
              ["excepted", "accepted"],
              ["affect", "effect"],
              ["affects", "effects"],
              ["affected", "effected"],
              ["your", "you're"],
              ["who", "that"],
              ["who", "whom", "whose", "who's"],
              ["in to", "into"],
              ["lose", "loose"]
             ]

#these are adjacent on a querty keyboard
#adjacent_list = "poiuytrewqasdfghjkl.,mnbvcxz"




def word_substitute(line, rules, prob):
    for group in rules:
        for word in group:
            word_len = len(word)
            where = 0
            while True:
                pos = line[where:].find(word)
                if pos < 0:
                    break
                where = where + pos
                if (where > 0 and line[where-1] != ' ') or (where+word_len < len(line) and line[where+word_len] != ' '):
                    where += 1
                    continue
                if random.random() < prob:
                    subst = random.choice(group)
                    line = line[:where] + subst + line[(where+word_len):]
                where += word_len
    return line

def strip_plural(line, prob):
    where = 0
    while True:
        pos = re.search("[a-zA-Z]s[ ,.;:$]", line[where:])
        if pos:
            pos = pos.start()
        else:
            break
        where += pos
        if random.random() < prob:
            line = line[:where+1] + line[where+2:]
        where += 2
    return line

def add_plural(line, prob):
    where = 0
    while True:
        pos = re.search("[a-zA-Z][ ,.;:$]", line[where:])
        if pos:
            pos = pos.start()
        else:
            break
        where += pos
        if random.random() < prob:
            line = line[:where+1] + 's' + line[where+1:]
        where += 4
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

def letter_subst(line, prob):
    line_len = len(line)
    pos = 0
    prob_1 = 1./prob
    while pos < line_len:
        uni = random.random()
        pos = pos - int(prob_1*math.log(.00001 + uni))
        if pos >= line_len:
            break
        line = line[:pos] + chr(32 + random.randrange(95)) + line[pos+1:]
    return line
