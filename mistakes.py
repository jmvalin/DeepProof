# -*- coding: utf-8 -*-
import sys
import random
import math
import re
from irregular import irregular_verbs
'''
verbs_rules = [["has", "have", "had"],
               ["was", "were", "are", "is"],
               ["go", "gone", "went", "goes"],
               ["be", "been"],
               ["know", "knew"],
               ["do", "did", "does"],
               ["understand", "understood"]
              ]
'''
acceptable_contractions = [["it is", "it's"],
                           ["I am", "I'm"],
                           ["you are", "you're"],
                           ["he is", "he's"],
                           ["she is", "she's"],
                           ["we are", "we're"],
                           ["they are", "they're"],
                           ["cannot", "can't"],
                           ["do not", "don't"],
                           ["does not", "doesn't"],
                           ["did not", "didn't"],
                           ["should not", "shouldn't"],
                           ["will not", "won't"]
                          ]

homonyms_rules = [["there", "their", "they're"],
                  ["to", "too", "two"],
                  ["break", "brake"],
                  ["its", "it's"],
                  ["then", "than"],
                  ["which", "witch"],
                  ["here", "hear"],
                  ["weather", "whether"],
                  ["bear", "bare"],
                  ["fore", "for", "four"],
                  ["meet", "meat"],
                  ["wear", "where", "ware"],
                  ["week", "weak"],
                  ["wait", "weight"],
                  ["waste", "waist"],
                  ["sweet", "suite", "sweat"],
                  ["steel", "steal"],
                  ["steak", "stake"],
                  ["sun", "son"],
                  ["no", "know"],
                  ["mail", "male"],
                  ["light", "lite"],
                  ["hole", "whole"],
                  ["maid", "made"],
                  ["fair", "fare"],
                  ["write", "right"],
                  ["advise", "advice"]
                 ]

prepositions_rules = [["to", "at", "in", "for"],
                      ["out", "off"],
                      ["on", "over"],
                      ["since", "for"],
                      ["from", "than"],
                      ["on", "at"]
                     ]

misc_rules = [["the", "a", "an"],
              ["you", "your", "you're", "yours"],
              ["I", "me", "my", "mine"],
              ["he", "him", "his"],
              ["she", "her", "hers"],
              ["this", "that"],
              ["excepted", "accepted"],
              ["affect", "effect"],
              ["affects", "effects"],
              ["affected", "effected"],
              ["your", "you're"],
              ["who", "that"],
              ["who", "whom", "whose", "who's"],
              ["in to", "into"],
              ["lose", "loose"],
              ["an", "and"],
              ["are", "our"],
              ["not", "now"],
              ["I", "i"],
              ["thing", "think"],
              ["complains", "complaints"],
              ["now", "know"],
              ["exit", "exist"]
             ]

omitted_words = ["the", "a", "an", "to", "on", "of", "is"]

subword_subst = [["ea", "ee"],
                 ["oo", "ou"],
                 ["gth", "ght"],
                 ["an", "en"],
                 ["on", "un"],
                 ["er", "ar"]
                ]

#these are adjacent on a querty keyboard
#adjacent_list = "poiuytrewqasdfghjkl.,mnbvcxz"

irregular_rules = []
for verb in irregular_verbs:
    present = verb[0]
    if present[-1] == 'e':
        badpast = present + 'd'
    else:
        badpast = present + 'ed'
    if verb[-1] == verb[-2]:
        verb = verb[:-1]
    if verb[-1] == verb[-2]:
        verb = verb[:-1]
    verb = verb + [badpast]
    if verb[-1] == verb[-2]:
        verb = verb[:-1]
    irregular_rules = irregular_rules + [verb]
    #print(verb);

#print(irregular_rules)

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

def word_delete(line, rules, prob):
    for word in rules:
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
                line = line[:where] + line[(where+word_len):]
            where += word_len
    return line
        
def subword_substitute(line, rules, prob):
    for group in rules:
        for subword in group:
            word_len = len(subword)
            where = 0
            while True:
                pos = line[where:].find(subword)
                if pos < 0:
                    break
                where = where + pos
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
