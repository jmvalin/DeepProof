
import numpy as np

maxord = 8192
#First three chars are special
#0: begin/end of sentence/paragraph
#1: truncated sentence/paragraph
#2: Unknown char
char_list = '|_~ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!"#$%&\'()*+,-./:;=?[]àéèêëïîìöôòûùỳçÀÉÈÊËÏÎÌÖÔÒÛÜÙỲÇ'

#not in list: < > @ \ ^ _ ` { | } ~ TAB

rev_list = dict(
    [(char, i) for i, char in enumerate(char_list[3:])])

charid = np.zeros(maxord+1, dtype='uint8') + 2
for i, char in enumerate(char_list):
    charid[ord(char)] = i


def encode_string(string, outlen, offset):
    strlen = len(string)
    out = np.zeros((outlen,), dtype='uint8')
    out[outlen-1] = 1 if strlen > outlen - 2 else 0
    out[0] = 1 if offset != 0 else 0
        
    copylen = min(strlen,outlen - 2)
    for i, char in enumerate(string[offset:copylen+offset]):
        out[i+1] = charid[min(maxord,ord(char))]
    return out
    
def decode_string(enc):
    out = ''.join([char_list[x] for x in enc])
    return out
