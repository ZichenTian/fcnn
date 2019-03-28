import os
import sys

def split_line(line):
    sub_str = line.split(', ')
    sub_str[-1] = sub_str[-1].strip()
    result = {}
    need_rid = ['MFLOPs', 'ms']
    for kv in sub_str:
        key, value = kv.split('=')
        for rid in need_rid:
            value = value.replace(rid, '')
        result[key] = value
    return result

def calc_average(logfile, keryword):
    with open(logfile, 'rb') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        results.append(split_line(line))
    value_sum = 0
    cnt = 0
    for result in results:
        value_sum += float(result[keryword])
        cnt += 1
    return value_sum / cnt


def main():
    if len(sys.argv) != 3:
        print 'Usage: python2 calc_average.py logfile keyword'
    else: 
        logfile = sys.argv[1]
        keyword = sys.argv[2]
        print calc_average(logfile, keyword)

if __name__ == '__main__':
    main()
