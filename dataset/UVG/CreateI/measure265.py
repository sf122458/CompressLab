import numpy
import sys
import re

def main():
    im_width = float(sys.argv[2])
    im_height = float(sys.argv[3])

    bpp_arr = []

    with open('ffreport.log') as f:
        lines = f.readlines()
    
    re_exp = re.compile(r" size ([0-9]+) ")
    size_line = []
    for l in lines:
        match_obj = re_exp.search(l)
        if match_obj:
            size_line.append(int(match_obj.group(1)))

    size_line = numpy.array(size_line)*8.0/(im_width*im_height)
    for i in range(len(size_line)):
        if i % 12 == 0:
            bpp_arr.append(size_line[i])

    print(sys.argv[1])
    print('bpp:' +str(numpy.array(bpp_arr).mean(0)))

if __name__ == '__main__':
    sys.exit(main())
