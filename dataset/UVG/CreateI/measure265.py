import numpy
import math
import sys
import numpy
from scipy import signal
from scipy import ndimage
from skimage import io
import sys
import re
import math
import sys
import numpy
from scipy import signal
from scipy import ndimage

# import gauss
import matplotlib.pyplot as plt

#!/usr/bin/env python
"""Module providing functionality surrounding gaussian function.
"""
SVN_REVISION = '$LastChangedRevision: 16541 $'

import sys
import numpy

def main():
    im_width = float(sys.argv[2])
    im_height = float(sys.argv[3])

    psnr_arr = []
    msssim_arr = []
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
    import time
    for i in range(len(size_line)):
        if (i) % 12 == 0:
            psnr_val = 0
            tmpssim = 0
            ms_ssim_val = tmpssim/3.0

            psnr_arr.append(psnr_val)
            msssim_arr.append(ms_ssim_val)
            bpp_arr.append(size_line[i])

    print(sys.argv[1])

    print('psnr:' +  str(numpy.array(psnr_arr).mean(0)))
    print('bpp:' +str(numpy.array(bpp_arr).mean(0)))
    print('msssim:' +str(numpy.array(msssim_arr).mean(0)))

if __name__ == '__main__':
    sys.exit(main())
