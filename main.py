import omfread
import numpy
import magn_d2fzdz2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename = '/home/fomin/oommf/ugol35/1x4_10x10v4a.omf'
[mx, my, mz, xnodes, ynodes, znodes, xstepsize, ystepsize, zstepsize] = omfread.read_bi4file(filename)

xstepsize=xstepsize*100#cm
ystepsize=ystepsize*100#cm
zstepsize=zstepsize*100#cm
zstep2=5e-7#cm

nZ=1

Z=nZ*zstep2
F= magn_d2fzdz2.f_d2fz_dz2(mx, my, mz, xnodes, ynodes, znodes, xstepsize, ystepsize, zstepsize, Z)

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.axes([0,0,1,1]) # Make the plot occupy the whole canvas
plt.axis('off')

imgplot = ax.imshow(F)
imgplot.set_cmap('gray')

# The figure can be saved
#fig.savefig('ToolPath2f.jpg')

plt.show()

