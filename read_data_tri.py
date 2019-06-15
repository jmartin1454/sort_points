#!/usr/bin/python

# Sorting points by Jeff and Rosie
# Jeff updating to parse u3 and u2-u3 for double-cos box coil


import numpy as np
import math
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from scipy import interpolate 
import io

from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.cm as cm

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# parse command line arguments

parser = OptionParser()

parser.add_option("-f", "--file", dest="infile",
                  default="data.txt", help="read data from file", metavar="FILE")

(options, args) = parser.parse_args()


with open(options.infile) as stream:
    d=np.loadtxt(stream,comments="%",unpack=True)

du1=d[:,(d[2]<100)] # remove NaN's in u1
x_inner,y_inner,u1_inner,u2_inner,u3_inner=du1

du2=d[:,(d[3]<100)] # remove NaN's in u2
x_outer,y_outer,u1_outer,u2_outer,u3_outer=du2

# geometry factors from COMSOL model
a_out = 1.8 # m
a_in = 1.4 # m

# mask out bad triangles that will be created when automatically triangulating outer (concave) region.

polygon_inner = Polygon([(0,0),(a_in/2,0),(a_in/2,a_in/2),(0,a_in/2)])
polygon_outer = Polygon([(a_in/2,0),(a_out/2,0),(a_out/2,a_out/2),(0,a_out/2),(0,a_in/2),(a_in/2,a_in/2)])

tri = Triangulation(x_outer,y_outer)
ntri = tri.triangles.shape[0]

# example of masking a region from https://matplotlib.org/examples/pylab_examples/tripcolor_demo.html

xmid = x_outer[tri.triangles].mean(axis=1)
ymid = y_outer[tri.triangles].mean(axis=1) # finds the center points of each triangle
mask = np.zeros(ntri, dtype=bool)
i=0
for x,y in zip(xmid,ymid):
    if not polygon_outer.contains(Point(x,y)):
        mask[i] = True
    i=i+1
print(mask)
tri.set_mask(mask)

# make graphs

levels = np.arange(-10.05,10.05,.1)

fig, (ax1, ax2) = plt.subplots(nrows=2)


#ax1.triplot(tri, color='0.7') # if you want to see the triangulation

contours=ax1.tricontour(tri,u2_outer-u3_outer,levels=levels)
print(contours.allsegs)
x,y=contours.allsegs[1][0][0]
ax1.plot(x_outer, y_outer, 'ko', ms=3)
ax1.axis((0,a_out/2,0,a_out/2))

ax2.tricontour(tri, u3_outer, levels=levels)
ax2.plot(x_outer, y_outer, 'ko', ms=3)


ax2.tricontour(x_inner, y_inner, u1_inner, levels=levels)
ax2.plot(x_inner, y_inner, 'ko', ms=3)
ax2.axis((0,a_out/2, 0,a_out/2))

plt.show()
