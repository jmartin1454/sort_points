#!/usr/bin/python

# Sorting points by Jeff and Rosie
# Jeff updating to parse u3 and u2-u3 for double-cos box coil
# Jeff updated to read data from comsol mesh
# Jeff updated to triangulate and plot contours in matplotlib
# Jeff updated to extract contours
# June 16, 2019 Jeff updated to use patch.py classes

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

from patch import *

from mayavi.mlab import *

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

u23_contours=ax1.tricontour(tri,u2_outer-u3_outer,levels=levels)
ax1.plot(x_outer, y_outer, 'ko', ms=3)
ax1.axis((0,a_out/2,0,a_out/2))

u3_contours=ax2.tricontour(tri, u3_outer, levels=levels)
ax2.plot(x_outer, y_outer, 'ko', ms=3)

u1_contours=ax2.tricontour(x_inner, y_inner, u1_inner, levels=levels)
ax2.plot(x_inner, y_inner, 'ko', ms=3)
ax2.axis((0,a_out/2, 0,a_out/2))

#plt.show()

## extracting all the contours and graphing them
#
#fig2, (ax3, ax4) = plt.subplots(nrows=2)
#
## nseg=len(u23_contours.allsegs)
#
#for i,cnt in enumerate(u23_contours.allsegs):
#    seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
#    x=seg[:,0]
#    y=seg[:,1]
#    ax3.plot(x,y,'.-',color='black')
#ax3.axis((0,a_out/2,0,a_out/2))
#
#for i,cnt in enumerate(u3_contours.allsegs):
#    seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
#    x=seg[:,0]
#    y=seg[:,1]
#    ax4.plot(x,y,'.-',color='black')
#for i,cnt in enumerate(u1_contours.allsegs):
#    seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
#    x=seg[:,0]
#    y=seg[:,1]
#    ax4.plot(x,y,'.-',color='black')
#ax4.axis((0,a_out/2,0,a_out/2))
#
## conclusion:  the contours are all there and are ordered in the same way relative to each other

# draw 3D coils
fig3 = plt.figure()
ax5 = fig3.add_subplot(111, projection='3d')

# rewrite using patch.py class library

mycoilset=coilset()


for i,cnt in enumerate(u23_contours.allsegs):
    seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
    # these go from outer to inner
    x=seg[:,0]
    y=seg[:,1]
    z=[-a_out/2]*len(y)
    xs=np.flip(x,0)
    ys=np.flip(y,0)
    zs=[a_out/2]*len(y)
    xnew=np.concatenate((x,xs))
    ynew=np.concatenate((y,ys))
    znew=np.concatenate((z,zs))
    mirrored=False
    if (x[0]<0.0001):
        # mirror through yz-plane
        print("Mirroring")
        xnew=np.concatenate((xnew,-x))
        xnew=np.concatenate((xnew,-xs))
        ynew=np.concatenate((ynew,y))
        ynew=np.concatenate((ynew,ys))
        znew=np.concatenate((znew,zs))
        znew=np.concatenate((znew,z))
        mirrored=True
    else:
        # complete loop
        xnew=np.append(xnew,xnew[0])
        ynew=np.append(ynew,ynew[0])
        znew=np.append(znew,znew[0])
    ax5.plot(xnew,ynew,znew,'.-',color='black')
    if (i==0):
        points=np.array(zip(xnew,ynew,znew))
        mycoilset.add_coil(points)
        mycoilset.draw_coil(i,ax5)
        mycoilset.set_common_current(0.1)
        print(mycoilset.b(np.array([0.,0.,0.])))
        print(mycoilset.b_prime(0.,0.,0.))
        xg,yg,zg=np.mgrid[-3:3:7j,-3:3:7j,-3:3:7j]
        bxg,byg,bzg=mycoilset.b_prime(xg,yg,zg)
        figtest=plt.figure()
        ax=figtest.gca(projection='3d')
        ax.quiver(xg,yg,zg,bxg*1e8,byg*1e8,bzg*1e8)
        ##figtest=figure(bgcolor=(1.0,1.0,1.0),size=(400,400),fgcolor=(0, 0, 0))
        ##st = mayavi.mlab.flow(XX,YY,ZZ,xx,yy,zz,line_width=4,seedtype='sphere',integration_direction='forward') #sphere is the default seed type
        ##obj=flow(xg,yg,zg,bxg,byg,bzg)
        ##axes(extent = [-3.0,3.0,-3.0,3.0,-3.0,3.0]) #set plot bounds
        ##figtest.scene.z_plus_view() #adjust the view for a perspective along z (xy plane flat)
    # reflect through xz-plane
    ynew=-ynew
    ax5.plot(xnew,ynew,znew,'.-',color='red')
    if not mirrored:
        ynew=-ynew # put it back for a sec
        # reflect separate trace through yz-plane
        xnew=-xnew
        ax5.plot(xnew,ynew,znew,'.-',color='green')
        # and through the xz-plane
        ynew=-ynew
        ax5.plot(xnew,ynew,znew,'.-',color='blue')

# now for the face plates

fig4 = plt.figure()
ax6 = fig4.add_subplot(111, projection='3d')

for i,cnt in enumerate(u1_contours.allsegs):
    seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
    # these go from outer to inner
    x=seg[:,0]
    y=seg[:,1]
    # get the corresponding contour from u1
    seg=u3_contours.allsegs[i][0]
    xs=seg[:,0]
    ys=seg[:,1]
    xnew=np.concatenate((x,xs))
    ynew=np.concatenate((y,ys))
    # mirror through yz-plane
    xnew=np.concatenate((xnew,np.flip(-xs,0)))
    xnew=np.concatenate((xnew,np.flip(-x,0)))
    ynew=np.concatenate((ynew,np.flip(ys,0)))
    ynew=np.concatenate((ynew,np.flip(y,0)))
    znew=[-a_out/2]*len(ynew)
    ax6.plot(xnew,ynew,znew,'.-',color='black')
    # mirror through xy-plane
    znew=[a_out/2]*len(ynew)
    ax6.plot(xnew,ynew,znew,'.-',color='black')
    # mirror through xz-plane
    ynew=-ynew
    ax6.plot(xnew,ynew,znew,'.-',color='black')
    znew=[-a_out/2]*len(ynew)
    ax6.plot(xnew,ynew,znew,'.-',color='black')


plt.show()
