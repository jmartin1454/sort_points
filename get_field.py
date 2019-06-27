#!/usr/bin/python

# Sorting points by Jeff and Rosie
# Jeff updating to parse u3 and u2-u3 for double-cos box coil
# Jeff updated to read data from comsol mesh
# Jeff updated to triangulate and plot contours in matplotlib
# Jeff updated to extract contours
# June 16, 2019 Jeff updated to use patch.py classes
# June 17, 2019 now working properly
# June 25, 2019 updates for different graphing

import numpy as np
import math
from optparse import OptionParser
import matplotlib
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

from mayavi import mlab

# parse command line arguments

parser = OptionParser()

parser.add_option("-f", "--file", dest="infile",
                  default="data.txt", help="read data from file",
                  metavar="FILE")

parser.add_option("-m", "--mesh", dest="plotmesh",
                  default=False, action="store_true",
                  help="plot where the mesh points are")

parser.add_option("-c", "--contours", dest="contours",
                  default=False, action="store_true",
                  help="show extracted contours")

parser.add_option("-i", "--current", dest="current", default=0.1,
                  help="design current (A) (default = 0.1)")

parser.add_option("-p", "--planes", dest="planes", default=False,
                  action="store_true",
                  help="show field maps in three cut planes")

parser.add_option("-t", "--traces", dest="traces", default=False,
                  action="store_true",
                  help="show 3D view of coil traces")

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

## refiner from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/tricontour_smooth_delaunay.html
#subdiv=3
#refiner = UniformTriRefiner(tri)
#tri_refi, u3_refi = refiner.refine_field(u3_outer, subdiv=subdiv)
#tri_refi, u23_refi = refiner.refine_field(u2_outer-u3_outer, subdiv=subdiv)
## refine inner coils too... seems to affect levels(?)
tri_inner = Triangulation(x_inner,y_inner)
##refiner_inner = UniformTriRefiner(tri_inner)
##tri_inner_refi, u1_refi = refiner.refine_field(u1_inner, subdiv=subdiv)


# scipy.interpolate attempt
#from scipy.interpolate import griddata
#grid_z2 = griddata(tri, u2_outer-u3_outer, (grid_x, grid_y), method='cubic')

# make graphs

current = float(options.current) # amperes; design current = step in scalar potential
maxphi = 10 # amperes; biggest you can imagine the scalar potential to be
num = round(maxphi/current) # half the number of equipotentials
maxlevel = (2*num-1)*current/2
minlevel = -maxlevel
levels = np.arange(minlevel,maxlevel,current)
print(levels)

fig, (ax1, ax2) = plt.subplots(nrows=2)

#ax1.triplot(tri, color='0.7') # if you want to see the triangulation

#u23_contours=ax1.tricontour(tri_refi,u23_refi,levels=levels)
u23_contours=ax1.tricontour(tri,u2_outer-u3_outer,levels=levels)
if (options.plotmesh):
    ax1.plot(x_outer, y_outer, 'ko', ms=1)
ax1.axis((0,a_out/2,0,a_out/2))
fig.colorbar(u23_contours,ax=ax1)

#u3_contours=ax2.tricontour(tri_refi, u3_refi, levels=levels)
u3_contours=ax2.tricontour(tri, u3_outer, levels=levels)
if (options.plotmesh):
    ax2.plot(x_outer, y_outer, 'ko', ms=1)

#u1_contours=ax2.tricontour(x_inner, y_inner, u1_inner, levels=levels)
u1_contours=ax2.tricontour(tri_inner, u1_inner, levels=levels)
if (options.plotmesh):
    ax2.plot(x_inner, y_inner, 'ko', ms=1)
ax2.axis((0,a_out/2, 0,a_out/2))
fig.colorbar(u1_contours,ax=ax2)

#plt.show()

## extracting all the contours and graphing them
if (options.contours):
    fig2, (ax3, ax4) = plt.subplots(nrows=2)

    # nseg=len(u23_contours.allsegs)

    for i,cnt in enumerate(u23_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        x=seg[:,0]
        y=seg[:,1]
        ax3.plot(x,y,'.-',color='black',ms=1)
    ax3.axis((0,a_out/2,0,a_out/2))

    for i,cnt in enumerate(u3_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        x=seg[:,0]
        y=seg[:,1]
        ax4.plot(x,y,'.-',color='black',ms=1)
    for i,cnt in enumerate(u1_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        x=seg[:,0]
        y=seg[:,1]
        ax4.plot(x,y,'.-',color='black',ms=1)
    ax4.axis((0,a_out/2,0,a_out/2))

    plt.show()
# conclusion:  the contours are all there and are ordered in the same way relative to each other

# assemble 3D coils

# rewrite using patch.py class library
mycoilset=coilset()

print("There are %d outer coils."%len(u23_contours.allsegs))

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
    points=np.array(zip(xnew,ynew,znew))
    mycoilset.add_coil(points)

    # reflect through xz-plane
    ynew=-ynew
    points=np.array(zip(xnew,ynew,znew))
    mycoilset.add_coil(points)
    if not mirrored:
        ynew=-ynew # put it back for a sec
        # reflect separate trace through yz-plane
        xnew=-xnew
        # reverse the windings
        xnew=np.flip(xnew,0)
        ynew=np.flip(ynew,0)
        znew=np.flip(znew,0)
        points=np.array(zip(xnew,ynew,znew))
        mycoilset.add_coil(points)
        # and through the xz-plane
        ynew=-ynew
        points=np.array(zip(xnew,ynew,znew))
        mycoilset.add_coil(points)

        
# now for the face plates

print("There are %d face coils."%len(u1_contours.allsegs))

for i,cnt in enumerate(u1_contours.allsegs):
    seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
    # these go from outer to inner
    x=seg[:,0]
    y=seg[:,1]
    # get the corresponding contour from u3
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
    points=np.array(zip(xnew,ynew,znew))
    mycoilset.add_coil(points)
    # mirror through xy-plane
    znew=[a_out/2]*len(ynew)
    xnew=np.flip(xnew,0)
    ynew=np.flip(ynew,0)
    points=np.array(zip(xnew,ynew,znew))
    mycoilset.add_coil(points)
    # mirror through xz-plane
    ynew=-ynew
    points=np.array(zip(xnew,ynew,znew))
    mycoilset.add_coil(points)
    znew=[-a_out/2]*len(ynew)
    xnew=np.flip(xnew,0)
    ynew=np.flip(ynew,0)    
    points=np.array(zip(xnew,ynew,znew))
    mycoilset.add_coil(points)

if (options.traces):
    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111, projection='3d')
    mycoilset.draw_coils(ax5)
    plt.show()


mycoilset.set_common_current(current)

design_field=4*pi/10*1.e-6
delta_field=5.e-9
min_field=design_field-delta_field
max_field=design_field+delta_field
print(design_field,delta_field)

if (options.planes):
    figtest, (axtest1, axtest2, axtest3) = plt.subplots(nrows=3)

    x2d,y2d=np.mgrid[-1:1:100j,-1:1:100j]
    bx2d,by2d,bz2d=mycoilset.b_prime(x2d,y2d,0.)
    im=axtest1.pcolormesh(x2d,y2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=min_field,vmax=max_field)
    #im=axtest1.pcolormesh(x2d,y2d,bx2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest1)

    x2d,z2d=np.mgrid[-1:1:100j,-1:1:100j]
    bx2d,by2d,bz2d=mycoilset.b_prime(x2d,0.,z2d)
    im=axtest2.pcolormesh(z2d,x2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=min_field,vmax=max_field)
    #im=axtest2.pcolormesh(z2d,x2d,by2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest2)

    y2d,z2d=np.mgrid[-1:1:100j,-1:1:100j]
    bx2d,by2d,bz2d=mycoilset.b_prime(0.,y2d,z2d)
    im=axtest3.pcolormesh(z2d,y2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=min_field,vmax=max_field)
    #im=axtest3.pcolormesh(z2d,y2d,by2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest3)

    plt.show()

fig7, (ax71) = plt.subplots(nrows=1)

points1d=np.mgrid[-1:1:101j]
bx1d,by1d,bz1d=mycoilset.b_prime(0.,points1d,0.)
ax71.plot(points1d,by1d)
bx1d,by1d,bz1d=mycoilset.b_prime(points1d,0.,0.)
ax71.plot(points1d,by1d)
bx1d,by1d,bz1d=mycoilset.b_prime(0.,0.,points1d)
ax71.plot(points1d,by1d)
ax71.axis((-.5,.5,-min_field,-max_field))

plt.show()
