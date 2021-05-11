#!/usr/bin/python

# Sorting points by Jeff and Rosie
# Jeff updating to parse u3 and u2-u3 for double-cos box coil
# Jeff updated to read data from comsol mesh
# Jeff updated to triangulate and plot contours in matplotlib
# Jeff updated to extract contours
# June 16, 2019 Jeff updated to use patch.py classes
# July 10, 2019 Jeff updated for cylindrical inner surface and mu-metal outer surface

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

from scipy.optimize import curve_fit

# parse command line arguments

parser = OptionParser()

parser.add_option("-f","--file",dest="infile",
                  default="boxy-surface.txt",
                  help="read data from file",metavar="FILE")

parser.add_option("-m", "--mesh", dest="plotmesh",
                  default=False, action="store_true",
                  help="plot where the mesh points are")

parser.add_option("-i", "--current", dest="current", default=0.1,
                  help="design current (A) (default = 0.1)")

parser.add_option("-p", "--planes", dest="planes", default=False,
                  action="store_true",
                  help="show field maps in three cut planes")

parser.add_option("-t", "--traces", dest="traces", default=False,
                  action="store_true",
                  help="show 3D view of coil traces")

(options,args)=parser.parse_args()


with open(options.infile) as stream:
    d=np.loadtxt(stream,comments="%",unpack=True)

# geometry factors from COMSOL model
aout=1.8 # m
ain=1.0 # m

# resolution to identify difference in position
resz=0.000000001

# get data into top, bottom, and sides of box
dtop=d[:,abs(d[2]-ain/2)<resz]
x_top,y_top,z_top,u_top=dtop
dbot=d[:,abs(d[2]+ain/2)<resz]
x_bot,y_bot,z_bot,u_bot=dbot
# left and right (x)
dleft=d[:,abs(d[0]+ain/2)<resz]
x_left,y_left,z_left,u_left=dleft
dright=d[:,abs(d[0]-ain/2)<resz]
x_right,y_right,z_right,u_right=dright
# back and front (y)
dfront=d[:,abs(d[1]+ain/2)<resz]
x_front,y_front,z_front,u_front=dfront
dback=d[:,abs(d[1]-ain/2)<resz]
x_back,y_back,z_back,u_back=dback

# define desired design current

current = float(options.current) # amperes; design current = step in scalar potential
maxphi = 10 # amperes; biggest you can imagine the scalar potential to be
num = round(maxphi/current) # half the number of equipotentials
maxlevel = (2*num-1)*current/2
minlevel = -maxlevel
levels = np.arange(minlevel,maxlevel,current)


# make graphs of contours

fig,((ax_dummy1,ax_top,ax_dummy2,ax_dummy3),(ax_left,ax_front,ax_right,ax_back),(ax_dummy4,ax_bot,ax_dummy5,ax_dummy6))=plt.subplots(3,4)

if (options.plotmesh):
    ax_top.plot(x_top,y_top,'k.')
top_contours=ax_top.tricontour(x_top,y_top,u_top,levels=levels)
ax_top.axis((-ain/2,ain/2,-ain/2,ain/2))

if (options.plotmesh):
    ax_bot.plot(x_bot,y_bot,'k.')
bot_contours=ax_bot.tricontour(x_bot,y_bot,u_bot,levels=levels)
ax_bot.axis((-ain/2,ain/2,-ain/2,ain/2))

if (options.plotmesh):
    ax_left.plot(y_left,z_left,'k.')
left_contours=ax_left.tricontour(y_left,z_left,u_left,levels=levels)
ax_left.axis((-ain/2,ain/2,-ain/2,ain/2))

if (options.plotmesh):
    ax_right.plot(y_right,z_right,'k.')
right_contours=ax_right.tricontour(y_right,z_right,u_right,levels=levels)
ax_right.axis((-ain/2,ain/2,-ain/2,ain/2))

if (options.plotmesh):
    ax_front.plot(y_front,z_front,'k.')
front_contours=ax_front.tricontour(x_front,z_front,u_front,levels=levels)
ax_front.axis((-ain/2,ain/2,-ain/2,ain/2))

if (options.plotmesh):
    ax_back.plot(y_back,z_back,'k.')
back_contours=ax_back.tricontour(x_back,z_back,u_back,levels=levels)
ax_back.axis((-ain/2,ain/2,-ain/2,ain/2))

plt.show()

# arrange into coils

all_levels=np.sort(np.unique(np.concatenate([top_contours.levels,bot_contours.levels,left_contours.levels,right_contours.levels,front_contours.levels,back_contours.levels])))

mycoilset=coilset()
for level in all_levels:
    print(level)
    mysegs=[]
    if (level in top_contours.levels):
        print("%f is a top level"%level)
        top_index=np.where(top_contours.levels==level)[0][0]
        for seg in top_contours.allsegs[top_index]:
            x=seg[:,0]
            y=seg[:,1]
            z=[ain/2]*len(y)
            points=np.array(zip(x,y,z))
            mysegs.append(points)
    if (level in bot_contours.levels):
        print("%f is a bot level"%level)
        bot_index=np.where(bot_contours.levels==level)[0][0]
        for seg in bot_contours.allsegs[bot_index]:
            x=seg[:,0]
            y=seg[:,1]
            z=[-ain/2]*len(y)
            x=np.flip(x,0)
            y=np.flip(y,0)
            points=np.array(zip(x,y,z))
            mysegs.append(points)
    if (level in right_contours.levels):
        print("%f is a right level"%level)
        right_index=np.where(right_contours.levels==level)[0][0]
        for seg in right_contours.allsegs[right_index]:
            y=seg[:,0]
            z=seg[:,1]
            x=[ain/2]*len(z)
            points=np.array(zip(x,y,z))
            mysegs.append(points)
    if (level in left_contours.levels):
        print("%f is a left level"%level)
        left_index=np.where(left_contours.levels==level)[0][0]
        for seg in left_contours.allsegs[left_index]:
            y=seg[:,0]
            z=seg[:,1]
            x=[-ain/2]*len(z)
            y=np.flip(y,0)
            z=np.flip(z,0)
            points=np.array(zip(x,y,z))
            mysegs.append(points)
    if (level in back_contours.levels):
        print("%f is a back level"%level)
        back_index=np.where(back_contours.levels==level)[0][0]
        for seg in back_contours.allsegs[back_index]:
            x=seg[:,0]
            z=seg[:,1]
            y=[ain/2]*len(z)
            x=np.flip(x,0)
            z=np.flip(z,0)
            points=np.array(zip(x,y,z))
            mysegs.append(points)
    if (level in front_contours.levels):
        print("%f is a front level"%level)
        front_index=np.where(front_contours.levels==level)[0][0]
        for seg in front_contours.allsegs[front_index]:
            x=seg[:,0]
            z=seg[:,1]
            y=[-ain/2]*len(z)
            points=np.array(zip(x,y,z))
            mysegs.append(points)
    # arrange mysegs
    currentseg=0
    points=mysegs[currentseg]
    segsleft=np.arange(len(mysegs))
    segsleft=segsleft[segsleft!=currentseg]
    print(segsleft)
    while (len(segsleft>0)):
        # Find the smallest distance from the end of this segment:
        # either to its own start (meaning a closed loop), or
        # to the start of another segment.
        smallest=np.linalg.norm(points[-1]-points[0]) # consider this might be a closed loop
        sindex=-1
        print(points[-1],points[0])
        for seg in segsleft:
            print(mysegs[seg][0])
            distance=np.linalg.norm(points[-1]-mysegs[seg][0])
            if(distance<smallest):
                smallest=distance
                sindex=seg
        if(sindex!=-1):
            print('The correct index is %d'%sindex)
            points=np.concatenate([points,mysegs[sindex]])
            segsleft=segsleft[segsleft!=sindex]
            print(segsleft)
        else:
            print('Closed loop found!')
            mycoilset.add_coil(points)
            currentseg=segsleft[0] # move on to the next segment
            points=mysegs[currentseg]
            segsleft=segsleft[segsleft!=currentseg]
            print(segsleft)

    mycoilset.add_coil(points)

# draw 3D coils
if (options.traces):
    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111, projection='3d')
    mycoilset.draw_coils(ax5)
    #mycoilset.draw_coil(2,ax5,'-','black')
    mycoilset.output_scad('g.scad')
    plt.show()

mycoilset.set_common_current(current) # turn on the coils

min_field=-1.e-6
max_field=1.e-6
if (options.planes):
    figtest, (axtest1, axtest2, axtest3) = plt.subplots(nrows=3)

    x2d,y2d=np.mgrid[-1:1:100j,-1:1:100j]
    bx2d,by2d,bz2d=mycoilset.b_prime(x2d,y2d,0.)
    #im=axtest1.pcolormesh(x2d,y2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=min_field,vmax=max_field)
    im=axtest1.pcolormesh(x2d,y2d,bz2d,vmin=min_field,vmax=max_field)
    figtest.colorbar(im,ax=axtest1)

    x2d,z2d=np.mgrid[-1:1:100j,-1:1:100j]
    bx2d,by2d,bz2d=mycoilset.b_prime(x2d,0.,z2d)
    #im=axtest2.pcolormesh(z2d,x2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=min_field,vmax=max_field)
    im=axtest2.pcolormesh(z2d,x2d,bz2d,vmin=min_field,vmax=max_field)
    figtest.colorbar(im,ax=axtest2)

    y2d,z2d=np.mgrid[-1:1:100j,-1:1:100j]
    bx2d,by2d,bz2d=mycoilset.b_prime(0.,y2d,z2d)
    #im=axtest3.pcolormesh(z2d,y2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=min_field,vmax=max_field)
    im=axtest3.pcolormesh(z2d,y2d,bz2d,vmin=min_field,vmax=max_field)
    figtest.colorbar(im,ax=axtest3)

    plt.show()

fig7,(ax71)=plt.subplots(nrows=1)

def fiteven(x,p0,p2,p4,p6):
    return p0+p2*x**2+p4*x**4+p6*x**6

def fitodd(x,p1,p3,p5,p7):
    return p1*x+p3*x**3+p5*x**5+p7*x**7

def fitgraph(xdata,ydata,ax):
    popt,pcov=curve_fit(fitodd,xdata[abs(xdata)<.5],ydata[abs(xdata)<.5])
    print(popt)
    ax.plot(points1d,fitodd(xdata,*popt),'r--',label='$p_0$=%2.1e,$p_2$=%2.1e,$p_4$=%2.1e,$p_6$=%2.1e'%tuple(popt))

print('In case you are interested, 4*pi/10 is %f'%(4.*pi/10))

points1d=np.mgrid[-1:1:1001j]
bx1d,by1d,bz1d=mycoilset.b_prime(0.,points1d,0.)
fitgraph(points1d,bz1d,ax71)
ax71.plot(points1d,bz1d,label='$B_z(0,y,0)$')
bx1d,by1d,bz1d=mycoilset.b_prime(points1d,0.,0.)
fitgraph(points1d,bz1d,ax71)
ax71.plot(points1d,bz1d,label='$B_z(x,0,0)$')
bx1d,by1d,bz1d=mycoilset.b_prime(0.,0.,points1d)
fitgraph(points1d,bz1d,ax71)
ax71.plot(points1d,bz1d,label='$B_z(0,0,z)$')

ax71.axis((-.5,.5,min_field,max_field))
ax71.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax71.legend()

plt.show()
