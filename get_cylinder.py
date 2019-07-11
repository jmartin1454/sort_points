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

from Arrow3D import *

# parse command line arguments

parser = OptionParser()

parser.add_option("-f","--file",dest="infile",
                  default="cylinder-surface.txt",
                  help="read data from file",metavar="FILE")

(options,args)=parser.parse_args()


with open(options.infile) as stream:
    d=np.loadtxt(stream,comments="%",unpack=True)

# geometry factors from COMSOL model
aout=1.8 # m
rcyl=0.5 # m
hcyl=1.0 # m

# resolution to identify difference in z and radius
resz=0.000000001

# get data into top, bottom, and sides of cylinder
dtop=d[:,abs(d[2]-hcyl/2)<resz]
x_top,y_top,z_top,u_top=dtop
dbot=d[:,abs(d[2]+hcyl/2)<resz]
x_bot,y_bot,z_bot,u_bot=dbot
dsid=d[:,abs(np.sqrt(d[0]**2+d[1]**2)-rcyl)<resz]
x_sid,y_sid,z_sid,u_sid=dsid
phi_sid=np.arctan2(y_sid,x_sid)
# arctan2 seems to place points at +pi; place them also at -pi to make square triangulation region
phi_pi=-phi_sid[abs(phi_sid-pi)<resz]
z_pi=z_sid[abs(phi_sid-pi)<resz]
u_pi=u_sid[abs(phi_sid-pi)<resz]
phi_sid=np.concatenate([phi_sid,phi_pi])
z_sid=np.concatenate([z_sid,z_pi])
u_sid=np.concatenate([u_sid,u_pi])

# make graphs

levels = np.arange(-10.05,10.05,.1)

fig,(ax_top,ax_bot,ax_sid)=plt.subplots(nrows=3)

ax_top.plot(x_top,y_top,'k.')
top_contours=ax_top.tricontour(x_top,y_top,u_top,levels=levels)
ax_top.axis((-rcyl,rcyl,-rcyl,rcyl))

ax_bot.plot(x_bot,y_bot,'k.')
bot_contours=ax_bot.tricontour(x_bot,y_bot,u_bot,levels=levels)
ax_bot.axis((-rcyl,rcyl,-rcyl,rcyl))

ax_sid.plot(phi_sid,z_sid,'k.')
sid_contours=ax_sid.tricontour(phi_sid,z_sid,u_sid,levels=levels)
ax_sid.axis((-pi,pi,-hcyl/2,hcyl/2))

all_levels=np.sort(np.unique(np.concatenate([top_contours.levels,bot_contours.levels,sid_contours.levels])))

# draw 3D coils
fig3 = plt.figure()
ax5 = fig3.add_subplot(111, projection='3d')
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
            z=[hcyl/2]*len(y)
            points=np.array(zip(x,y,z))
            mysegs.append(points)
    if (level in sid_contours.levels):
        print("%f is a sid level"%level)
        sid_index=np.where(sid_contours.levels==level)[0][0]
        for seg in sid_contours.allsegs[sid_index]:
            phi=seg[:,0]
            z=seg[:,1]
            x=rcyl*np.cos(phi)
            y=rcyl*np.sin(phi)
            points=np.array(zip(x,y,z))
            mysegs.append(points)
    if (level in bot_contours.levels):
        print("%f is a bot level"%level)
        bot_index=np.where(bot_contours.levels==level)[0][0]
        for seg in bot_contours.allsegs[bot_index]:
            x=seg[:,0]
            y=seg[:,1]
            z=[-hcyl/2]*len(y)
            x=np.flip(x,0)
            y=np.flip(y,0)
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
mycoilset.draw_coils(ax5)
plt.show()
