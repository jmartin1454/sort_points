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
                  default="cylinder-surface.txt",
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
rcyl=0.5 # m
hcyl=1.0 # m

# resolution to identify difference in z and radius
resz=0.000000001
resr=0.00001 # This seems to be the minimum allowable resolution in
             # radius before I start losing points; thank you COMSOL.
             # Another method might be to take the points that aren't
             # on top and bottom, but I'd still want a nice row of
             # points along the top and bottom of the sides.

# get data into top, bottom, and sides of cylinder
dtop=d[:,abs(d[2]-hcyl/2)<resz]
x_top,y_top,z_top,u_top=dtop
dbot=d[:,abs(d[2]+hcyl/2)<resz]
x_bot,y_bot,z_bot,u_bot=dbot
dsid=d[:,abs(np.sqrt(d[0]**2+d[1]**2)-rcyl)<resr]
x_sid,y_sid,z_sid,u_sid=dsid
phi_sid=np.arctan2(y_sid,x_sid)
# arctan2 seems to place points at +pi; place them also at -pi to make square triangulation region
phi_pi=-phi_sid[abs(phi_sid-pi)<resz]
z_pi=z_sid[abs(phi_sid-pi)<resz]
u_pi=u_sid[abs(phi_sid-pi)<resz]
phi_sid=np.concatenate([phi_sid,phi_pi])
z_sid=np.concatenate([z_sid,z_pi])
u_sid=np.concatenate([u_sid,u_pi])
print(len(x_top),len(x_bot),len(x_sid),len(x_top)+len(x_bot)+len(x_sid))

# define desired design current

current = float(options.current) # amperes; design current = step in scalar potential
maxphi = 10 # amperes; biggest you can imagine the scalar potential to be
num = round(maxphi/current) # half the number of equipotentials
maxlevel = (2*num-1)*current/2
minlevel = -maxlevel
levels = np.arange(minlevel,maxlevel,current)


# make graphs of contours

fig,(ax_top,ax_sid,ax_bot)=plt.subplots(nrows=3)

if (options.plotmesh):
    ax_top.plot(x_top,y_top,'k.')
top_contours=ax_top.tricontour(x_top,y_top,u_top,levels=levels)
ax_top.axis((-rcyl,rcyl,-rcyl,rcyl))

if (options.plotmesh):
    ax_sid.plot(phi_sid,z_sid,'k.')
sid_contours=ax_sid.tricontour(phi_sid,z_sid,u_sid,levels=levels)
ax_sid.axis((-pi,pi,-hcyl/2,hcyl/2))

if (options.plotmesh):
    ax_bot.plot(x_bot,y_bot,'k.')
bot_contours=ax_bot.tricontour(x_bot,y_bot,u_bot,levels=levels)
ax_bot.axis((-rcyl,rcyl,-rcyl,rcyl))

# arrange into coils

all_levels=np.sort(np.unique(np.concatenate([top_contours.levels,bot_contours.levels,sid_contours.levels])))

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

# draw 3D coils
if (options.traces):
    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111, projection='3d')
    mycoilset.draw_coils(ax5)
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
    popt,pcov=curve_fit(fiteven,xdata[abs(xdata)<.5],ydata[abs(xdata)<.5])
    print(popt)
    ax.plot(points1d,fiteven(xdata,*popt),'r--',label='$p_0$=%2.1e,$p_2$=%2.1e,$p_4$=%2.1e,$p_6$=%2.1e'%tuple(popt))

print('In case you are interested, 4*pi/10 is %f'%(4.*pi/10))

points1d=np.mgrid[-1:1:101j]
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
