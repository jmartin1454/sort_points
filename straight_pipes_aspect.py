#!/usr/bin/python

# Sorting points by Jeff and Rosie
# Jeff updating to parse u3 and u2-u3 for double-cos box coil
# Jeff updated to read data from comsol mesh
# Jeff updated to triangulate and plot contours in matplotlib
# Jeff updated to extract contours
# June 16, 2019 Jeff updated to use patch.py classes
# June 17, 2019 now working properly
# June 25, 2019 updates for different graphing
# February 27, 2021 Added rerouting for side pipes
# May 10, 2021 The straight pipes

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

# line simplification

from simplification.cutil import simplify_coords, simplify_coords_vw

# parse command line arguments

parser = OptionParser()

parser.add_option("-i", "--current", dest="current", default=0.1,
                  help="design current (A) (default = 0.1)")

parser.add_option("-p", "--planes", dest="planes", default=False,
                  action="store_true",
                  help="show field maps in three cut planes")

parser.add_option("-t", "--traces", dest="traces", default=False,
                  action="store_true",
                  help="show 3D view of coil traces")

parser.add_option("-w", "--wiggle", dest="wiggle",
                  default=-1, help="sigma to wiggle each point (m)",
                  metavar="sigma")


(options, args) = parser.parse_args()

current=float(options.current)

# geometry factors
a_out_z = 2.3 # m
a_out_y = 2.2 # m
a_out_x = 2.3 # m
a_in_y = 1.8 # m
a_in_x = 1.8 # m


body_coil=coilset()

# Design H is 1 A/m
body_spacing=current

def reflect_y(points,flip):
    newpoints=np.copy(points)
    newpoints[:,1]=-newpoints[:,1]
    if(flip):
        newpoints=np.flip(newpoints,0) # wind them in the opposite direction
    return newpoints

def reflect_x(points,flip):
    newpoints=np.copy(points)
    newpoints[:,0]=-newpoints[:,0]
    if(flip):
        newpoints=np.flip(newpoints,0) # wind them in the opposite direction
    return newpoints

for y in arange(body_spacing/2,a_out_y/2,body_spacing):
    if(y<a_in_y/2):
        point1=(a_in_x/2,y,a_out_z/2)
        point2=(-a_in_x/2,y,a_out_z/2)
        point3=(-a_in_x/2,y,-a_out_z/2)
        point4=(a_in_x/2,y,-a_out_z/2)
    else:
        xpos=a_in_x/2-(y-a_in_y/2)/(a_out_y/2-a_in_y/2)*a_in_x/2
        point1=(xpos,y,a_out_z/2)
        point2=(-xpos,y,a_out_z/2)
        point3=(-xpos,y,-a_out_z/2)
        point4=(xpos,y,-a_out_z/2)
    points=np.array((point1,point2,point3,point4))
    body_coil.add_coil(points)
    body_coil.add_coil(reflect_y(points,False))
    print(points)
    

side_spacing=current*(a_out_x-a_in_x)/a_in_x

side_coil=coilset()
for y in arange(side_spacing/2,a_out_y/2,side_spacing):
    print(y)
    if(y<a_in_y/2):
        point1=(a_in_x/2,y,a_out_z/2)
        point2=(a_out_x/2,y,a_out_z/2)
        point3=(a_out_x/2,y,-a_out_z/2)
        point4=(a_in_x/2,y,-a_out_z/2)
    else:
        xpos=a_in_x/2+(y-a_in_y/2)/(a_out_y/2-a_in_y/2)*(a_out_x/2-a_in_x/2)
        point1=(xpos,y,a_out_z/2)
        point2=(a_out_x/2,y,a_out_z/2)
        point3=(a_out_x/2,y,-a_out_z/2)
        point4=(xpos,y,-a_out_z/2)
    points=np.array((point1,point2,point3,point4))
    side_coil.add_coil(points)
    side_coil.add_coil(reflect_x(points,True))
    side_coil.add_coil(reflect_y(points,False))
    side_coil.add_coil(reflect_x(reflect_y(points,False),True))
    print(points)
    
top_spacing=current*(a_out_y-a_in_y)/a_in_x
top_coil=coilset()
for x in arange(top_spacing/2,a_out_x/2,top_spacing):
    if(x<a_in_x/2):
        ypos=a_in_y/2-(x-a_in_x/2)/(a_in_x/2)*(a_out_y/2-a_in_y/2)
    else:
        ypos=a_in_y/2+(x-a_in_x/2)/(a_out_x/2-a_in_x/2)*(a_out_y/2-a_in_y/2)
    point1=(x,a_out_y/2,a_out_z/2)
    point2=(x,a_out_y/2,-a_out_z/2)
    point3=(x,ypos,-a_out_z/2)
    point4=(x,ypos,a_out_z/2)
    points=np.array((point1,point2,point3,point4))
    top_coil.add_coil(points)
    top_coil.add_coil(reflect_x(points,True))
    top_coil.add_coil(reflect_y(points,False))
    top_coil.add_coil(reflect_y(reflect_x(points,True),False))
    print(points)

print("There are %d body coils"%body_coil.ncoils)
print("There are %d side coils"%side_coil.ncoils)
print("There are %d top coils"%top_coil.ncoils)

if(options.wiggle>0):
    body_coil.wiggle(float(options.wiggle))
    front_face_coil.wiggle(float(options.wiggle))
    back_face_coil.wiggle(float(options.wiggle))

if(options.traces):
    fig4 = plt.figure()
    ax6 = fig4.gca()
    body_coil.draw_xy(ax6,'-','black')
    side_coil.draw_xy(ax6,'-','blue')
    top_coil.draw_xy(ax6,'-','green')

    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111, projection='3d')
    side_coil.draw_coils(ax5,'-','blue')
    top_coil.draw_coils(ax5,'-','green')
    body_coil.draw_coils(ax5,'-','black')
    
    body_coil.output_solidworks('body_coil_point_cloud.txt')
    side_coil.output_solidworks('side_coil_point_cloud.txt')
    top_coil.output_solidworks('top_coil_point_cloud.txt')

    body_coil.output_scad('body_coil.scad')
    side_coil.output_scad('front_face_coil.scad')
    top_coil.output_scad('back_face_coil.scad')

    plt.show()


body_coil.set_common_current(current)
top_coil.set_common_current(current)
side_coil.set_common_current(current)

def vecb(x,y,z):
    bx_body,by_body,bz_body=body_coil.b_prime(x,y,z)
    bx_side,by_side,bz_side=side_coil.b_prime(x,y,z)
    bx_top,by_top,bz_top=top_coil.b_prime(x,y,z)
    bx=bx_body+bx_side+bx_top
    by=by_body+by_side+by_top
    bz=bz_body+bz_side+bz_top
    return bx,by,bz

design_field=-4*pi/10*1.e-6
bx,by,bz=vecb(0.,0.,0.)
central_field=by
delta_field=1.e-9
min_field=central_field-delta_field
max_field=central_field+delta_field
print(central_field,delta_field)

if (options.planes):
    # show the inner field

    figtest,(axtest1,axtest2,axtest3)=plt.subplots(nrows=3)
    
    x2d,y2d=np.mgrid[-1.0:1.0:100j,-1.0:1.0:100j]
    bx2d,by2d,bz2d=vecb(x2d,y2d,0.)
    im=axtest1.pcolormesh(x2d,y2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=abs(min_field),vmax=abs(max_field))
    #im=axtest1.pcolormesh(x2d,y2d,bx2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest1,format='%.5e')

    x2d,z2d=np.mgrid[-1.0:1.0:100j,-1.0:1.0:100j]
    bx2d,by2d,bz2d=vecb(x2d,0.,z2d)
    im=axtest2.pcolormesh(z2d,x2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=abs(min_field),vmax=abs(max_field))
    #im=axtest2.pcolormesh(z2d,x2d,by2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest2,format='%.5e')

    y2d,z2d=np.mgrid[-1.0:1.0:100j,-1.0:1.0:100j]
    bx2d,by2d,bz2d=vecb(0.,y2d,z2d)
    im=axtest3.pcolormesh(z2d,y2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=abs(min_field),vmax=abs(max_field))
    #im=axtest3.pcolormesh(z2d,y2d,by2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest3,format='%.5e')

    # show the outer field
    
    figouter,(axouter1,axouter2,axouter3)=plt.subplots(nrows=3)

    x2d,y2d=np.mgrid[-1.5:1.5:100j,-1.5:1.5:100j]
    bx2d,by2d,bz2d=vecb(x2d,y2d,0.)
    bmod=np.sqrt(bx2d**2+by2d**2+bz2d**2)
    mask=((abs(x2d)<1.2)&(abs(y2d)<1.2))
    x2d_masked=np.ma.masked_where(mask,x2d)
    y2d_masked=np.ma.masked_where(mask,y2d)
    im=axouter1.pcolor(x2d_masked,y2d_masked,bmod)
    figouter.colorbar(im,ax=axouter1,format='%.2e')

    x2d,z2d=np.mgrid[-1.5:1.5:100j,-1.5:1.5:100j]
    bx2d,by2d,bz2d=vecb(x2d,0.,z2d)
    bmod=np.sqrt(bx2d**2+by2d**2+bz2d**2)
    mask=((abs(x2d)<1.2)&(abs(z2d)<1.2))
    x2d_masked=np.ma.masked_where(mask,x2d)
    z2d_masked=np.ma.masked_where(mask,z2d)
    im=axouter2.pcolor(z2d_masked,x2d_masked,bmod)
    figouter.colorbar(im,ax=axouter2,format='%.2e')

    y2d,z2d=np.mgrid[-1.5:1.5:100j,-1.5:1.5:100j]
    bx2d,by2d,bz2d=vecb(0.,y2d,z2d)
    bmod=np.sqrt(bx2d**2+by2d**2+bz2d**2)
    mask=((abs(y2d)<1.2)&(abs(z2d)<1.2))
    y2d_masked=np.ma.masked_where(mask,y2d)
    z2d_masked=np.ma.masked_where(mask,z2d)
    im=axouter3.pcolor(z2d_masked,y2d_masked,bmod)
    figouter.colorbar(im,ax=axouter3,format='%.2e')

    plt.show()

fig7, (ax71) = plt.subplots(nrows=1)

def fitfunc(x,p0,p2,p4,p6):
    return p0+p2*x**2+p4*x**4+p6*x**6

def fitgraph(xdata,ydata,ax):
    popt,pcov=curve_fit(fitfunc,xdata[abs(xdata)<.5],ydata[abs(xdata)<.5])
    print(popt)
    ax.plot(xdata,fitfunc(xdata,*popt),'r--',label='$p_0$=%2.1e,$p_2$=%2.1e,$p_4$=%2.1e,$p_6$=%2.1e'%tuple(popt))

points1d=np.mgrid[-1:1:101j]
bx1d,by1d,bz1d=vecb(0.,points1d,0.)
fitgraph(points1d,by1d,ax71)
ax71.plot(points1d,by1d,label='$B_y(0,y,0)$')
bx1d,by1d,bz1d=vecb(points1d,0.,0.)
fitgraph(points1d,by1d,ax71)
ax71.plot(points1d,by1d,label='$B_y(x,0,0)$')
bx1d,by1d,bz1d=vecb(0.,0.,points1d)
fitgraph(points1d,by1d,ax71)
ax71.plot(points1d,by1d,label='$B_y(0,0,z)$')

ax71.axis((-.5,.5,min_field,max_field))
ax71.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax71.legend()

plt.show()


# Statistics in ROI

print
print('Statistics on the ROI')
print

x,y,z=np.mgrid[-.49:.49:100j,-.49:.49:100j,-.49:.49:100j]

rcell=0.3 # m, cell radius
hcell=0.1601 # m, cell height
dcell=0.08 # m, bottom to top distance of cells
mask=(abs(z)>=dcell/2)&(abs(z)<=dcell/2+hcell)&(x**2+y**2<rcell**2)
mask_upper=mask&(z>0)
mask_lower=mask&(z<0)

# This is used to test the cell dimensions.

#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
#scat=ax.scatter(x[mask_upper],y[mask_upper],z[mask_upper])
#plt.show()

bx_roi,by_roi,bz_roi=vecb(x,y,z)

print('Both cells')
by_mask_max=np.amax(by_roi[mask])
by_mask_min=np.amin(by_roi[mask])
by_mask_delta=by_mask_max-by_mask_min
print('The max/min/diff By masks are %e %e %e'%(by_mask_max,by_mask_min,by_mask_delta))
by_std=np.std(by_roi[mask])
print('The masked standard deviation of By is %e'%by_std)
print

print('Upper cell')
by_mask_max=np.amax(by_roi[mask_upper])
by_mask_min=np.amin(by_roi[mask_upper])
by_mask_delta=by_mask_max-by_mask_min
print('The max/min/diff By masks are %e %e %e'%(by_mask_max,by_mask_min,by_mask_delta))
by_std=np.std(by_roi[mask_upper])
print('The masked standard deviation of By is %e'%by_std)
print

print('Lower cell')
by_mask_max=np.amax(by_roi[mask_lower])
by_mask_min=np.amin(by_roi[mask_lower])
by_mask_delta=by_mask_max-by_mask_min
print('The max/min/diff By masks are %e %e %e'%(by_mask_max,by_mask_min,by_mask_delta))
by_std=np.std(by_roi[mask_lower])
print('The masked standard deviation of By is %e'%by_std)
print

print('Both cells BT2')
bt2_roi=bx_roi**2+bz_roi**2
bt2_ave=np.average(bt2_roi[mask])
print('The BT2 is %e'%bt2_ave)
print
