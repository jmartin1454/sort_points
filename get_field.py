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

from scipy.optimize import curve_fit

# line simplification

from simplification.cutil import simplify_coords, simplify_coords_vw

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

parser.add_option("-a", "--nou1", dest="nou1", default=False,
                  action="store_true",
                  help="no solution for u1; just use u3 alone")

parser.add_option("-s", "--simplify", dest="simplify",
                  default=-1, help="factor for VW simplification",
                  metavar="factor")

# simplify is used to remove points, making it easier to draw and
# faster to calculate the field

parser.add_option("-d", "--densify", dest="densify",
                  default=-1, help="add <densify> points to sides",
                  metavar="densify")

# densify is used to add points so you can see where the wires are, if
# just exporting points to a file for direct import to SolidWorks



(options, args) = parser.parse_args()

densify=int(options.densify)

with open(options.infile) as stream:
    d=np.loadtxt(stream,comments="%",unpack=True)

du1=d[:,(d[2]<100)] # remove NaN's in u1
x_inner,y_inner,u1_inner,u2_inner,u3_inner=du1

du2=d[:,(d[3]<100)] # remove NaN's in u2
x_outer,y_outer,u1_outer,u2_outer,u3_outer=du2

# geometry factors from COMSOL model
a_out = 2.2 # m
a_in = 2.0 # m
#a_out = 1.8 # m
#a_in = 1.4 # m

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
maxphi = 30 # amperes; biggest you can imagine the scalar potential to be
num = round(maxphi/current) # half the number of equipotentials
maxlevel = (2*num-1)*current/2
minlevel = -maxlevel
levels = np.arange(minlevel,maxlevel,current)
print(levels)

fig, ax1 = plt.subplots()

#ax1.triplot(tri, color='0.7') # if you want to see the triangulation

#u23_contours=ax1.tricontour(tri_refi,u23_refi,levels=levels)
u23_contours=ax1.tricontour(tri,u2_outer-u3_outer,levels=levels)
if (options.plotmesh):
    ax1.plot(x_outer, y_outer, 'k.')
ax1.axis((0,a_out/2,0,a_out/2))
fig.colorbar(u23_contours,ax=ax1)

#u3_contours=ax1.tricontour(tri_refi, u3_refi, levels=levels)
u3_contours=ax1.tricontour(tri, u3_outer, levels=levels)
if (options.plotmesh):
    ax1.plot(x_outer, y_outer, 'k.')

#u1_contours=ax1.tricontour(x_inner, y_inner, u1_inner, levels=levels)
u1_contours=ax1.tricontour(tri_inner, u1_inner, levels=levels)
if (options.plotmesh):
    ax1.plot(x_inner, y_inner, 'k.')
ax1.axis((0,a_out/2, 0,a_out/2))
fig.colorbar(u1_contours,ax=ax1)

gcc_x=.444 #m, guide center-to-center in x direction
gcc_y=.764 #m, guide center-to-center in y direction
gdia=.15 #m, guide diameter
circle1 = plt.Circle((gcc_x/2,gcc_y/2),gdia/2,color='r')
circle2 = plt.Circle((0,0),gdia/2,color='r')
ax1.add_patch(circle1)
ax1.add_patch(circle2)

## extracting all the contours and graphing them
if (options.contours):
    fig2, (ax3, ax4) = plt.subplots(nrows=2)

    # nseg=len(u23_contours.allsegs)

    for i,cnt in enumerate(u23_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        x=seg[:,0]
        y=seg[:,1]
        ax3.plot(x,y,'.-',color='black',ms=1)
        if(options.simplify>0):
            segsimp=simplify_coords_vw(seg,float(options.simplify))
            xsimp=segsimp[:,0]
            ysimp=segsimp[:,1]
            ax3.plot(xsimp,ysimp,'.-',color='red')
    ax3.axis((0,a_out/2,0,a_out/2))

    for i,cnt in enumerate(u3_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        x=seg[:,0]
        y=seg[:,1]
        ax4.plot(x,y,'.-',color='black',ms=1)
        if(options.simplify>0):
            segsimp=simplify_coords_vw(seg,float(options.simplify))
            xsimp=segsimp[:,0]
            ysimp=segsimp[:,1]
            ax4.plot(xsimp,ysimp,'.-',color='red')
    for i,cnt in enumerate(u1_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        x=seg[:,0]
        y=seg[:,1]
        ax4.plot(x,y,'.-',color='black',ms=1)
        if(options.simplify>0):
            segsimp=simplify_coords_vw(seg,float(options.simplify))
            xsimp=segsimp[:,0]
            ysimp=segsimp[:,1]
            ax4.plot(xsimp,ysimp,'.-',color='red')
    ax4.axis((0,a_out/2,0,a_out/2))

    plt.show()
# conclusion:  the contours are all there and are ordered in the same way relative to each other

# assemble 3D coils

body_coil=coilset()

print("There are %d outer coils."%len(u23_contours.allsegs))

for i,cnt in enumerate(u23_contours.allsegs):
    seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
    # these go from outer to inner
    if(float(options.simplify)<0):
        x=seg[:,0]
        y=seg[:,1]
    else:
        segsimp=simplify_coords_vw(seg,float(options.simplify))
        xsimp=segsimp[:,0]
        ysimp=segsimp[:,1]
        x=xsimp
        y=ysimp
    z=[-a_out/2]*len(y)
    if(densify>0):
        xfake_outer=[x[0]]*(densify-1)
        yfake_outer=[y[0]]*(densify-1)
        xfake_inner=[x[-1]]*(densify-1)
        yfake_inner=[y[-1]]*(densify-1)
        step=a_out/densify
        zfake=np.arange(-a_out/2+step,a_out/2,step)
    xs=np.flip(x,0)
    ys=np.flip(y,0)
    zs=[a_out/2]*len(y)
    if(densify>0):
        xnew=np.concatenate((x,xfake_inner,xs))
        ynew=np.concatenate((y,yfake_inner,ys))
        znew=np.concatenate((z,zfake,zs))
    else:
        xnew=np.concatenate((x,xs))
        ynew=np.concatenate((y,ys))
        znew=np.concatenate((z,zs))
    mirrored=False
    if (x[0]<0.000001):
        # mirror through yz-plane
        xnew=np.concatenate((xnew,np.delete(-x,0))) # avoid repeating point
        ynew=np.concatenate((ynew,np.delete(y,0)))
        znew=np.concatenate((znew,np.delete(zs,0)))
        if(densify>0):
            xnew=np.concatenate((xnew,np.negative(xfake_inner)))
            ynew=np.concatenate((ynew,yfake_inner))
            znew=np.concatenate((znew,-zfake)) # -zfake is same as np.flip(zfake,0)
        xnew=np.concatenate((xnew,-xs))
        ynew=np.concatenate((ynew,ys))
        znew=np.concatenate((znew,z))
        mirrored=True
    else:
        # complete loop
        if(densify>0):
            xnew=np.concatenate((xnew,xfake_outer))
            ynew=np.concatenate((ynew,yfake_outer))
            znew=np.concatenate((znew,-zfake)) # -zfake is same as np.flip(zfake,0)
        xnew=np.append(xnew,xnew[0])
        ynew=np.append(ynew,ynew[0])
        znew=np.append(znew,znew[0])
    points=np.array(zip(xnew,ynew,znew))
    body_coil.add_coil(points)

    # reflect through xz-plane
    ynew=-ynew
    points=np.array(zip(xnew,ynew,znew))
    body_coil.add_coil(points)
    if not mirrored:
        ynew=-ynew # put it back for a sec
        # reflect separate trace through yz-plane
        xnew=-xnew
        # reverse the windings
        xnew=np.flip(xnew,0)
        ynew=np.flip(ynew,0)
        znew=np.flip(znew,0)
        points=np.array(zip(xnew,ynew,znew))
        body_coil.add_coil(points)
        # and through the xz-plane
        ynew=-ynew
        points=np.array(zip(xnew,ynew,znew))
        body_coil.add_coil(points)


# now for the face plates
front_face_coil=coilset()
back_face_coil=coilset()

print("There are %d face coils in %d levels."%(len(u1_contours.allsegs),len(u1_contours.levels)))

if (not options.nou1):
    for i,cnt in enumerate(u1_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        # these go from outer to inner
        x=seg[:,0]
        y=seg[:,1]
        # get the corresponding contour from u3
        seg=u3_contours.allsegs[i][0]
        xs=seg[:,0]
        ys=seg[:,1]
        xnew=np.concatenate((x,np.delete(xs,0))) # xs on the boundary is usually a repeated point
        ynew=np.concatenate((y,np.delete(ys,0)))
        # mirror through yz-plane
        xnew=np.concatenate((xnew,np.delete(np.flip(-xs,0),[0,len(xs)-1]))) # first and last point are repeated
        xnew=np.concatenate((xnew,np.flip(-x,0)))
        ynew=np.concatenate((ynew,np.delete(np.flip(ys,0),[0,len(ys)-1])))
        ynew=np.concatenate((ynew,np.flip(y,0)))
        znew=[-a_out/2]*len(ynew)
        points=np.array(zip(xnew,ynew,znew))
        back_face_coil.add_coil(points)
        # mirror through xy-plane
        znew=[a_out/2]*len(ynew)
        xnew=np.flip(xnew,0)
        ynew=np.flip(ynew,0)
        points=np.array(zip(xnew,ynew,znew))
        front_face_coil.add_coil(points)
        # mirror through xz-plane
        ynew=-ynew
        points=np.array(zip(xnew,ynew,znew))
        front_face_coil.add_coil(points)
        znew=[-a_out/2]*len(ynew)
        xnew=np.flip(xnew,0)
        ynew=np.flip(ynew,0)    
        points=np.array(zip(xnew,ynew,znew))
        back_face_coil.add_coil(points)
else:
    for i,cnt in enumerate(u3_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        if(float(options.simplify)<0):
            xs=seg[:,0]
            ys=seg[:,1]
        else:
            segsimp=simplify_coords_vw(seg,float(options.simplify))
            xsimp=segsimp[:,0]
            ysimp=segsimp[:,1]
            xs=xsimp
            ys=ysimp
        xnew=np.concatenate((xs,np.delete(np.flip(-xs,0),0))) # delete repeated point on axis
        ynew=np.concatenate((ys,np.delete(np.flip(ys,0),0)))
        xnew=np.concatenate((xnew,[xnew[0]])) # really force closing the loop
        ynew=np.concatenate((ynew,[ynew[0]]))
        znew=[-a_out/2]*len(ynew)
        points=np.array(zip(xnew,ynew,znew))
        back_face_coil.add_coil(points)
        # mirror through xy-plane
        znew=[a_out/2]*len(ynew)
        xnew=np.flip(xnew,0)
        ynew=np.flip(ynew,0)
        points=np.array(zip(xnew,ynew,znew))
        front_face_coil.add_coil(points)
        # mirror through xz-plane
        ynew=-ynew
        points=np.array(zip(xnew,ynew,znew))
        front_face_coil.add_coil(points)
        znew=[-a_out/2]*len(ynew)
        xnew=np.flip(xnew,0)
        ynew=np.flip(ynew,0)    
        points=np.array(zip(xnew,ynew,znew))
        back_face_coil.add_coil(points)

print("There are %d body coils"%body_coil.ncoils)
print("There are %d front face coils"%front_face_coil.ncoils)
print("There are %d back face coils"%back_face_coil.ncoils)
        

        
if(options.traces):
    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111, projection='3d')
    if(densify>0):
        body_coil.draw_coils(ax5,'.','black')
        front_face_coil.draw_coils(ax5,'.','blue')
        back_face_coil.draw_coils(ax5,'.','green')
    else:
        body_coil.draw_coils(ax5,'-','black')
        front_face_coil.draw_coils(ax5,'-','blue')
        back_face_coil.draw_coils(ax5,'-','green')

    body_coil.output_solidworks('body_coil_point_cloud.txt')
    front_face_coil.output_solidworks('front_face_coil_point_cloud.txt')
    back_face_coil.output_solidworks('back_face_coil_point_cloud.txt')

    body_coil.output_scad('body_coil.scad')
    front_face_coil.output_scad('front_face_coil.scad')
    back_face_coil.output_scad('back_face_coil.scad')

    plt.show()


body_coil.set_common_current(current)
front_face_coil.set_common_current(current)
back_face_coil.set_common_current(current)

def vecb(x,y,z):
    bx_body,by_body,bz_body=body_coil.b_prime(x,y,z)
    bx_front_face,by_front_face,bz_front_face=front_face_coil.b_prime(x,y,z)
    bx_back_face,by_back_face,bz_back_face=back_face_coil.b_prime(x,y,z)
    bx=bx_body+bx_front_face+bx_back_face
    by=by_body+by_front_face+by_back_face
    bz=bz_body+bz_front_face+bz_back_face
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
