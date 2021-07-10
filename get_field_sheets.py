#!/usr/bin/python3

# Sorting points by Jeff and Rosie
# Jeff updating to parse u3 and u2-u3 for double-cos box coil
# Jeff updated to read data from comsol mesh
# Jeff updated to triangulate and plot contours in matplotlib
# Jeff updated to extract contours
# June 16, 2019 Jeff updated to use patch.py classes
# June 17, 2019 now working properly
# June 25, 2019 updates for different graphing
# February 27, 2021 Added rerouting for side pipes
# May 21, 2021 Divide up into multiple coils for deformation studies
# June 1, 2021 Start to add pipes class, add mayavi use for traces

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
from pipes import *

from scipy.optimize import curve_fit
from scipy.optimize import minimize

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

parser.add_option("-g", "--graph", dest="graph",
                  default=False, action="store_true",
                  help="show graph of quarter magnet and contours")

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

parser.add_option("-r", "--roi", dest="roi", default=False,
                  action="store_true",
                  help="calculate statistics on ROI")

parser.add_option("-v", "--verbose", dest="verbose", default=False,
                  action="store_true",
                  help="verbose option")

parser.add_option("-s", "--simplify", dest="simplify",
                  default=-1, help="factor for VW simplification",
                  metavar="factor")

# simplify is used to remove points, making it easier to draw and
# faster to calculate the field

parser.add_option("-x", dest="x", default=False, action="store_true",
                  help="use file containing parameters")

parser.add_option("-1", "--and1", dest="and1", default=False,
                  action="store_true", help="add one parameter")

parser.add_option("-w", "--wiggle", dest="wiggle",
                  default=-1, help="sigma to wiggle each point (m)",
                  metavar="sigma")


(options, args) = parser.parse_args()

with open(options.infile) as stream:
    d=np.loadtxt(stream,comments="%",unpack=True)
    
du1=d[:,~np.isnan(d[2])] # remove NaN's in u1
x_inner,y_inner,u1_inner,u2_inner,u3_inner=du1

du2=d[:,~np.isnan(d[3])] # remove NaN's in u2
x_outer,y_outer,u1_outer,u2_outer,u3_outer=du2

# geometry factors from COMSOL model
a_out = 2.2 # m
a_in = 2.0 # m
#a_out = 1.8 # m
#a_in = 1.4 # m


# specify levels

current=float(options.current) # amperes; design current = step in scalar potential

if options.and1:
    if options.x:
        n=len(np.loadtxt("x.txt",ndmin=1))+1
    else:
        n=2
else:
    n=10 # set what you want here, for the order of the polynomial 2n+1
xarray=np.zeros(n)
xarray[0]=current

if options.x:
    xarray_loaded=np.loadtxt("x.txt",ndmin=1)
    print("Loaded file x.txt: ",xarray_loaded)
    current=xarray_loaded[0]
    for i in range(min(n,len(xarray_loaded))):
        xarray[i]=xarray_loaded[i]
    
#def get_levels(current,alpha=0,beta=0):
#    maxu1=np.max(u1_inner) # u1 always goes positive
#    maxu23=np.max(u2_outer-u3_outer)
#    minu23=np.min(u2_outer-u3_outer)
#    maxu3=np.max(u3_outer)
#    minu3=np.min(u3_outer)
#    maxphi=maxu1
#    minphi=minu23

#    nmax=int(maxphi/current+.5)
#    nmin=int(minphi/current-.5)
#    maxlevel=(nmax+.5)*current
#    minlevel=(nmin+.5)*current
#    levels=[]
#    for n in range(nmin,nmax):
#        levels.append((n+.5)*current+(n+.5)**3*alpha/1e6+(n+.5)**5*beta/1e10)
#    return levels

delta_n_spread=0.1 # relative width of trace
nspread=5 # number of wires to use to make up trace
def get_levels(parameters):
    current=parameters[0]
    
    maxu1=np.max(u1_inner) # u1 always goes positive
    maxu23=np.max(u2_outer-u3_outer)
    minu23=np.min(u2_outer-u3_outer)
    maxu3=np.max(u3_outer)
    minu3=np.min(u3_outer)
    maxphi=maxu1
    minphi=minu23

    nmax=int(maxphi/current+.5)
    nmin=int(minphi/current-.5)
    maxlevel=(nmax+.5)*current
    minlevel=(nmin+.5)*current
    levels=[]
    for n in range(nmin,nmax):
        for m in range(nspread):
            delta_n=(m-(nspread-1)/2)/((nspread-1)/2)*delta_n_spread/2
            print(m,delta_n,n+delta_n)
            thislevel=(n+delta_n+.5)*current
            for order in range(1,len(parameters)):
                thislevel=thislevel+(n+delta_n+.5)**(2*order+1)*parameters[order]/10**(2*(2*order+1))
            levels.append(thislevel)
    return levels

mylevels=get_levels(xarray)

# mask out bad triangles that will be created when automatically
# triangulating outer (concave) region.

polygon_inner=Polygon([(0,0),(a_in/2,0),(a_in/2,a_in/2),(0,a_in/2)])
polygon_outer=Polygon([(a_in/2,0),(a_out/2,0),(a_out/2,a_out/2),(0,a_out/2),(0,a_in/2),(a_in/2,a_in/2)])

tri=Triangulation(x_outer,y_outer)
ntri=tri.triangles.shape[0]

# example of masking a region from https://matplotlib.org/examples/pylab_examples/tripcolor_demo.html

xmid=x_outer[tri.triangles].mean(axis=1)
ymid=y_outer[tri.triangles].mean(axis=1) # finds the center points of each triangle
mask=np.zeros(ntri,dtype=bool)
i=0
for x,y in zip(xmid,ymid):
    if not polygon_outer.contains(Point(x,y)):
        mask[i]=True
    i=i+1
print(mask)
tri.set_mask(mask)

## refiner from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/tricontour_smooth_delaunay.html
#subdiv=3
#refiner = UniformTriRefiner(tri)
#tri_refi, u3_refi = refiner.refine_field(u3_outer, subdiv=subdiv)
#tri_refi, u23_refi = refiner.refine_field(u2_outer-u3_outer, subdiv=subdiv)
## refine inner coils too... seems to affect levels(?)
tri_inner=Triangulation(x_inner,y_inner)
##refiner_inner = UniformTriRefiner(tri_inner)
##tri_inner_refi, u1_refi = refiner.refine_field(u1_inner, subdiv=subdiv)


# scipy.interpolate attempt
#from scipy.interpolate import griddata
#grid_z2 = griddata(tri, u2_outer-u3_outer, (grid_x, grid_y), method='cubic')

# make graphs


def wind_coils(levels):

    fig,ax1=plt.subplots()

    #ax1.triplot(tri,color='0.7') # if you want to see the triangulation

    #u23_contours=ax1.tricontour(tri_refi,u23_refi,levels=levels)
    u23_contours=ax1.tricontour(tri,u2_outer-u3_outer,levels=levels)
    if (options.plotmesh):
        ax1.plot(x_outer,y_outer,'k.')
    ax1.axis((0,a_out/2,0,a_out/2))
    fig.colorbar(u23_contours,ax=ax1)

    #u3_contours=ax1.tricontour(tri_refi, u3_refi, levels=levels)
    u3_contours=ax1.tricontour(tri,u3_outer,levels=levels)
    if (options.plotmesh):
        ax1.plot(x_outer,y_outer,'k.')

    #u1_contours=ax1.tricontour(x_inner, y_inner, u1_inner, levels=levels)
    u1_contours=ax1.tricontour(tri_inner,u1_inner,levels=levels)
    if (options.plotmesh):
        ax1.plot(x_inner,y_inner,'k.')


    # In python3, there might be empty contours, which I am cutting out
    # using the following commands
    u23_contours.allsegs=[x for x in u23_contours.allsegs if x]
    u3_contours.allsegs=[x for x in u3_contours.allsegs if x]
    u1_contours.allsegs=[x for x in u1_contours.allsegs if x]


    # define penetrating pipes

    mypipes=pipelist()
    
    gcc_x=.444 #m, guide center-to-center in x direction
    gcc_y=.764 #m, guide center-to-center in y direction
    gdia=.15 #m, guide diameter

    circle1 = plt.Circle((gcc_x/2,gcc_y/2),gdia/2,color='r')
    circle2 = plt.Circle((0,0),gdia/2,color='r')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)

    mypipes.add_pipe(gdia/2,gcc_x/2,gcc_y/2,'z')
    mypipes.add_pipe(gdia/2,gcc_x/2,-gcc_y/2,'z')
    mypipes.add_pipe(gdia/2,-gcc_x/2,gcc_y/2,'z')
    mypipes.add_pipe(gdia/2,-gcc_x/2,-gcc_y/2,'z')

    mypipes.add_pipe(gdia/2,0,0,'z')

    
    rpipes=[0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.015,0.015] #m
    ypipes=[0,0,0,0,0,0.4,0.4,0.4,0.4,0.4,0.14,0.185/2] #m
    zpipes=[-0.44,-0.22,0,0.22,0.44,-0.44,-0.22,0,0.22,0.44,0,0] #m
    pipe_density=10 # number of points to inscribe around the pipe
    # to remove the pipes, uncomment below
    #rpipes=[]
    #ypipes=[]
    #zpipes=[]

    for j in range(len(rpipes)):
        mypipes.add_pipe(rpipes[j],ypipes[j],zpipes[j],'x')

    for j in range(len(rpipes)):
        ax1.add_patch(plt.Rectangle((a_in/2,ypipes[j]-rpipes[j]),(a_out-a_in)/2,2*rpipes[j]))

    rpipes_floor=[0.0697/2,0.0697/2,0.0697/2,0.0697/2,
                  .015,.015,.015,.015,.015,.015,.015,
                  .015,.015,.015,.015,.015,.015,.015] #m
    xpipes_floor=[0,0,0,0.2,
                  .38,.38,.38,.38,.38,.38,.38,
                  .795,.795,.795,.795,.795,.795,.795] #m
    zpipes_floor=[-0.2,0,0.2,0,
                  -.41-.35-.25,-.41-.35,-.41,0,.41,.41+.35,.41+.35+.25,
                  -.41-.35-.25,-.41-.35,-.41,0,.41,.41+.35,.41+.35+.25] #m


    for j in range(len(rpipes_floor)):
        mypipes.add_pipe(rpipes_floor[j],zpipes_floor[j],xpipes_floor[j],'y')

    for j in range(len(rpipes_floor)):
        ax1.add_patch(plt.Rectangle((xpipes_floor[j]-rpipes_floor[j],a_in/2),2*rpipes_floor[j],(a_out-a_in)/2))
        
    if not options.graph:
        plt.close()
    plt.show()

    ## extracting all the contours and graphing them
    if (options.contours):
        fig2, (ax3, ax4) = plt.subplots(nrows=2)

        # nseg=len(u23_contours.allsegs)

        for i,cnt in enumerate(u23_contours.allsegs):
            seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
            x=seg[:,0]
            y=seg[:,1]
            ax3.plot(x,y,'.-',color='black',ms=1)
            if(float(options.simplify)>0):
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
            if(float(options.simplify)>0):
                segsimp=simplify_coords_vw(seg,float(options.simplify))
                xsimp=segsimp[:,0]
                ysimp=segsimp[:,1]
                ax4.plot(xsimp,ysimp,'.-',color='red')
        for i,cnt in enumerate(u1_contours.allsegs):
            seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
            x=seg[:,0]
            y=seg[:,1]
            ax4.plot(x,y,'.-',color='black',ms=1)
            if(float(options.simplify)>0):
                segsimp=simplify_coords_vw(seg,float(options.simplify))
                xsimp=segsimp[:,0]
                ysimp=segsimp[:,1]
                ax4.plot(xsimp,ysimp,'.-',color='red')
        ax4.axis((0,a_out/2,0,a_out/2))

        plt.show()
    # conclusion:  the contours are all there and are ordered in the same way relative to each other

    # assemble 3D coils

    body_t=coilset()
    body_b=coilset()
    body_tr=coilset()
    body_tl=coilset()
    body_br=coilset()
    body_bl=coilset()

    body_list=[body_t,body_b,body_tr,body_tl,body_br,body_bl]

    #print("There are %d outer contours."%len(u23_contours.allsegs))

    for i,cnt in enumerate(u23_contours.allsegs):
        seg=cnt[0] # if there are multiple contours at same level there will be more than one seg
        # these go from inner to outer!
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
        xs=np.flip(x,0)
        ys=np.flip(y,0)
        zs=[a_out/2]*len(y)

        xnew=x
        ynew=y
        znew=z
        # check for pipe conflict
        for j in range(len(rpipes)):
            rpipe=rpipes[j]
            ypipe=ypipes[j]
            zpipe=zpipes[j]
            x_around=[]
            y_around=[]
            z_around=[]
            if(ynew[-1]<ypipe+rpipe and ynew[-1]>ypipe-rpipe):
                #print('Pipe conflict inner! Horizontal')
                #print(xnew[-1],ynew[-1],znew[-1])
                x_around=[xnew[-1]]*pipe_density
                if(ynew[-1]>ypipe): # go around the top side
                    theta_start=math.atan2(ynew[-1]-ypipe,sqrt(rpipe**2-(ynew[-1]-ypipe)**2))
                    theta_end=pi-theta_start
                    theta_around=[theta_start+(theta_end-theta_start)*i/(pipe_density-1) for i in range(0,pipe_density)]
                    y_around=ypipe+rpipe*sin(theta_around)
                else: # go around the bottom side
                    theta_start=math.atan2(ypipe-ynew[-1],sqrt(rpipe**2-(ynew[-1]-ypipe)**2))
                    theta_end=pi-theta_start
                    theta_around=[theta_start+(theta_end-theta_start)*i/(pipe_density-1) for i in range(0,pipe_density)]
                    y_around=ypipe-rpipe*sin(theta_around)
                z_around=zpipe-rpipe*cos(theta_around)
            xnew=np.concatenate((xnew,x_around))
            ynew=np.concatenate((ynew,y_around))
            znew=np.concatenate((znew,z_around))
        # check for vertical pipe conflict
        x_arounds=[]
        y_arounds=[]
        z_arounds=[]
        for j in range(len(rpipes_floor)):
            rpipe=rpipes_floor[j]
            xpipe=xpipes_floor[j]
            zpipe=zpipes_floor[j]
            x_around=[]
            y_around=[]
            z_around=[]
            if(xnew[-1]<xpipe+rpipe and xnew[-1]>xpipe-rpipe):
                #print('Pipe conflict inner! Vertical')
                #print(xnew[-1],ynew[-1],znew[-1])
                y_around=[ynew[-1]]*pipe_density
                if(xnew[-1]>xpipe): # go around the right side
                    theta_start=math.atan2(xnew[-1]-xpipe,sqrt(rpipe**2-(xnew[-1]-xpipe)**2))
                    theta_end=pi-theta_start
                    theta_around=[theta_start+(theta_end-theta_start)*i/(pipe_density-1) for i in range(0,pipe_density)]
                    x_around=xpipe+rpipe*sin(theta_around)
                else: # go around the left side
                    theta_start=math.atan2(xpipe-xnew[-1],sqrt(rpipe**2-(xnew[-1]-xpipe)**2))
                    theta_end=pi-theta_start
                    theta_around=[theta_start+(theta_end-theta_start)*i/(pipe_density-1) for i in range(0,pipe_density)]
                    x_around=xpipe-rpipe*sin(theta_around)
                z_around=zpipe-rpipe*cos(theta_around)
                #print(x_around,y_around,z_around)
            x_arounds=np.concatenate((x_arounds,x_around))
            y_arounds=np.concatenate((y_arounds,y_around))
            z_arounds=np.concatenate((z_arounds,z_around))
        xnew=np.concatenate((xnew,x_arounds))
        ynew=np.concatenate((ynew,y_arounds))
        znew=np.concatenate((znew,z_arounds))
        xnew=np.concatenate((xnew,xs))
        ynew=np.concatenate((ynew,ys))
        znew=np.concatenate((znew,zs))
        mirrored=False
        if (x[0]<0.000001):
            # mirror through yz-plane
            xnew=np.concatenate((xnew,np.delete(-x,0))) # avoid repeating point
            ynew=np.concatenate((ynew,np.delete(y,0)))
            znew=np.concatenate((znew,np.delete(zs,0)))

            xnew=np.concatenate((xnew,-np.flip(x_arounds,0)))
            ynew=np.concatenate((ynew,np.flip(y_arounds,0)))
            znew=np.concatenate((znew,np.flip(z_arounds,0)))

            xnew=np.concatenate((xnew,-xs))
            ynew=np.concatenate((ynew,ys))
            znew=np.concatenate((znew,z))
            mirrored=True
            points=np.column_stack((xnew,ynew,znew))
            body_t.add_coil(points)
        else:
            # complete loop
            # check for pipe conflict
            for j in range(len(rpipes)):
                rpipe=rpipes[len(rpipes)-1-j]
                ypipe=ypipes[len(rpipes)-1-j]
                zpipe=zpipes[len(rpipes)-1-j] # pipes specified in
                                              # direction of
                                              # increasing z; we're
                                              # going in the
                                              # decreasing z direction
                x_around=[]
                y_around=[]
                z_around=[]
                if(ynew[-1]<ypipe+rpipe and ynew[-1]>ypipe-rpipe):
                    #print('Pipe conflict outer!')
                    #print(xnew[-1],ynew[-1],znew[-1],ys[0],ys[-1],ynew[0])
                    x_around=[xnew[-1]]*pipe_density
                    if(ynew[-1]>ypipe): # go around the top side
                        theta_start=math.atan2(ynew[-1]-ypipe,sqrt(rpipe**2-(ynew[-1]-ypipe)**2))
                        theta_end=pi-theta_start
                        theta_around=[theta_start+(theta_end-theta_start)*i/(pipe_density-1) for i in range(0,pipe_density)]
                        y_around=ypipe+rpipe*sin(theta_around)
                    else: # go around the bottom side
                        theta_start=math.atan2(ypipe-ynew[-1],sqrt(rpipe**2-(ynew[-1]-ypipe)**2))
                        theta_end=pi-theta_start
                        theta_around=[theta_start+(theta_end-theta_start)*i/(pipe_density-1) for i in range(0,pipe_density)]
                        y_around=ypipe-rpipe*sin(theta_around)
                    z_around=zpipe+rpipe*cos(theta_around)
                xnew=np.append(xnew,x_around)
                ynew=np.append(ynew,y_around)
                znew=np.append(znew,z_around)
            # check for pipe conflict (Vertical)
            for j in range(len(rpipes_floor)):
                rpipe=rpipes_floor[len(rpipes_floor)-1-j]
                xpipe=xpipes_floor[len(rpipes_floor)-1-j]
                zpipe=zpipes_floor[len(rpipes_floor)-1-j] # pipes
                                              # specified in direction
                                              # of increasing z; we're
                                              # going in the
                                              # decreasing z direction
                x_around=[]
                y_around=[]
                z_around=[]
                if(xnew[-1]<xpipe+rpipe and xnew[-1]>xpipe-rpipe):
                    #print('Pipe conflict outer!')
                    #print(xnew[-1],ynew[-1],znew[-1],ys[0],ys[-1],ynew[0])
                    y_around=[ynew[-1]]*pipe_density
                    if(xnew[-1]>xpipe): # go around the right side
                        theta_start=math.atan2(xnew[-1]-xpipe,sqrt(rpipe**2-(xnew[-1]-xpipe)**2))
                        theta_end=pi-theta_start
                        theta_around=[theta_start+(theta_end-theta_start)*i/(pipe_density-1) for i in range(0,pipe_density)]
                        x_around=xpipe+rpipe*sin(theta_around)
                    else: # go around the left side
                        theta_start=math.atan2(xpipe-xnew[-1],sqrt(rpipe**2-(xnew[-1]-xpipe)**2))
                        theta_end=pi-theta_start
                        theta_around=[theta_start+(theta_end-theta_start)*i/(pipe_density-1) for i in range(0,pipe_density)]
                        x_around=xpipe-rpipe*sin(theta_around)
                    z_around=zpipe+rpipe*cos(theta_around)
                xnew=np.append(xnew,x_around)
                ynew=np.append(ynew,y_around)
                znew=np.append(znew,z_around)

            xnew=np.append(xnew,xnew[0])
            ynew=np.append(ynew,ynew[0])
            znew=np.append(znew,znew[0])
            points=np.column_stack((xnew,ynew,znew))
            body_tr.add_coil(points)

        # reflect through xz-plane
        ynew=-ynew
        points=np.column_stack((xnew,ynew,znew))
        if mirrored:
            body_b.add_coil(points)
        else:
            body_br.add_coil(points)
            ynew=-ynew # put it back for a sec
            # reflect separate trace through yz-plane
            xnew=-xnew
            # reverse the windings
            xnew=np.flip(xnew,0)
            ynew=np.flip(ynew,0)
            znew=np.flip(znew,0)
            points=np.column_stack((xnew,ynew,znew))
            body_tl.add_coil(points)
            # and through the xz-plane
            ynew=-ynew
            points=np.column_stack((xnew,ynew,znew))
            body_bl.add_coil(points)

    # now for the face plates
    front_face_coil=coilset()
    back_face_coil=coilset()

    #print("There are %d face coils in %d levels."%(len(u1_contours.allsegs),len(u1_contours.levels)))

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
            points=np.column_stack((xnew,ynew,znew))
            back_face_coil.add_coil(points)
            # mirror through xy-plane
            znew=[a_out/2]*len(ynew)
            xnew=np.flip(xnew,0)
            ynew=np.flip(ynew,0)
            points=np.column_stack((xnew,ynew,znew))
            front_face_coil.add_coil(points)
            # mirror through xz-plane
            ynew=-ynew
            points=np.column_stack((xnew,ynew,znew))
            front_face_coil.add_coil(points)
            znew=[-a_out/2]*len(ynew)
            xnew=np.flip(xnew,0)
            ynew=np.flip(ynew,0)    
            points=np.column_stack((xnew,ynew,znew))
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
            points=np.column_stack((xnew,ynew,znew))
            back_face_coil.add_coil(points)
            # mirror through xy-plane
            znew=[a_out/2]*len(ynew)
            xnew=np.flip(xnew,0)
            ynew=np.flip(ynew,0)
            points=np.column_stack((xnew,ynew,znew))
            front_face_coil.add_coil(points)
            # mirror through xz-plane
            ynew=-ynew
            points=np.column_stack((xnew,ynew,znew))
            front_face_coil.add_coil(points)
            znew=[-a_out/2]*len(ynew)
            xnew=np.flip(xnew,0)
            ynew=np.flip(ynew,0)    
            points=np.column_stack((xnew,ynew,znew))
            back_face_coil.add_coil(points)

    all_coil_list=body_list
    all_coil_list.append(front_face_coil)
    all_coil_list.append(back_face_coil)
    # Apply coil deformations, if to be included in design optimization
    #body_tr.move(.001,0,0)
    #body_tr.move(.001,0,0)
    front_face_coil.move(0,0,.002)
    back_face_coil.move(0,0,-.002)

    return all_coil_list

all_coil_list=wind_coils(mylevels)

for coil in all_coil_list:
    print("There are %d coils in this coil"%coil.ncoils)

if(options.wiggle>0):
    for coil in all_coil_list:
        coil.wiggle(float(options.wiggle))

# Apply coil deformations post winding
#body_tr.move(.001,0,0)
#body_tr.move(.001,0,0)

if(options.traces):
    fig3 = plt.figure()
    ax5 = fig3.add_subplot(111, projection='3d')
    colors=['black','grey','darkgrey','silver','lightgrey','whitesmoke','blue','green']
    for i,coil in enumerate(all_coil_list):
        coil.draw_coils(ax5,'-',colors[i])
        coil.draw_coils_mayavi()
    mlab.show()
    

    #testing length method
    
    #front_face_coil.draw_coil(3,ax5,'-','red')
    #print(front_face_coil.coils[3].length())

    #body_tr.draw_coil(100,ax5,'-','red')
    #print(body_tr.coils[100].length())

    l=0
    for coil in all_coil_list:
        thisl=coil.length()
        print('Length %f'%thisl)
        l=l+thisl
    print('Total length %f'%l)

    # table from https://bulkwire.com/magnet-wire
    
    ohmkm=21.37
    print('18 AWG is %f Ohm/km'%ohmkm)

    resistance=l*ohmkm/1000
    print('Total resistance %f Ohm'%resistance)

    voltage=current*resistance
    print('Voltage %f V'%voltage)

    power=current*voltage
    print('Power %f W'%power)

    kgkm=7.47
    print('18 AWG is %f kg/km'%kgkm)

    weight=l*kgkm/1000
    print('Weight %f kg'%weight)
    plt.show()

def vecb(coil_list,x,y,z):
    bx=0.*x
    by=0.*y
    bz=0.*z
    for coil in coil_list:
        bx_tmp,by_tmp,bz_tmp=coil.b_prime(x,y,z)
        bx=bx+bx_tmp
        by=by+by_tmp
        bz=bz+bz_tmp
    return bx,by,bz

for coil in all_coil_list:
    coil.set_common_current(current/nspread)

design_field=-4*pi/10*1.e-6
bx,by,bz=vecb(all_coil_list,0.,0.,0.)
central_field=by
delta_field=1.e-9
min_field=central_field-delta_field
max_field=central_field+delta_field
print(central_field,delta_field)

if (options.planes):
    # show the inner field

    figtest,(axtest1,axtest2,axtest3)=plt.subplots(nrows=3)
    
    x2d,y2d=np.mgrid[-1.0:1.0:101j,-1.0:1.0:101j]
    bx2d,by2d,bz2d=vecb(all_coil_list,x2d,y2d,0.)
    im=axtest1.pcolormesh(x2d,y2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=abs(min_field),vmax=abs(max_field))
    #im=axtest1.pcolormesh(x2d,y2d,bx2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest1,format='%.5e')

    x2d,z2d=np.mgrid[-1.0:1.0:101j,-1.0:1.0:101j]
    bx2d,by2d,bz2d=vecb(all_coil_list,x2d,0.,z2d)
    im=axtest2.pcolormesh(z2d,x2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=abs(min_field),vmax=abs(max_field))
    #im=axtest2.pcolormesh(z2d,x2d,by2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest2,format='%.5e')

    y2d,z2d=np.mgrid[-1.0:1.0:101j,-1.0:1.0:101j]
    bx2d,by2d,bz2d=vecb(all_coil_list,0.,y2d,z2d)
    im=axtest3.pcolormesh(z2d,y2d,np.sqrt(bx2d**2+by2d**2+bz2d**2),vmin=abs(min_field),vmax=abs(max_field))
    #im=axtest3.pcolormesh(z2d,y2d,by2d,vmin=-3e-6,vmax=3e-6)
    figtest.colorbar(im,ax=axtest3,format='%.5e')

    # show the outer field
    
    figouter,(axouter1,axouter2,axouter3)=plt.subplots(nrows=3)

    outer_roi=1.5
    inner_roi=1.2
    
    x2d,y2d=np.mgrid[-outer_roi:outer_roi:101j,-outer_roi:outer_roi:101j]
    bx2d,by2d,bz2d=vecb(all_coil_list,x2d,y2d,0.)
    bmod=np.sqrt(bx2d**2+by2d**2+bz2d**2)
    mask=((abs(x2d)<inner_roi)&(abs(y2d)<inner_roi))
    x2d_masked=np.ma.masked_where(mask,x2d)
    y2d_masked=np.ma.masked_where(mask,y2d)
    im=axouter1.pcolor(x2d_masked,y2d_masked,bmod)
    figouter.colorbar(im,ax=axouter1,format='%.2e')

    x2d,z2d=np.mgrid[-outer_roi:outer_roi:101j,-outer_roi:outer_roi:101j]
    bx2d,by2d,bz2d=vecb(all_coil_list,x2d,0.,z2d)
    bmod=np.sqrt(bx2d**2+by2d**2+bz2d**2)
    mask=((abs(x2d)<inner_roi)&(abs(z2d)<inner_roi))
    x2d_masked=np.ma.masked_where(mask,x2d)
    z2d_masked=np.ma.masked_where(mask,z2d)
    im=axouter2.pcolor(z2d_masked,x2d_masked,bmod)
    figouter.colorbar(im,ax=axouter2,format='%.2e')

    y2d,z2d=np.mgrid[-outer_roi:outer_roi:101j,-outer_roi:outer_roi:101j]
    bx2d,by2d,bz2d=vecb(all_coil_list,0.,y2d,z2d)
    bmod=np.sqrt(bx2d**2+by2d**2+bz2d**2)
    mask=((abs(y2d)<inner_roi)&(abs(z2d)<inner_roi))
    y2d_masked=np.ma.masked_where(mask,y2d)
    z2d_masked=np.ma.masked_where(mask,z2d)
    im=axouter3.pcolor(z2d_masked,y2d_masked,bmod)
    figouter.colorbar(im,ax=axouter3,format='%.2e')

    plt.show()

def fitfunc(x,p0,p2,p4,p6):
    return p0+p2*x**2+p4*x**4+p6*x**6

def fitgraph(xdata,ydata,ax):
    popt,pcov=curve_fit(fitfunc,xdata[abs(xdata)<.5],ydata[abs(xdata)<.5])
    ax.plot(xdata,fitfunc(xdata,*popt),'r--',label='$p_0$=%2.1e,$p_2$=%2.1e,$p_4$=%2.1e,$p_6$=%2.1e'%tuple(popt))
    #print(popt)
    return(popt)

def fitnograph(xdata,ydata):
    popt,pcov=curve_fit(fitfunc,xdata[abs(xdata)<.5],ydata[abs(xdata)<.5])
    return(popt)

def dofit(coil_list):

    fig7, (ax71) = plt.subplots(nrows=1)

    points1d=np.mgrid[-1:1:101j]
    bx1d,by1d,bz1d=vecb(coil_list,0.,points1d,0.)
    popt=fitgraph(points1d,by1d,ax71)
    ax71.plot(points1d,by1d,label='$B_y(0,y,0)$')
    bx1d,by1d,bz1d=vecb(coil_list,points1d,0.,0.)
    fitgraph(points1d,by1d,ax71)
    ax71.plot(points1d,by1d,label='$B_y(x,0,0)$')
    bx1d,by1d,bz1d=vecb(coil_list,0.,0.,points1d)
    fitgraph(points1d,by1d,ax71)
    ax71.plot(points1d,by1d,label='$B_y(0,0,z)$')

    ax71.axis((-.5,.5,min_field,max_field))
    ax71.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax71.legend()

    if not options.graph:
        plt.close()
    plt.show()

    return popt[1]

def dofity(coil_list):

    points1d=np.mgrid[-1:1:101j]
    bx1d,by1d,bz1d=vecb(coil_list,0.,points1d,0.)
    popt=fitnograph(points1d,by1d)

    return popt[1]

def dofitxyz(coil_list):
    points1d=np.mgrid[-1:1:101j]
    bx1d,by1d,bz1d=vecb(coil_list,0.,points1d,0.)
    popty=fitnograph(points1d,by1d)
    bx1d,by1d,bz1d=vecb(coil_list,points1d,0.,0.)
    poptx=fitnograph(points1d,by1d)
    bx1d,by1d,bz1d=vecb(coil_list,0.,0.,points1d)
    poptz=fitnograph(points1d,by1d)
    wt2=1/3*.5**3
    wt4=1/5*.5**5
    wt6=1/7*.5**7
    return sqrt((popty[1]**2+poptx[1]**2+poptz[1]**2)*wt2**2
                +(popty[2]**2+poptx[2]**2+poptz[2]**2)*wt4**2
                +(popty[3]**2+poptx[3]**2+poptz[3]**2)*wt6**2)
    #return sqrt((popty[1]**2+poptx[1]**2+poptz[1]**2)*wt2**2)


dofit(all_coil_list)
# Statistics in ROI

if options.roi:

    print
    print('Statistics on the ROI')
    print

    x,y,z=np.mgrid[-.3:.3:61j,-.3:.3:61j,-.3:.3:61j]

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

    bx_roi,by_roi,bz_roi=vecb(all_coil_list,x,y,z)

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

def fun(x):
    current=x[0]
    levels=get_levels(x)
    all_coil_list=wind_coils(levels)
    for coil in all_coil_list:
        coil.set_common_current(current/nspread)
    fitresult=dofitxyz(all_coil_list)
    err=fitresult**2*1e20
    print('FUN',x,err)
    return err

from scipy.optimize import Bounds
    
if not options.graph:
    # sweeping current
    #current_step=0.0001
    #theis=[]
    #errs=[]
    #for i in arange(current-20*current_step,current+20*current_step,current_step):
    #    err=fun([i])
    #    theis.append(i)
    #    errs.append(err)
    #fig8, (ax81) = plt.subplots(nrows=1)
    #ax81.plot(theis,errs)
    #ax81.set_yscale('log')
    #plt.show()

    # fitting current
    
    res=minimize(fun,xarray,method='Nelder-Mead')
    #res=minimize(fun,[current],method='Nelder-Mead')
    print('res.x',res.x)
    np.savetxt('x.txt',res.x)
