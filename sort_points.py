#!/usr/bin/python

# Sorting points by Jeff and Rosie

import numpy as np
import math
from optparse import OptionParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from scipy import interpolate 
import io

# parse command line arguments

parser = OptionParser()

parser.add_option("-i", "--inner", dest="innerfile",
                  default="Cylinder_inner_0.2mA.txt", help="read inner coils from FILE", metavar="FILE")

parser.add_option("-o", "--outer", dest="outerfile",
                  default="Cylinder_outer_0.2mA.txt", help="read outer coils from FILE", metavar="FILE")

(options, args) = parser.parse_args()

# output_file writes the redistibuted points inter seperate files for
# each isolevel

def output_file(matrix, i):
    with open("sorted_points_o_iso_%s.txt" % i,'ab') as outputfile:
        np.savetxt(outputfile, matrix)

def sorted_points(points):
    myspace=points # allocate space for the sorted points
    nnindex=0 # index of next nearest point
   
    # algorithm: take the first point, delete it out of points, then
    # find the next nearest point, delete it out of points, and so on
    # until there aren't any points left

    for i in range(len(myspace)-1):
        point=points[nnindex]
        myspace[i]=x,y,z=point
        points=np.delete(points,nnindex,0)
        xs,ys,zs=np.transpose(points)
        distance2=(x-xs)**2+(y-ys)**2+(z-zs)**2
        nnindex=np.where(distance2==distance2.min())[0][0]
    myspace[len(myspace)-1]=points # Get the last point
    
    # problem: the above algorithm sometimes gets the order of the
    # first few points wrong because the points are not all equally
    # spaced

    # algorithm: work backwards through the points and if a backward
    # step is taken (in the backward hemisphere relative to the
    # present step), then swap the points.  Keep doing this until
    # there aren't ever any swaps.

    swaps=1
    while swaps>0:
        swaps=0
        for i in np.arange(len(myspace)-1,1,-1):
            deltar1=myspace[i]-myspace[i-1]
            deltar2=myspace[i-1]-myspace[i-2]
            thedot=np.dot(deltar1,deltar2)
            thecross=np.cross(myspace[i],myspace[i-1])
            if thedot<0:
                myspace[[i-1,i-2]]=myspace[[i-2,i-1]]
                swaps+=1
        print "There were",swaps,"swaps."
    return myspace


with open(options.innerfile) as innerstream:
    d=np.loadtxt(innerstream,comments="%",unpack=True)

unique=np.unique(d[3])

print (unique)

for i in range(len(unique)):
    value=unique[i]
    print( "Processing iso level ",value)
    iso=d[:,(d[3]==value)]  # all the points in this iso
    points=np.transpose(iso[:3])  # np array with all the points in this iso
    this_sorted_points=sorted_points(points)

    # problem: the points on subsequent iso's might not be going in
    # the same direction.

    # algorithm: find the closest point to point[0] on the last iso.
    # If the that point and the next one don't point in the same
    # direction as point[0] and point[1], then reverse the direction
    # of this iso.

    if i>0:
        x,y,z=this_sorted_points[0]
        xs,ys,zs=np.transpose(last_sorted_points)
        distance2=(x-xs)**2+(y-ys)**2+(z-zs)**2
        nnindex=np.where(distance2==distance2.min())[0][0]
        this_deltar=this_sorted_points[0]-this_sorted_points[-1]
        last_deltar=last_sorted_points[nnindex]-last_sorted_points[nnindex-1]
        if np.dot(this_deltar,last_deltar)<0:
            this_sorted_points=this_sorted_points[::-1]
    last_sorted_points=this_sorted_points

    iso_level=np.full((len(this_sorted_points),1),value)
    thisd=np.concatenate((this_sorted_points,iso_level),axis=1)
    if i==0:
        alld=thisd
    else:
        alld=np.concatenate((alld,thisd))
    # save individual iso's of redistributed points
    # output_file(xy,value)

with open('sortedPoints.dat','w') as outputfile:
    np.savetxt(outputfile,alld)


#with open('sortedPoints.dat') as inputfile:
#    d2=np.loadtxt(inputfile)
p=np.transpose(alld)

X=p[0]
Y=p[1]
Z=p[2]

fig=plt.figure
ax=plt.axes(projection='3d')

ax.plot(X,Y,Z)
plt.suptitle('Extracted Traces')
ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('z(m)')
plt.savefig('extracted_traces.png')
plt.show()


    
