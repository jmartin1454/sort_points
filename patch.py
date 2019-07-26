# Fri May 24 10:11:45 CDT 2019 Jeff added this line.
# June 16, 2019 Jeff modified to limit to class definitions.

from scipy.constants import mu_0, pi
import numpy as np
from Arrow3D import *

def b_segment(i,p0,p1,r):
    # p0 is one end (vector in m)
    # p1 is the other (m)
    # r is the position of interest (m)
    # i is the current (A) (flowing from p0 to p1, I think)
    d0 = r - p0
    d1 = r - p1
    ell = p1 - p0
    lend0 = np.sqrt(d0.dot(d0))
    lend1 = np.sqrt(d1.dot(d1))
    lenell = np.sqrt(ell.dot(ell))
    b_total = np.array([0,0,0])
    if(lenell!=0): # watch out for repeated points
        costheta0 = np.inner(ell,d0)/lenell/lend0
        costheta1 = -np.inner(ell,d1)/lenell/lend1
        ellcrossd0 = np.cross(ell,d0)
        lenellcrossd0 = np.sqrt(ellcrossd0.dot(ellcrossd0))
        modsintheta0 = lenellcrossd0/lenell/lend0
        a = lend0 * modsintheta0
        if(lenellcrossd0>0):
            nhat = ellcrossd0/lenellcrossd0
        else:
            nhat = np.array([0,0,0])

        if(a>0):
            b_total=mu_0*i/4.0/pi/a*(costheta0+costheta1)*nhat
        else:
            b_total = np.array([0,0,0])

    return b_total

def b_segment_2(i,p0,p1,x,y,z):
    # p0 is one end (vector in m)
    # p1 is the other (m)
    # x,y,z is the position of interest (m)
    # i is the current (A) (flowing from p0 to p1, I think)
    d0x=x-p0[0]
    d0y=y-p0[1]
    d0z=z-p0[2]
    d1x=x-p1[0]
    d1y=y-p1[1]
    d1z=z-p1[2]
    ell=p1-p0
    lend0=np.sqrt(d0x**2+d0y**2+d0z**2)
    lend1=np.sqrt(d1x**2+d1y**2+d1z**2)
    lenell=np.sqrt(ell.dot(ell))
    b_total_x=0.*x
    b_total_y=0.*y
    b_total_z=0.*z
    if(lenell!=0): # watch out for repeated points
        costheta0=(ell[0]*d0x+ell[1]*d0y+ell[2]*d0z)/lenell/lend0
        costheta1=-(ell[0]*d1x+ell[1]*d1y+ell[2]*d1z)/lenell/lend1
        ellcrossd0x=ell[1]*d0z-ell[2]*d0y
        ellcrossd0y=ell[2]*d0x-ell[0]*d0z
        ellcrossd0z=ell[0]*d0y-ell[1]*d0x
        lenellcrossd0=np.sqrt(ellcrossd0x**2+ellcrossd0y**2+ellcrossd0z**2)
        modsintheta0=lenellcrossd0/lenell/lend0
        a=lend0*modsintheta0
        nhatx=np.divide(ellcrossd0x,lenellcrossd0,out=np.zeros_like(ellcrossd0x*lenellcrossd0),where=lenellcrossd0!=0)
        nhaty=np.divide(ellcrossd0y,lenellcrossd0,out=np.zeros_like(ellcrossd0y*lenellcrossd0),where=lenellcrossd0!=0)
        nhatz=np.divide(ellcrossd0z,lenellcrossd0,out=np.zeros_like(ellcrossd0z*lenellcrossd0),where=lenellcrossd0!=0)

        pre=mu_0*i/4.0/pi*(costheta0+costheta1)
        b_total_x=np.divide(pre*nhatx,a,out=np.zeros_like(pre*nhatx),where=a!=0)
        b_total_y=np.divide(pre*nhaty,a,out=np.zeros_like(pre*nhaty),where=a!=0)
        b_total_z=np.divide(pre*nhatz,a,out=np.zeros_like(pre*nhatz),where=a!=0)

    return b_total_x,b_total_y,b_total_z

def b_loop(i,points,r):
    # i is the current (A)
    # points is a list of numpy 3-arrays defining the loop (m)
    # r is the position of interest (m)
    # returns the magnetic field as a numpy 3-array (T)
    # Note:  assumes that points[-1] to points[0] should be counted (closed loop)
    # This is why j starts at zero in the loop below
    b_total = np.array([0,0,0])
    for j in range(len(points)):
        b_total = b_total + b_segment(i,points[j-1],points[j],r)
    return b_total

def b_loop_2(i,points,x,y,z):
    # i is the current (A)
    # points is a list of numpy 3-arrays defining the loop (m)
    # x,y,z is the position of interest (m)
    # returns the magnetic field components (T)
    # Note:  assumes that points[-1] to points[0] should be counted (closed loop)
    # This is why j starts at zero in the loop below
    b_total_x=0.*x
    b_total_y=0.*y
    b_total_z=0.*z
    for j in range(len(points)):
        b_seg_x,b_seg_y,b_seg_z=b_segment_2(i,points[j-1],points[j],x,y,z)
        b_total_x=b_total_x+b_seg_x
        b_total_y=b_total_y+b_seg_y
        b_total_z=b_total_z+b_seg_z
    return b_total_x,b_total_y,b_total_z

class coil:
    def __init__(self,points,current):
        self.points = points
        self.current = current
    def set_current(self,current):
        self.current = current
    def b(self,r):
        return b_loop(self.current,self.points,r)
    def b_prime(self,x,y,z):
        return b_loop_2(self.current,self.points,x,y,z)

class coilset:
    def __init__(self):
        self.coils=[]
        self.ncoils=len(self.coils)

    def add_coil(self,points):
        c=coil(points,0.0)
        self.coils.append(c)
        self.ncoils=len(self.coils)
        
    def set_current_in_coil(self,coilnum,i):
        if(coilnum<self.ncoils):
            self.coils[coilnum].set_current(i)
        else:
            print("Error %d is larger than number of coils %d"%coilnum,self.ncoils)
            
    def set_common_current(self,i):
        for coilnum in range(self.ncoils):
            self.set_current_in_coil(coilnum,i)

    def b(self,r):
        b_total=0.
        for coilnum in range(self.ncoils):
            b_total=b_total+self.coils[coilnum].b(r)
        return b_total
        
    def b_prime(self,x,y,z):
        b_total_x=0.*x
        b_total_y=0.*y
        b_total_z=0.*z
        for coilnum in range(self.ncoils):
            b_coil_x,b_coil_y,b_coil_z=self.coils[coilnum].b_prime(x,y,z)
            b_total_x=b_total_x+b_coil_x
            b_total_y=b_total_y+b_coil_y
            b_total_z=b_total_z+b_coil_z
        return b_total_x,b_total_y,b_total_z
        
    def draw_coil(self,number,ax,style,color):
        coil = self.coils[number]
        points = coil.points
        points=np.append(points,[points[0]],axis=0) # force draw closed loop
        x = ([p[0] for p in points])
        y = ([p[1] for p in points])
        z = ([p[2] for p in points])
        ax.plot(x,y,z,style,color=color)
        a=Arrow3D([x[0],x[1]],[y[0],y[1]],[z[0],z[1]],mutation_scale=20, 
                  lw=3,arrowstyle="-|>",color="r")
        ax.add_artist(a)
        ax.text(x[0],y[0],z[0],"%d"%number,color="r")

    def draw_coils(self,ax,style='-',color='black'):
        for number in range(self.ncoils):
            self.draw_coil(number,ax,style,color)

    def output_solidworks(self,outfile):
        with open(outfile,'w') as f:
            for number in range(self.ncoils):
                coil = self.coils[number]
                points = coil.points
                firstpoint=points[0]
                lastpoint=points[-1]
                if (not(firstpoint[0]==lastpoint[0] and
                        firstpoint[1]==lastpoint[1] and
                        firstpoint[2]==lastpoint[2])):
                    points=np.append(points,[points[0]],axis=0) # force draw closed loop
                for p in points:
                    f.write("{0}\t{1}\t{2}\n".format(p[0],p[1],p[2]))

    def output_scad(self,outfile,thickness=0.001):
        with open(outfile,'w') as f:
            f.write("module line(start, end, thickness = %f) {\n"%thickness)
            f.write("hull() {\n")
            f.write("translate(start) sphere(thickness);\n")
            f.write("translate(end) sphere(thickness);\n")
            f.write("}\n")
            f.write("}\n")
            for number in range(self.ncoils):
                coil = self.coils[number]
                points = coil.points
                firstpoint=points[0]
                lastpoint=points[-1]
                if (not(firstpoint[0]==lastpoint[0] and
                        firstpoint[1]==lastpoint[1] and
                        firstpoint[2]==lastpoint[2])):
                    points=np.append(points,[points[0]],axis=0) # force draw closed loop
                f.write("// coil %d\n"%number)
                for i in range(len(points)):
                    lastpoint=points[i-1]
                    thispoint=points[i]
                    f.write("line([%f,%f,%f],[%f,%f,%f]);\n"%(lastpoint[0],lastpoint[1],lastpoint[2],thispoint[0],thispoint[1],thispoint[2]))

    def output_scad_prime(self,outfile,thickness=0.001):
        with open(outfile,'w') as f:
            f.write("thickness = %f\n"%thickness)
            f.write("translate(end) sphere(thickness);\n")
            for number in range(self.ncoils):
                coil = self.coils[number]
                points = coil.points
                firstpoint=points[0]
                lastpoint=points[-1]
                if (not(firstpoint[0]==lastpoint[0] and
                        firstpoint[1]==lastpoint[1] and
                        firstpoint[2]==lastpoint[2])):
                    points=np.append(points,[points[0]],axis=0) # force draw closed loop
                f.write("// coil %d\n"%number)
                f.write("hull() {\n")
                for p in points:
                    f.write("translate([%f,%f,%f]) sphere(thickness);\n"%(p[0],p[1],p[2]))
                f.write("}\n")

