#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:08:19 2018

@author: top40ub
"""

import numpy as np
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool, Value, Array

class Object3D:
    Object_Counter = 0
    def __del__(self):
        Object3D.Object_Counter -= 1

    def __init__(self):
        Object3D.Object_Counter +=1
        
    def trilinear_int(self,x,y,z):
        x1 = x - int(x)
        x2 = int(x) + 1 -x
        if x < 0:
            x2 = -1*x1 
            x1 = 1 + x1
        y1 = y - int(y)
        y2 = int(y) + 1 -y
        if y < 0:
            y2 = -1 * y1 
            y1 = 1 + y1
        z1 = z - int(z)
        z2= int(z)  +1 -z
        if z < 0:
            z2 = -1 * z1 
            z1 = 1 + z1    
        px1y1z1 = x1*y1*z1
        px1y2z1 = x1*y2*z1
        px1y1z2 = x1*y1*z2
        px1y2z2 = x1*y2*z2
        px2y1z1 = x2*y1*z1
        px2y2z1 = x2*y2*z1
        px2y1z2 = x2*y1*z2
        px2y2z2 = x2*y2*z2
        return px2y2z2, px2y1z2, px2y2z1 ,px2y1z1, px1y2z2, px1y1z2, px1y2z1, px1y1z1 
    
    
    def points(self,x,y,z):
        
        xp1 = int(x)
        xp2 = int(x+1)
        if x < 0:
            xp1 = int(x-1)
            xp2 = int(x)
        yp1 = int(y)
        yp2 = int(y+1)
        if y < 0:
            yp1 = int(y-1)
            yp2 = int(y)
        zp1 = int(z)
        zp2 = int(z+1)
        if z < 0:
            zp1 = int(z-1)
            zp2 = int(z)    
        p1 = np.array([xp1, yp1, zp1])
        p2 = np.array([xp1, yp2, zp1])
        p3 = np.array([xp1, yp1, zp2])
        p4 = np.array([xp1, yp2, zp2])
        p5 = np.array([xp2, yp1, zp1])
        p6 = np.array([xp2, yp2, zp1])
        p7 = np.array([xp2, yp1, zp2])
        p8 = np.array([xp2, yp2, zp2])
        
        return [p1,p2,p3,p4,p5,p6,p7,p8]
    
    def rotation(self,x,y,z):
        
        tz,tx,tz2 = np.deg2rad(self.Object_Parameter["rotation"])
 
        Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])
        Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
        Rz2 = np.array([[np.cos(tz2), -np.sin(tz2), 0], [np.sin(tz2), np.cos(tz2), 0], [0,0,1]])
        
        rot_mat = np.linalg.inv(np.dot(np.dot(Rz2,Rx), Rz)) 
        x,y,z = np.dot(rot_mat,[x,y,z])
        
        return x, y, z
    
    def rotation_inv(self,x,y,z):
        tz,tx,tz2 = np.deg2rad(self.Object_Parameter["rotation"])
 
        Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])
        Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
        Rz2 = np.array([[np.cos(tz2), -np.sin(tz2), 0], [np.sin(tz2), np.cos(tz2), 0], [0,0,1]])
        
        rot_mat = np.dot(np.dot(Rz2,Rx), Rz) 
        x,y,z = np.dot(rot_mat,[x,y,z])
        
        return x, y, z
    
    def volume_old(self):
        for index in np.ndenumerate(self.box):
            x, y, z = self.rotation_inv(index[0][0],index[0][1],index[0][2])
            point_list = self.point(x,y,z)
            for pos in point_list:
                x1, y1, z1 = pos
                if self.point_in_object(x1,y1,z1):
                    x1r, y1r, z1r = self.rotation(x1,y1,z1)
                    point_listr = self.point(x1r, y1r, z1r)
                    intpols = self.trilinear_int(x1r, y1r, z1r)
                    for posr, intpol in zip(point_listr,intpols):
                        self.box[posr]=self.box[posr]+self.Object_Parameter["color"] * intpol
    

    def vol_parallel(self,ind,v):
        
        x1, y1,z1 = ind[0][0]-v[0]/2,ind[0][1]-v[1]/2,ind[0][2]-v[2]/2
        x, y, z = self.rotation_inv(x1,y1,z1)
    
        if self.point_in_object(x,y,z):
                return [ind[0][0],ind[0][1],ind[0][2]]
    
    def volumepar(self):
        
        self.vol_parallelp = partial(self.vol_parallel,v=self.box.shape)
        with mp.Pool(processes = 30) as pool:
            L = pool.map(self.vol_parallelp, [index for index in np.ndenumerate(self.box)])
        L1 =[i for i in list(L) if i != None]
        for i in L1:
            self.box[i[0],i[1],i[2]] = self.Object_Parameter["color"]
        return self.box    
        
    
    
    
    def place_object_involume(self,volume, overwrite = False):
        if overwrite is True:
            self.vol_parallelp = partial(self.vol_parallel,v=self.box.shape)
            with mp.Pool(processes = 30) as pool:
                L = pool.map(self.vol_parallelp, [index for index in np.ndenumerate(self.box)])
            L1 =[i for i in list(L) if i != None]
            for i in L1:        
                self.pos1 = i[0] + self.Object_Parameter["pos"][0] - int(self.box.shape[0]/2)
                self.pos2 = i[1] + self.Object_Parameter["pos"][1] - int(self.box.shape[1]/2)
                self.pos3 = i[2] + self.Object_Parameter["pos"][2] - int(self.box.shape[2]/2)
                volume[self.pos1,self.pos2,self.pos3]=self.Object_Parameter["color"]
        return volume            

    def calc_dims(self):
        l_max = np.sqrt((2*self.Object_Parameter["radius_dim"][0])**2+self.Object_Parameter["radius_dim"][1]**2+self.Object_Parameter["radius_dim"][2]**2)
        e1 = np.array([self.Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],self.Object_Parameter["radius_dim"][2]])
        e2 = np.array([self.Object_Parameter["radius_dim"][0],-self.Object_Parameter["radius_dim"][1],self.Object_Parameter["radius_dim"][2]])
        e3 = np.array([-self.Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],self.Object_Parameter["radius_dim"][2]])
        e4 = np.array([-self.Object_Parameter["radius_dim"][0],-self.Object_Parameter["radius_dim"][1],self.Object_Parameter["radius_dim"][2]])
        e5 = np.array([self.Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],-self.Object_Parameter["radius_dim"][2]])
        e6 = np.array([self.Object_Parameter["radius_dim"][0],-self.Object_Parameter["radius_dim"][1],-self.Object_Parameter["radius_dim"][2]])
        e7 = np.array([-self.Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],-self.Object_Parameter["radius_dim"][2]])
        e8 = np.array([-self.Object_Parameter["radius_dim"][0],-self.Object_Parameter["radius_dim"][1],-self.Object_Parameter["radius_dim"][2]])
        
        e1r = self.rotation(*e1)
        e2r = self.rotation(*e2)
        e3r = self.rotation(*e3)
        e4r = self.rotation(*e4)
        e5r = self.rotation(*e5)
        e6r = self.rotation(*e6)
        e7r = self.rotation(*e7)
        e8r = self.rotation(*e8)
    
        max_x = max(e1r[0],e2r[0],e3r[0],e4r[0],e5r[0],e6r[0],e7r[0],e8r[0])+1
        max_y = max(e1r[1],e2r[1],e3r[1],e4r[1],e5r[1],e6r[1],e7r[1],e8r[0])+1
        max_z = max(e1r[2],e2r[2],e3r[2],e4r[2],e5r[2],e6r[2],e7r[2],e8r[2])+1
        
        return [(int(max_x)*2,int(max_y)*2,int(max_z)*2),(int(l_max*2),int(l_max*2),int(l_max*2))]

    
class Sphere(Object3D):
    Sphere_Counter = 0
    def __del__(self):
        Object3D.Object_Counter -= 1
        Sphere.Sphere_Counter -=1
    
    def __init__(self, name, radius_dim=(1,1,1), pos=(0,0,0), color=100, rotation=(0,0,0)):
        self.name = name
        Object3D.Object_Counter +=1
        Sphere.Sphere_Counter +=1
        self.Object_Parameter = {"objecttype":"sphere", "radius_dim": radius_dim, "pos":pos, "color": color, "rotation":rotation}
        self.rotated_dims = self.calc_dims()
        self.box = np.zeros(self.rotated_dims[0])
        
    def point_in_object(self, x, y, z):
        xoff, yoff, zoff =self. Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],self.Object_Parameter['radius_dim'][2]
        x_prim = (x)**2/(xoff**2)
        y_prim = (y)**2/(yoff**2)
        z_prim = (z)**2/(zoff**2)
        return x_prim + y_prim + z_prim <= 1
    
        
        
class Ellipsoid(Object3D):
    Ellipsoid_Counter = 0
    def __del__(self):
        Object3D.Object_Counter -= 1
        Sphere.Ellipsoid_Counter -=1
    
    def __init__(self, name, radius_dim=(1,1,1), pos=(0,0,0), color=100, rotation = (0,0,0)):
        self.name = name
        Object3D.Object_Counter +=1
        Sphere.Ellipsoid_Counter +=1
        self.Object_Parameter = {"objecttype":"sphere", "radius_dim": radius_dim, "pos":pos, "color": color, "rotation":rotation}
        self.rotated_dims = self.calc_dims()
        self.box = np.zeros(self.rotated_dims[0])
    
    def point_in_object(self, x, y, z):
        xoff, yoff, zoff =self. Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],self.Object_Parameter['radius_dim'][2]
        x_prim = (x)**2/(xoff**2)
        y_prim = (y)**2/(yoff**2)
        z_prim = (z)**2/(zoff**2)
        return x_prim + y_prim + z_prim <= 1



class Square(Object3D):
    Square_Counter = 0
    def __del__(self):
        Object3D.Object_Counter -= 1
        Square.Square_Counter -= 1
    
    def __init__(self, name, radius_dim = (10,10,10), pos = (0,0,0), color = 100, rotation = (0,0,0)):
        self.name = name
        Object3D.Object_Counter += 1
        Square.Square_Counter +=1
        self.Object_Parameter = {"objecttype":"square", "radius_dim": radius_dim, "pos":pos, "color": color, "rotation":rotation}
        self.rotated_dims = self.calc_dims()
        self.box = np.zeros(self.rotated_dims[0])
        
    def point_in_object(self, x, y, z):
        xoff, yoff, zoff =self. Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],self.Object_Parameter['radius_dim'][2]
        x_prim = x**2
        y_prim = y**2
        z_prim = z **2
        return x_prim <= xoff**2 and y_prim <= yoff**2 and z_prim <= zoff**2

                
class Box(Object3D):
    Box_Counter = 0
    def __del__(self):
        Object3D.Object_Counter -= 1
        Box.Box_Counter -= 1
    
    def __init__(self, name, radius_dim = (10,10,10), pos = (0,0,0), color = 100, rotation = (0,0,0)):
        self.name = name
        Object3D.Object_Counter += 1
        Box.Box_Counter +=1
        self.Object_Parameter = {"objecttype":"box", "radius_dim": radius_dim, "pos":pos, "color": color, "rotation":rotation}
        self.rotated_dims = self.calc_dims()
        self.box = np.zeros(self.rotated_dims[0])
        
    def point_in_object(self, x, y, z):
        xoff, yoff, zoff =self. Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],self.Object_Parameter['radius_dim'][2]
        x_prim = x**2
        y_prim = y**2
        z_prim = z**2
        return x_prim <= xoff**2 and y_prim <= yoff**2 and z_prim <= zoff**2



class Tubus(Object3D):
    Tubus_Counter = 0
    def __del__(self):
        Object3D.Object_Counter -= 1
        Tubus.Tubus_Counter -= 1
    
    def __init__(self, name, radius_dim = (10,10,10), pos = (0,0,0), color = 100, rotation = (0,0,0)):
        self.name = name
        Object3D.Object_Counter += 1
        Box.Box_Counter +=1
        self.Object_Parameter = {"objecttype":"tubus", "radius_dim": radius_dim, "pos":pos, "color": color, "rotation":rotation}
        self.rotated_dims = self.calc_dims()
        self.box = np.zeros(self.rotated_dims[0])
        
    def point_in_object(self, x, y, z):
        xoff, yoff, zoff =self. Object_Parameter["radius_dim"][0],self.Object_Parameter["radius_dim"][1],self.Object_Parameter['radius_dim'][2]
        x_prim = (x)**2/(xoff**2)
        y_prim = (y)**2/(yoff**2)
        z_prim = (z)**2 
        return x_prim + y_prim <= 1 and z_prim <= zoff**2

class Helix(Object3D):
    pass

class Cone(Object3D):
    pass
    
class Pyramide3(Object3D):
    pass

class Pyramide4(Object3D):
    pass

if __name__=='__main__':
    volume = np.zeros((150,150,150))
    tb1 = Tubus('tubus1', radius_dim = (7,7,20), pos=(75,75,20), color = 255, rotation = (0,0,0))
    tb2 = Tubus('tubus2', radius_dim = (7,7,20), pos=(65,75,55),color = 255, rotation = (90,135,0))
    tb3 = Tubus('tubus3', radius_dim = (7,7,20), pos=(85,75,55),color = 255, rotation = (90,225,0))
    
    tb4 = Tubus('tubus1', radius_dim = (7,7,20), pos=(50,75,85), color = 255, rotation = (0,0,0))
    tb5 = Tubus('tubus2', radius_dim = (7,7,20), pos=(40,75,120),color = 255, rotation = (90,135,0))
    tb6 = Tubus('tubus3', radius_dim = (7,7,20), pos=(60,75,120),color = 255, rotation = (90,225,0))
    tb7 = Tubus('tubus1', radius_dim = (7,7,20), pos=(100,75,85), color = 255, rotation = (0,0,0))
    tb8 = Tubus('tubus2', radius_dim = (7,7,20), pos=(90,75,120),color = 255, rotation = (90,135,0))
    tb9 = Tubus('tubus3', radius_dim = (7,7,20), pos=(110,75,120),color = 255, rotation = (90,225,0))
    """
    tb1i = Tubus('tubus1', radius_dim = (3,3,20), pos=(75,75,20), color = 0, rotation = (0,0,0))
    tb2i = Tubus('tubus2', radius_dim = (3,3,20), pos=(60,75,55),color = 0, rotation = (90,135,0))
    tb3i = Tubus('tubus3', radius_dim = (3,3,20), pos=(90,75,55),color = 0, rotation = (90,225,0))
    tb4i = Tubus('tubus1', radius_dim = (3,3,20), pos=(45,75,85), color = 0, rotation = (0,0,0))
    tb5i = Tubus('tubus2', radius_dim = (3,3,20), pos=(30,75,120),color = 0, rotation = (90,135,0))
    tb6i = Tubus('tubus3', radius_dim = (3,3,20), pos=(60,75,120),color = 0, rotation = (90,225,0))
    tb7i = Tubus('tubus1', radius_dim = (3,3,20), pos=(105,75,85), color = 0, rotation = (0,0,0))
    tb8i = Tubus('tubus2', radius_dim = (3,3,20), pos=(90,75,120),color = 0, rotation = (90,135,0))
    tb9i = Tubus('tubus3', radius_dim = (3,3,20), pos=(120,75,120),color = 0, rotation = (90,225,0))
    """
    volume = tb1.place_object_involume(volume, overwrite = True)
    volume = tb2.place_object_involume(volume, overwrite = True)
    volume = tb3.place_object_involume(volume, overwrite = True)
  
    volume = tb4.place_object_involume(volume, overwrite = True)
    volume = tb5.place_object_involume(volume, overwrite = True)
    volume = tb6.place_object_involume(volume, overwrite = True)
    volume = tb7.place_object_involume(volume, overwrite = True)
    volume = tb8.place_object_involume(volume, overwrite = True)
    volume = tb9.place_object_involume(volume, overwrite = True)
   
    """
    volume = tb1i.place_object_involume(volume, overwrite = True)
    volume = tb2i.place_object_involume(volume, overwrite = True)
    volume = tb3i.place_object_involume(volume, overwrite = True)
    volume = tb4i.place_object_involume(volume, overwrite = True)
    volume = tb5i.place_object_involume(volume, overwrite = True)
    volume = tb6i.place_object_involume(volume, overwrite = True)
    volume = tb7i.place_object_involume(volume, overwrite = True)
    volume = tb8i.place_object_involume(volume, overwrite = True)
    volume = tb9i.place_object_involume(volume, overwrite = True)
    """    
    
    
    
    import tifffile as tiff
    image = volume
    #image = np.uint8(volume)
    tiff.imsave('volume_test_tuben.tif',np.float32(image))