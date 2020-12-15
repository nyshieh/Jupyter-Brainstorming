# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:25:21 2020

@author: Norm
"""
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.integrate import simps

class halfplane:
    def __init__(self, strike, dip, electrode_point, plunge_azimuth):
        """
        strike: Strike of the half plane in degrees
        dip: Dip of the half plane in  degrees
        electrode_point: Point on the plane that will be used to define the origin
                         where the current is injected
        plunge_azimuth: Azimuth in degrees of the edge of the halfplane. Our y axis will align
                        with this edge
        """
        
        rad_strike = np.pi * strike / 180
        rad_dip = np.pi * dip / 180
        self.rad_strike = rad_strike
        self.rad_dip = rad_dip
        self.strike = strike
        self.dip = dip
        self.normal = self.fnormal(strike, dip)
        
        # Get plane offset
        # Form: ax + by + cz = d
        self.d = np.dot(self.normal, electrode_point)
        
        # Init unit axis vectors
        self.x_h, self.y_h, self.z_h = np.array([1,0,0]), \
                                       np.array([0,1,0]), \
                                       np.array([0,0,1])
                                       
        # Get the y' axis that is along the line of the halfplane
        self.plunge_vector = self._project_plane(plunge_azimuth)
        
        if self.plunge_vector[2] > 0:
            self.plunge_vector *= -1
            
        self.lcoal_z = np.cross(self.normal, self.plunge_vector)
        
        # Defines the rotation to the frame of reference used for all the math
        # Note that this frame of reference is different from the one in Pai and
        # Yeoh 1993
        # 1.) Rotate y axis about z to the plunge vector projected onto x-y
        # 2.) Rotate y' axis about x' axis to the plunge vector
        # 3.) Rotate about y'' such that the Z vector lies on the plane
        
        #self.rotation = R.from_euler()
    
    def _project_plane(self, azimuth):
        """
        Parameters
        ----------
        azimuth : float
            The azimuth of the line to project down/up to the plane defined in
            this class
    
        Returns
        -------
        np.array()

        """
        azi_rad = azimuth * np.pi / 180
        
        xy_proj = np.array([np.sin(azi_rad), np.cos(azi_rad), 0])
        xy_proj /= np.linalg.norm(xy_proj)
        projected = xy_proj - np.dot(xy_proj, self.normal) * self.normal
        
        # Make sure the vector is on the plane
        assert np.dot(projected, self.normal) < 1e-6
        
        return projected
    
    def z_plane(self, x, y):
        """
        Returns the z coordinates of the plane

        Parameters
        ----------
        x : ndarray of any shape
            x coordinates to evaluate at
        y : ndarray of any shape
            y coordinates to evaluate at
        x, y must be of same shape
        Returns
        -------
        ndarray

        """
        z = (self.d - self.normal[0] * x - self.normal[1] * y) / self.normal[2]
        return z
    
    def in_space(self, pt):
        """
        Returns bool value if pt is within the half plane defined by the class
        """
        # This function below evaluates z coordinate of self.div_line at point [x, y]
        z_func_line = lambda x, y: -np.dot(np.array([x, y]), self.div_line[:2]) / self.div_line[2]
        if np.dot(self.normal, pt) < 1e-6 and pt[2] < z_func_line(pt[0], pt[1]):
            return True
        else:
            return False
    
    def fnormal(self, strike, dip):
        """
        Returns the normal vector given strike and dip
        """
        deg_to_rad = lambda x: x * np.pi / 180
        r_strike = deg_to_rad(strike)
        r_dip = deg_to_rad(dip)
        n = np.array([
            np.sin(r_dip) * np.cos(r_strike),
            -np.sin(r_dip) * np.sin(r_strike),
            -np.cos(r_dip)
        ])
        
        n /= np.linalg.norm(n)
        return n

# Convenience function init

def line_output(start, length, deg_azimuth, deg_dip):
    """
    Calculate the endpoint of a line from start and endpoint specified by 
    length, deg_azimuth, and deg_dip
    """
    start = np.array(start)
    azi = deg_azimuth * np.pi / 180
    phi = (deg_dip + 90) * np.pi / 180
    # Translate out of spherical coordinates
    x,y,z = np.sin(azi), np.cos(azi), np.cos(phi)
    dir_hat = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    direction = length * dir_hat
    endpoint = start + direction
    return endpoint

BH_AZIMUTH = 50
# Try a line from [0,0,0] with length 1 and turn it into an object for plotting
loc_electrode = np.array([0,0,0])
line_seg_end = line_output(loc_electrode, 1, BH_AZIMUTH, 40)

myhalfplane = halfplane(45, 
                        dip=50, 
                        electrode_point=loc_electrode, 
                        plunge_azimuth=30)

unit_vects = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])


x = y = z = [0,0,0]
u, v, w = unit_vects[:,0], unit_vects[:,1], unit_vects[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x, y, z, u, v, w)

plt.show()
