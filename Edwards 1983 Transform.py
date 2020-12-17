# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:25:21 2020

@author: Norm
"""
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


class halfplane:
    def __init__(self, strike, dip, electrode_point, plunge_azimuth=None):
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
        self.electrode_point = electrode_point
        
        # Get plane offset
        # Form: ax + by + cz = d
        self.d = np.dot(self.normal, electrode_point)
                                       
        if plunge_azimuth is None:
            plunge_azimuth = (strike + 180) % 360
        # Get the y' axis that is along the line of the halfplane
        self.plunge_vector = self._project_plane(plunge_azimuth)
        print(f"Plunge Line Vector: {self.plunge_vector}")
        if self.plunge_vector[2] > 0:
            Warning("Halfplane plunge vector is oriented up. Config is a halfplane infinitely UPWARDS, are you sure?")
        if self.normal[2] < 0:
            self.normal *= -1
            
        # Debug Plots
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-1,1)
        self.ax.set_ylim(-1,1)
        self.ax.set_zlim(-1,1)
        
        self.basis = np.identity(3)
        self.ax.quiver(0,0,0, self.plunge_vector[0], self.plunge_vector[1], self.plunge_vector[2])
        self._calculate_rotation()
        # Defines the rotation to the frame of reference used for all the math
        # Note that this frame of reference is different from the one in Pai and
        # Yeoh 1993
        # 1.) Rotate y axis about z to the plunge vector projected onto x-y
        # 2.) Rotate y' axis about x' axis to the plunge vector
        # 3.) Rotate about y'' such that the Z vector lies on the plane
        
        
    def _calculate_rotation(self):
        """
        Return the scipy rotation to rotate to the coordinate frame

        Returns
        -------
        scipy.spatial.transform.Rotation instance

        """
        
         # 1.) Rotate y axis about z to the plunge vector projected onto x-y
         
        rot1 = self._angle_2V(self.plunge_vector[0:2], [0,1])
        
        if self.plunge_vector[0] > 0:
            # Clockwise rotation so we have to flip the sign of the rotation
            rot1 *= -1
            
        # Debug checkpoint rot1
        r1 = R.from_euler('z', [rot1])
        v = [0,1,0]
        result_r1 = r1.apply(v).squeeze()
        
        # 2.) Rotate y' axis about x' axis to the plunge vector
        
        rot2 = self._angle_2V(result_r1, self.plunge_vector) 
        print(f"R2: Rot2 {rot2 * 180 / np.pi}")
        
        # Check if the plunge is upwards or downwards 
        if self.plunge_vector[2] < 0:
            rot2 *= -1
            
        # Debug checkpoint rot2
        r2 = R.from_euler('ZX', [rot1, rot2])
        v = [1,0,0]
        result_r2 = r1.apply(v).squeeze()
        # 3.) Rotate about y'' such that the Z vector lies on the plane
        # by checking the angle between x'' and the normal
        rot3 = self._angle_2V(result_r2, self.normal) + np.pi / 2
        r3 = R.from_euler('ZXY', [rot1, rot2, -rot3])
        self.rotation = r3
        self.basis = r3.apply(self.basis)
        
    def debug_MPL_plot_plane(self):
        """
        Plots the plane and the basis vectors on the matplotlib axis in ax3d

        Parameters
        ----------
        ax3d : matplotlib axis object
            Must have 3d axis enabled

        Returns
        -------
        None.

        """
        const_offset = 1
        x = np.linspace(self.electrode_point[0] - const_offset, 
                        self.electrode_point + const_offset,
                        num=5)
        y = np.linspace(self.electrode_point[1] - const_offset, 
                        self.electrode_point + const_offset,
                        num=5)
        
        # Plot the plane in the area around the electrode
        xx, yy = np.meshgrid(x, y)
        zz = self.z_plane(xx, yy)
        
        self.ax.scatter(xx, yy, zz)
        
        x, y, z = self.electrode_point
        # Plot the basis vectors
        self.ax.quiver(x, y, z, self.basis[0][0], self.basis[0][1], self.basis[0][2], color='red')
        self.ax.quiver(x, y, z, self.basis[1][0], self.basis[1][1], self.basis[1][2], color='green')
        self.ax.quiver(x, y, z, self.basis[2][0], self.basis[2][1], self.basis[2][2], color='blue')
        #self.ax.quiver(x, y, z, self.normal[0], self.normal[1], self.normal[2]) 
        
    @staticmethod
    def _angle_2V(v1, v2):
        a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if abs(a - 1) < 1e-9:
            return 0
        elif abs(a + 1) < 1e-9:
            return np.pi()
        else:
            return np.arccos(a)

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
        try:
            assert np.dot(projected, self.normal) < 1e-6
        except AssertionError:
            raise "Dot product suggests computed plunge line not on plane???"
            
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
            np.cos(r_dip)
        ])
        
        n /= np.linalg.norm(n)
        return n
    
    def rotate(self, vect, invert_rot=False):
        """
        Rotate the row vector vect

        Parameters
        ----------
        vect : ndarray
            Vector to rotate to the coordinate system in Edwards 1983

        Returns
        -------
        ndarray
        Rotated Vector
        """
        self.rotation.apply(vect, inverse=invert_rot)
        
def Edwards1983_BField(halfplane):
    def __init__(self, strike, dip, electrode_point, plunge_azimuth):
        super().__init__(strike, dip, electrode_point, plunge_azimuth)
    
    def BField(self, pos_vect):
        """
        Calculates the BField given the position (in global cartesian)
        given the plane initialized in the class

        Parameters
        ----------
        pos_vect : ndarray
            Position of the point to evaluate the field

        Returns
        -------
        ndarray
        BField calculated at pos_vect in Tesla
        """
        normalized_pos = pos_vect - self.electrode_point
        # Localized and rotated position vector
        local_rot_pos = self.rotate(normalized_pos)
        # Equations from Edwards 1983
        x, y, z = local_rot_pos
        w = x * np.cos(self.rad_dip) + z * np.sin(self.rad_dip)
        r = np.sqrt(np.pow(w, 2) + np.pow(y, 2))
        v = x * np.sin(self.rad_dip) - z * np.cos(self.rad_dip)
        phi = np.arctan(w/y)                
        
        
BH_AZIMUTH = 50
# Try a line from [0,0,0] with length 1 and turn it into an object for plotting
loc_electrode = np.array([0,0,0])

myhalfplane = halfplane(strike=280, 
                        dip=75, 
                        electrode_point=loc_electrode)

unit_vects = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])

myhalfplane.debug_MPL_plot_plane()

plt.show()
