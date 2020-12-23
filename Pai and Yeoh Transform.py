# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:36:10 2020

@author: Norm
"""
from mpl_toolkits.mplot3d import Axes3D
from Edwards1983Transform import Edwards1983_BField as Edwards1983Field
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import simps

class halfplane:
    def __init__(self, strike, dip, electrode_point, line_point, plunge_azimuth=None):
        """
        strike: Strike of the half plane in degrees
        dip: Dip of the half plane in  degrees
        electrode_point: Point on the plane that will be used to define the origin
                         where the current is injected
        line_point: Point that defines the line that defines the halfspace
        plunge_azimuth: Azimuth in degrees of the edge of the halfplane. Our y axis will align
                        with this edge
        """
        # The Edwards 1983 Plane is required for calculating the integral
        # over the plate for cases where electrode is touching the plane
        self.edwardsPlane = Edwards1983Field(strike, 
                                             dip, 
                                             electrode_point, 
                                             plunge_azimuth)
        # Debug Plots
        self.fig = plt.figure('Pai and Yeoh')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X'), self.ax.set_ylabel('Y'), self.ax.set_zlabel('Z')
        
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
        self.d = np.dot(self.normal, line_point)
        
        if plunge_azimuth is None:
            plunge_azimuth = (strike + 180) % 360
            
        # Get the y' axis that is along the line of the halfplane
        self.plunge_vector = self._project_plane(plunge_azimuth)
        print(f"Plunge Line Vector: {self.plunge_vector}")
        if self.plunge_vector[2] > 0:
            Warning("Halfplane plunge vector is oriented up. Config is a halfplane infinitely UPWARDS, are you sure?")
        if self.normal[2] < 0:
            self.normal *= -1
            
        self.basis = np.identity(3)
        self.ax.quiver(0,0,0, self.plunge_vector[0], self.plunge_vector[1], self.plunge_vector[2])
        
        # Calculate the line that goes through line_point and the plane that goes
        # through the electrode and the line
        self._calculate_origin(line_point)
        self._calculate_rotation()
        
    def _calculate_origin(self, line_pt):
        """
        Calculate the origin point given one point that defines the line

        Parameters
        ----------
        line_pt : ndarray
            A point on the line that defines the line of the halfplane

        Returns
        -------
        None.
        Adds a parameter to self.origin

        """
        
        AP = self.electrode_point - line_pt
        AB = self.plunge_vector
        self.origin = line_pt + np.dot(AP, AB) / np.dot(AB, AB) * AB
        print(f"Origin point: {self.origin}")
        self.ax.scatter(self.origin[0], self.origin[1], self.origin[2], color='red') 
    
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
    
    def _calculate_rotation(self):
        """
        Return the scipy rotation to rotate to the coordinate frame
        IN PROGRESS
        
        Returns
        -------
        scipy.spatial.transform.Rotation instance

        """
        
        # Calculate the angle
        rot1 = self._angle_2V(self.plunge_vector[0:2], [0,-1])
        
        print(f"Rotation around to plunge: {rot1 * 180 / np.pi}")
        if self.plunge_vector[0] < 0:
            rot1 *= -1
            
        # 1.) Rotate Z Axis down to the xy plane 
        rotation = R.from_euler('XY', [np.pi/2, rot1])
        
        # Calculate the rotation down to the plane specified in the class
        xhat = np.array([1,0,0])
        xhat_rot = rotation.apply(xhat)
        a = np.arctan2(xhat_rot[0], xhat_rot[1])
        
        if a < 0: a += np.pi * 2
        
        azimuth = 180 / np.pi * a
        print(f"{xhat_rot}")
        print(f"Calculated rotated plunge vector azimuth {azimuth}")
        rot3 = self._angle_2V(self._project_plane(azimuth), xhat_rot)
        print(f"Rotation down to plunge: {rot3 * 180 / np.pi}")
        rotation = R.from_euler('XYZ', [np.pi/2, rot1, -rot3])
        
        self.rotation = rotation
        self.basis = rotation.apply(self.basis)
        
         
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
        x = np.linspace(self.origin[0] - const_offset, 
                        self.origin + const_offset,
                        num=4)
        y = np.linspace(self.origin[1] - const_offset, 
                        self.origin + const_offset,
                        num=4)
        
        # Plot the plane in the area around the electrode
        xx, yy = np.meshgrid(x, y)
        zz = self.z_plane(xx, yy)
        
        self.ax.scatter(xx, yy, zz)
        
        #x, y, z = self.electrode_point
        # Plot the basis vectors
        self.ax.quiver(self.origin[0], self.origin[1], self.origin[2], 
                       self.basis[0][0], self.basis[0][1], self.basis[0][2], color='red')
        self.ax.quiver(self.origin[0], self.origin[1], self.origin[2], 
                       self.basis[1][0], self.basis[1][1], self.basis[1][2], color='green')
        self.ax.quiver(self.origin[0], self.origin[1], self.origin[2], 
                       self.basis[2][0], self.basis[2][1], self.basis[2][2], color='blue')
        #self.ax.quiver(x, y, z, self.normal[0], self.normal[1], self.normal[2]) 
        
    @staticmethod
    def _angle_2V(v1, v2):
        a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if abs(a - 1) < 1e-9:
            return 0
        elif abs(a + 1) < 1e-9:
            return np.pi
        else:
            return np.arccos(a)
        
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
        return self.rotation.apply(vect, inverse=invert_rot)
    
    def mag_field_onplate(self, X, Y, Z, tx_current):
        """
        Calculate the BField of a halfplane given an electrode that resides on the plane

        Parameters
        ----------
        X : Global X Coordinate
        
        Y : Global Y Coordinate
        
        Z : Global Z Coordinate


        Returns
        -------
        float, float, float
            Calculated BField in Global Coordinates

        """
        # All coordinates that are used in this function need to be transformed into the local coordinates for the 
        # computations required
        u0 = 1.25663706e-6
        
        # We have to transform the input coordinates into the local coordinate system
        electrode_coord_local = self.rotate(self.electrode_point - self.origin)
        x0 = electrode_coord_local[0]
        z0 = electrode_coord_local[2]
        
        def Bx():
            t = np.linspace(0, np.pi / 2, num=200)
            frac_bot1 = np.power(np.abs(Y) + x0 * np.sin(t), 2) + \
                        np.power(X * np.cos(t) - (Z - z0) * np.sin(t), 2)
            frac_bot2 = np.power(np.abs(Y) + x0 * np.sin(t), 2) + \
                        np.power(X * np.cos(t) + (Z - z0) * np.sin(t), 2)
            frac_top = np.abs(Y) + x0 * np.sin(t)
            y = np.cos(t) * (frac_top / frac_bot1 - frac_top / frac_bot2)
            area = simps(y, t)
            return area * np.sign(Y) * -1 * u0 / 4 / np.pi ** 2
        
        def Bz():
            t = np.linspace(0, np.pi / 2, num=200)
            frac_bot1 = np.power(np.abs(Y) + x0 * np.sin(t), 2) + \
                        np.power(X * np.cos(t) - (Z - z0) * np.sin(t), 2)
            frac_bot2 = np.power(np.abs(Y) + x0 * np.sin(t), 2) + \
                        np.power(X * np.cos(t) + (Z - z0) * np.sin(t), 2)
            frac_top = np.abs(Y) + x0 * np.sin(t)
            y = np.sin(t) * (frac_top / frac_bot1 + frac_top / frac_bot2)
            area = simps(y, t)
            return area * np.sign(Y) * u0 / 4 / np.pi ** 2
        
        def By():
            t = np.linspace(0, np.pi / 2, num=200)
            
            frac_top1 = X * np.cos(t) - (Z - z0) * np.sin(t)
            frac_top2 = X * np.cos(t) + (Z - z0) * np.sin(t)
            frac_bot1 = np.power(np.abs(Y) + x0 * np.sin(t), 2) + \
                        np.power(X * np.cos(t) - (Z - z0) * np.sin(t), 2)
            frac_bot2 = np.power(np.abs(Y) + x0 * np.sin(t), 2) + \
                        np.power(X * np.cos(t) + (Z - z0) * np.sin(t), 2)
            
            y = frac_top1 / frac_bot1 - frac_top2 / frac_bot2
            area = simps(y, t)
            return area * u0 / 4 / np.pi ** 2
    
        global_field = self.rotate(np.array([Bx(), By(), Bz()]) * tx_current, 
                                   invert_rot=True)
        final_vect = global_field + self.edwardsPlane.BField(X, Y, Z, tx_current)
            
        return final_vect[0], final_vect[1], final_vect[2]
    
    def plot_field(self, xx, yy, zz, tx_current):
        """
        Plot the field on the internal class matplotlib plot

        Parameters
        ----------
        xx : ndarray
            X coordinates to evaluate at
        yy : ndarray
            Y coordinates to evaluate at
        zz : ndarray
            Z coordinates to evaluate at
        tx_current : float
            Transmitter current in amps

        Returns
        -------
        None.

        """
        vField = np.vectorize(self.mag_field_onplate, excluded=[3])
        
        fieldx, fieldy, fieldz = vField(xx, yy, zz, tx_current)
        
        self.ax.quiver(xx, yy, zz, fieldx, fieldy, fieldz, normalize=True, length=1, color='red')
    
    def current_sink(self, X, Y, Z):
        """
        Calculate the current sink given a point X, Y, Z that lies within 
        the halfspace specified in the class

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Y : TYPE
            DESCRIPTION.
        Z : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        pt = np.array([X,Y,Z])
        # Make sure that the provided point is in the halfplane
        assert self.in_space(pt)
        
        electrode_coord_local = self.rotate(self.electrode_point - self.origin)
        eval_coord_local = self.rotate(pt - self.origin)
        x0 = eval_coord_local[0]
        z0 = eval_coord_local[2]
        
        # Source electrode cylindrical position
        r = np.linalg.norm(eval_coord_local)
        r_s = np.linalg.norm(electrode_coord_local)
        
        # Absolute distance between electrode and evaluation point
        R = np.linalg.norm(electrode_coord_local - eval_coord_local)
        if r_s < 1e-8:
            alpha_s = 0  
        else: 
            alpha_s = self._angle_2V(electrode_coord_local, 
                                     np.array([1,0,0]))
            if electrode_coord_local[1] < 0: alpha_s = 360 - alpha_s 
                
        if r < 1e-8:
            alpha = 0  
        else: 
            alpha = self._angle_2V(eval_coord_local, 
                                   np.array([1,0,0]))
            if eval_coord_local[1] < 0: alpha = 360 - alpha 
        
        
        v1 = 2 * np.sqrt(x0 * r_s) * np.cos(alpha_s / 2)
        c = np.sqrt(R ** 2 + z0 ** 2)
        s1 = np.pi ** -2 * ((x0 + r_s) ** 2 + z0 ** 2) ** -1 * \
             np.sqrt(r_s / x0) * np.sin(alpha_s / 2)
        s2 = r_s * np.sin(alpha_s) * np.pi ** -2 * (v1 / c ** 2 / (c ** 2 + v1 ** 2) + \
                                                    np.arctan(v1 / c) / c ** 3)
        return s1, s2
       

borehole_stations = np.array([[-10, -10,   0],
                              [-10, -10, -10],
                              [-10, -10, -20],
                              [-10, -10, -30],
                              [-10, -10, -40],
                              [-10, -10, -50]]
                              )    
loc_electrode = np.array([0, 0, -10])
loc_line_pt = np.array([1,0,-5])
myhalfplane = halfplane(strike=90, dip=50, 
                        electrode_point=loc_electrode,
                        line_point=loc_line_pt
                        )

myhalfplane.ax.scatter(loc_electrode[0], 
                       loc_electrode[1], 
                       loc_electrode[2], color='black')

myhalfplane.ax.scatter(loc_line_pt[0], 
                       loc_line_pt[1], 
                       loc_line_pt[2], color='green')
myhalfplane.debug_MPL_plot_plane()

xx, yy, zz = np.meshgrid(np.linspace(-10,10,num=6), 
                         np.linspace(-10,10,num=6), 
                         np.linspace(-10,10,num=6))


myhalfplane.plot_field(xx, yy, zz, 1)
plt.show()





