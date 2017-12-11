#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


def plot_closest_points(elev, azim):
	deg2rad = (2*np.pi) / 360
	c_pts = get_cartesian_samplepoints()
	dsts = get_point_distances_sphere(deg2rad*elev, deg2rad*azim, c_pts);
	plot_points_3d(dsts, c_pts)

'''
TODO
find an algorithm to find three points from which we can interpolate to
a given point p, such that p is 'inside' the triangle formed by the
other three points

keywords
delaunay interpolation
barycentric coordinates
'''
def plot_points_3d(point_distances, cart_points):
	# find the closest points
	sorted_points_indices = np.argpartition(point_distances, 3)
	closest_points_indices = sorted_points_indices[:3]
	other_points_indices = sorted_points_indices[3:]
	close_pts = cart_points[closest_points_indices]
	other_pts = cart_points[other_points_indices]

	(xs, ys, zs) = (cart_points[:,0], cart_points[:,1], cart_points[:,2])

	# prepare plot
	fig = pl.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(close_pts[:,0], close_pts[:,1], close_pts[:,2], c='red')
	ax.scatter(other_pts[:,0], other_pts[:,1], other_pts[:,2], c='blue')
	pl.show()

def get_point_distances_sphere(elev, azim, cart_points):
	'''
	Same as get_point_distances_cartesian but takes polar coordinates (1, elev, azim)

	-pi/2 <= elev <= pi/2
	0 <= azim <= 2pi

	cart_points should be the array returned by get_cartesian_samplepoints
	'''

	point_cartesian = np.array(
			[
				-np.sin(azim) * np.cos(elev),
				np.cos(azim) * np.cos(elev),
				np.sin(elev)
			], dtype=np.float32)

	distances = get_point_distances_cartesian(point_cartesian, cart_points)
	return distances

def get_point_distances_cartesian(point_cartesian, cart_points):
	'''
	Return an array of shape (187,) containing the L2 distances of
	the point (x, y, z) to every sample point in cart_points

	point_cartesian: np.array with shape (3,)
	cart_points: array returned by get_cartesian_samplepoints
	'''

	vector_diffs = cart_points - point_cartesian
	distances = np.linalg.norm(vector_diffs, ord=2, axis=1)
	return distances

def azim_to_interpolation_params_deg(elev, azim):
	deg2rad = 2 * np.pi / 360
	return azim_to_index(elev * deg2rad, azim * deg2rad)

def azim_to_interpolation_params(elev, azim):
	'''
	Return the tuple (before, a, after) such that the angle azim
	corresponds to the index (1-a) * before + a * after
	elev, azim: polar angles
	elev needs to be an angle from the database: [-45,-30,-15,0,15,30,45,60,75,90] .* (2pi / 360)
	azim can technically be arbitrary but will be used mod 2pi
	'''
	azim = azim % (2*np.pi)
	assert azim >= 0
	elev = np.clip(elev, -np.pi/4, np.pi/2)

	float_radian_tolerance = 0.00001

	if abs(elev - np.pi/2) < float_radian_tolerance:
		return (186, 0., 186)

	# TODO
	# index_elev_azim = get_index_elev_azim()
	# only keep entries with the correct elevation
	index_azim = index_elev_azim[np.where(np.abs(index_elev_azim[:,1] - elev) < float_radian_tolerance)][:,[0,2]]

	if (index_azim.size == 0):
		raise ValueError('ele must be one of the values in the database: [-45,-30,-15,0,15,30,45,60,75,90] .* (2pi / 360)')

	before = int(0.5 + index_azim[np.where(index_azim[:,1] <= azim)][:,0].max())
	try:
		after = int(0.5 + index_azim[np.where(index_azim[:,1] > azim)][:,0].min())
	except ValueError:
		# if there is no index with azimuth > azim, then we're at the last entry of the array
		# so we can roll over to the first one
		after = int(0.5 + index_azim[0,0])


	before_azim = index_elev_azim[before,2]
	after_azim = index_elev_azim[after,2]

	if after_azim < before_azim:
		assert after_azim == 0
		after_azim = 2 * np.pi

	a = (azim - before_azim) / (after_azim - before_azim)

	return (before, a, after)

# {{{
def get_index_elev_azim():
	# differences to ./recherche.ircam.fr/indices:
	# - zero-based indexing (every index is one less)
	index_elev_azim = np.array([
	[0,-45,0],
	[1,-45,15],
	[2,-45,30],
	[3,-45,45],
	[4,-45,60],
	[5,-45,75],
	[6,-45,90],
	[7,-45,105],
	[8,-45,120],
	[9,-45,135],
	[10,-45,150],
	[11,-45,165],
	[12,-45,180],
	[13,-45,195],
	[14,-45,210],
	[15,-45,225],
	[16,-45,240],
	[17,-45,255],
	[18,-45,270],
	[19,-45,285],
	[20,-45,300],
	[21,-45,315],
	[22,-45,330],
	[23,-45,345],
	[24,-30,0],
	[25,-30,15],
	[26,-30,30],
	[27,-30,45],
	[28,-30,60],
	[29,-30,75],
	[30,-30,90],
	[31,-30,105],
	[32,-30,120],
	[33,-30,135],
	[34,-30,150],
	[35,-30,165],
	[36,-30,180],
	[37,-30,195],
	[38,-30,210],
	[39,-30,225],
	[40,-30,240],
	[41,-30,255],
	[42,-30,270],
	[43,-30,285],
	[44,-30,300],
	[45,-30,315],
	[46,-30,330],
	[47,-30,345],
	[48,-15,0],
	[49,-15,15],
	[50,-15,30],
	[51,-15,45],
	[52,-15,60],
	[53,-15,75],
	[54,-15,90],
	[55,-15,105],
	[56,-15,120],
	[57,-15,135],
	[58,-15,150],
	[59,-15,165],
	[60,-15,180],
	[61,-15,195],
	[62,-15,210],
	[63,-15,225],
	[64,-15,240],
	[65,-15,255],
	[66,-15,270],
	[67,-15,285],
	[68,-15,300],
	[69,-15,315],
	[70,-15,330],
	[71,-15,345],
	[72,0,0],
	[73,0,15],
	[74,0,30],
	[75,0,45],
	[76,0,60],
	[77,0,75],
	[78,0,90],
	[79,0,105],
	[80,0,120],
	[81,0,135],
	[82,0,150],
	[83,0,165],
	[84,0,180],
	[85,0,195],
	[86,0,210],
	[87,0,225],
	[88,0,240],
	[89,0,255],
	[90,0,270],
	[91,0,285],
	[92,0,300],
	[93,0,315],
	[94,0,330],
	[95,0,345],
	[96,15,0],
	[97,15,15],
	[98,15,30],
	[99,15,45],
	[100,15,60],
	[101,15,75],
	[102,15,90],
	[103,15,105],
	[104,15,120],
	[105,15,135],
	[106,15,150],
	[107,15,165],
	[108,15,180],
	[109,15,195],
	[110,15,210],
	[111,15,225],
	[112,15,240],
	[113,15,255],
	[114,15,270],
	[115,15,285],
	[116,15,300],
	[117,15,315],
	[118,15,330],
	[119,15,345],
	[120,30,0],
	[121,30,15],
	[122,30,30],
	[123,30,45],
	[124,30,60],
	[125,30,75],
	[126,30,90],
	[127,30,105],
	[128,30,120],
	[129,30,135],
	[130,30,150],
	[131,30,165],
	[132,30,180],
	[133,30,195],
	[134,30,210],
	[135,30,225],
	[136,30,240],
	[137,30,255],
	[138,30,270],
	[139,30,285],
	[140,30,300],
	[141,30,315],
	[142,30,330],
	[143,30,345],
	[144,45,0],
	[145,45,15],
	[146,45,30],
	[147,45,45],
	[148,45,60],
	[149,45,75],
	[150,45,90],
	[151,45,105],
	[152,45,120],
	[153,45,135],
	[154,45,150],
	[155,45,165],
	[156,45,180],
	[157,45,195],
	[158,45,210],
	[159,45,225],
	[160,45,240],
	[161,45,255],
	[162,45,270],
	[163,45,285],
	[164,45,300],
	[165,45,315],
	[166,45,330],
	[167,45,345],
	[168,60,0],
	[169,60,30],
	[170,60,60],
	[171,60,90],
	[172,60,120],
	[173,60,150],
	[174,60,180],
	[175,60,210],
	[176,60,240],
	[177,60,270],
	[178,60,300],
	[179,60,330],
	[180,75,0],
	[181,75,60],
	[182,75,120],
	[183,75,180],
	[184,75,240],
	[185,75,300],
	[186,90,0],
	], dtype = np.float32)

	# convert all angles to radians
	index_elev_azim[:,1:3] *= (2 * np.pi / 360);
	return index_elev_azim
# }}}

def get_cartesian_samplepoints():
	'''
	Return an array A of shape (187, 3) such that:
	A[i, :] = [x, y, z] of the point at (zero-based) index i in the HRTF database
	'''

	# convert sphere coordinates (r=1, elev, azim) to cartesian (x,y,z) with ||(x,y,z)|| = 1
	# {{{
	# coordinate system orientation:
	# e_x = (1, 0, (3/2) pi)
	# e_y = (1, 0, 0)
	# e_z = (1, pi/2, <arbitrary>)
	# (x points to the right, y points forward, z points up)
	cartesian_points = np.zeros((index_elev_azim.shape[0], 3))

	# column 0 = x coordinate
	cartesian_points[:,0] = -np.sin(index_elev_azim[:,2]) * np.cos(index_elev_azim[:,1]);
	# column 1 = y coordinate
	cartesian_points[:,1] = np.cos(index_elev_azim[:,2]) * np.cos(index_elev_azim[:,1]);
	# column 2 = z coordinate
	cartesian_points[:,2] = np.sin(index_elev_azim[:,1])
	# }}}

	return cartesian_points;

# this is a global variable so that it doesn't have to parse the python
# list to an np.ndarray every time it's used
# sorry for hack
index_elev_azim = get_index_elev_azim()
