import numpy as np 
import matplotlib.colors as mcolors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
	if n == -1:
		n = cmap.N
	new_cmap = mcolors.LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f}'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
	return new_cmap