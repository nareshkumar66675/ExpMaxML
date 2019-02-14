import numpy as np

def pdf(x,mean,var):    # write Fibonacci series up to n
	eVal = np.exp((-1*np.power((x-mean),2)/(2*var*var)))
	return (1/(var*np.sqrt(2*np.pi)))*eVal