import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == '__main__':
	train_points = np.loadtxt('train.txt')
	test_points = np.loadtxt('test.txt')
	
	red_patch = mpatches.Patch(color='red', label='Train')	
	green_patch = mpatches.Patch(color='green', label='Test')
	plt.legend(handles=[red_patch, green_patch], loc=1)
	
	for i in range(train_points.shape[0]):
		plt.plot(train_points[i][0], train_points[i][1], 'r.')
		plt.plot(test_points[i][0], test_points[i][1], 'g.')
		
	plt.show()