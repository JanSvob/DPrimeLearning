'''
Helper functions for visualization.

'''

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

'''
	Taken from nolearn:
	https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/visualize.py

	Plot the weights of a specific layer.
	Only really makes sense with convolutional layers.
	Parameters
	----------
	layer : lasagne.layers.Layer
'''


def plot_conv_weights(layer, name, figsize=(9, 9)):
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    cntr = 0
    for feature_map in range(shape[1]):
        figsize = W[0, feature_map].shape
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

            for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
                if i >= shape[0]:
                    break
                aFilter = W[i, feature_map]
                minVal, maxVal = np.min(aFilter), np.max(aFilter)
                aFilter = (aFilter - minVal) / (maxVal - minVal)
                axes[r, c].imshow(aFilter, cmap='gray')
                # interpolation = 'nearest')

                np.savetxt('filter' + str(i) + '_' + name + '.csv', W[i, feature_map], delimiter=',')

        figs.show()
        # figs.savefig('weights' + str(cntr) + '_' + name + '.png')
        cntr += 1


'''
Taken from lasagne:
https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb

Plot saliency map.
'''


def plot_saliency_map(origImg, saliency, idx):
    origImg = origImg[0][0]

    saliency = saliency[0][0]

    fig = plt.figure(figsize=(16, 16), facecolor='w')
    plt.grid(False)
    plt.subplot(1, 2, 1)
    plt.title('input')
    plt.grid(False)
    plt.imshow(origImg)
    plt.axis('off')
    plt.set_cmap('gray')
    plt.subplot(1, 2, 2)
    plt.title('abs. saliency')
    plt.grid(False)
    fi = plt.imshow(np.abs(saliency))
    plt.axis('off')
    plt.set_cmap('coolwarm')
    fig.savefig('saliency_' + str(idx) + '.png')


from mpl_toolkits.axes_grid1 import ImageGrid


def plot_saliency_maps(maps):
    # fig, axn = plt.subplots(2, 3, sharex=True, sharey=True)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])

    # for i, ax in enumerate(axn.flat):
    #	sns.heatmap(maps[i], ax=ax,
    #		cbar=i == 0,
    #		vmin=0, vmax=1,
    #		cbar_ax=None if i else cbar_ax)

    fig = plt.figure(1, (8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    maxv = np.mean(np.array([np.max(maps[1]), np.max(maps[2]), np.max(maps[3]), np.max(maps[4])]))

    grid[0].imshow(maps[0], cmap='gray', norm=colors.Normalize(vmin=np.min(maps[0]), vmax=np.max(maps[0])))
    grid[0].axis('off')
    grid[0].grid(False)

    grid[1].imshow(maps[1], cmap='coolwarm', norm=colors.Normalize(vmin=0.0, vmax=maxv))
    grid[1].axis('off')
    grid[1].grid(False)

    grid[2].imshow(maps[2], cmap='coolwarm', norm=colors.Normalize(vmin=0.0, vmax=maxv))
    grid[2].axis('off')
    grid[2].grid(False)

    grid[3].imshow(maps[5], cmap='coolwarm', norm=colors.Normalize(vmin=0.0, vmax=maxv))
    grid[3].axis('off')
    grid[3].grid(False)

    grid[4].imshow(maps[3], cmap='coolwarm', norm=colors.Normalize(vmin=0.0, vmax=maxv))
    grid[4].axis('off')
    grid[4].grid(False)

    grid[5].imshow(maps[4], cmap='coolwarm', norm=colors.Normalize(vmin=0.0, vmax=maxv))
    grid[5].axis('off')
    grid[5].grid(False)

    plt.show()

    fig.savefig('saliency_maps.eps', bbox_inches='tight', pad_inches=0)
