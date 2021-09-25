import sys
import numpy as np
from tqdm import trange
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve

'''
For each pixel in the image, for every channel, we perform the following:
    
    Find the partial derivative in the x axis
    Find the partial derivative in the y axis
    Sum their absolute values
It is archieve using the sober filters    
'''
def cal_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3,axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
    ])

    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3,axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img,filter_du)) + np.absolute(convolve(img,filter_dv))
    
    # We sum the energies in the red. green and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

'''
Lets create a 2D array call M to store the minimum energy value seen upto that pixel. 
If you are unfamiliar with dynamic programming, this basically says that M[i,j] will 
contain the smallest energy at that point in the image, considering all the possible 
seams upto that point from the top of the image. So, the minimum energy required to 
traverse from the top of the image to bottom will be present in the last row of M.
 We need to backtrack from this to find the list of pixels
 present in this seam, so we’ll hold onto those values with a 2D array call backtrack.
'''
def minimum_seam(img):
    r,c,_ = img.shape
    energy_map = cal_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M,dtype=np.int)

    for i in range(1,r):
        for j in range(0,c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i,j] = idx + j
                min_energy = M[i-1,idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i,j] += min_energy


    return M, backtrack

'''
Then remove the seam with the lowest energy, and return a new image
'''
def carve_column(img):
    r,c, _ = img.shape

    M,backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r,c),dtype=np.bool)

    # Find the postion of the smallest element in the last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i,j] = False;
        j = backtrack[i,j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))

    return img


'''
Now create a crop_c function which takes as input the image, and a scale factor. 
If the image is of dimensions (300, 600) and we want to reduce it to (150, 600),
You'll pass 0.5 as scale_c. This will continue until the desired number of columns
are removed
'''
def crop_c(img, scale_c):
    r,c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c): # Use range if don't want to use tqdm
        img = carve_column(img)

    return img    

'''
Now create a crop_c function which takes as input the image, and a scale factor. 
If the image is of dimensions (300, 600) and we want to reduce it to (150, 600),
You'll pass 0.5 as scale_c. This will continue until the desired number of rows
are removed
'''
def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img


def main():
    if len(sys.argv) != 5:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]

    img = imread(in_filename)

    if which_axis == 'r':
        out = crop_r(img, scale)
    elif which_axis == 'c':
        out = crop_c(img, scale)
    else:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)
    
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()