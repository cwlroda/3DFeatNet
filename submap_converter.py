import os
import sys
import time
import numpy as np
from numpy import matlib as mb
from scipy import spatial
import multiprocessing as mp
from multiprocessing import Pool

FEATURE_DIM = 32

def writeBin(file, data):
    filename = os.path.basename(file).split('.')[0] + '_3dfeatnet.txt'
    outfile = os.path.join('./Data/', filename)
    np.savetxt(outfile, data, delimiter=',')

def kClosest(points, K):
    n = []
    tree = spatial.KDTree(points)

    # k is the number of closest neighbors, p=2 refers to choosing l2 norm (euclidean distance)
    for point in points:
        _, idx = tree.query(x=point, k=K+1, p=2)
        n.append(idx[1:])

    return np.array(n)

def computeNorms(points, numNeighbours=9, viewPoint=[0.0,0.0,0.0], dirLargest=True):
    neighbours = kClosest(points, numNeighbours)

    # find difference in position from neighbouring points
    p = mb.repmat(points[:,:3], numNeighbours, 1) - points[neighbours.flatten('F'),:3]
    p = np.reshape(p, (len(points), numNeighbours, 3))

    # calculate values for covariance matrix
    C = np.zeros((len(points), 6));
    C[:,0] = np.sum(np.multiply(p[:,:,0], p[:,:,0]), 1)
    C[:,1] = np.sum(np.multiply(p[:,:,0], p[:,:,1]), 1)
    C[:,2] = np.sum(np.multiply(p[:,:,0], p[:,:,2]), 1)
    C[:,3] = np.sum(np.multiply(p[:,:,1], p[:,:,1]), 1)
    C[:,4] = np.sum(np.multiply(p[:,:,1], p[:,:,2]), 1)
    C[:,5] = np.sum(np.multiply(p[:,:,2], p[:,:,2]), 1)
    C = np.true_divide(C, numNeighbours)

    # normals and curvature calculation
    normals = np.zeros_like(points)
    # curvature = np.zeros((len(points), 1))

    for i in range(len(points)):
        # form covariance matrix
        Cmat = [[C[i,0], C[i,1], C[i,2]],
                [C[i,1], C[i,3], C[i,4]],
                [C[i,2], C[i,4], C[i,5]]];

        # get eigenvalues and vectors
        [d, v] = np.linalg.eig(Cmat)
        d = np.diag(d)
        k = np.argmin(d)

        # store normals
        normals[i,:] = v[:,k].conj().T

        # store curvature
        # curvature[i] = l / np.sum(d);

    # flipping normals
    # ensure normals point towards viewPoint
    points = points - mb.repmat(viewPoint, len(points), 1)

    # if dirLargest:
    #     idx = np.argmax(np.abs(normals), 1)
    #     print(idx)
    #     idx = np.zeros(len(normals)).conj().T + (idx-1) * len(normals)
    #     print(idx)
    #     dir = np.multiply(normals[idx], points[idx]) > 0

    # else:
    dir = np.sum(np.multiply(normals, points), 1) > 0
    normals[dir,:] = -normals[dir,:]

    return normals

def convert(file):
    points = []

    if file.endswith('bin'):
        with open(file, 'r') as f:
            dt = np.dtype('i8,i4,i8,?,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,i4,i4')
            vals = list(np.fromfile(f, dtype=dt, count=1)[0])

            numFeatures = vals[16]
            numPoints = vals[17]

            for _ in range(numFeatures):
                _ = np.fromfile(f, dtype=np.dtype('f4,f4,f4'), count=1)

                for _ in range(FEATURE_DIM):
                    _ = np.fromfile(f, dtype=np.dtype('f4'), count=1)

            for _ in range(numPoints):
                points.append(list(np.fromfile(f, dtype=np.dtype('f4,f4,f4'), count=1)[0]))
                _ = np.fromfile(f, dtype=np.dtype('f4,f4,f4,u1,u1,u1,i8'), count=1)

        points = np.array(points)
        normals = np.zeros_like(points)
        # normals = computeNorms(points)

        data = np.block([points, normals])
        data = np.float32(data)

        writeBin(file, data)

        print('Succesfully converted {}'.format(file))

if __name__ == '__main__':
    numCores = mp.cpu_count()
    start = time.time()

    with Pool(numCores) as p:
        p.map(convert, [sys.argv[i] for i in range(1, len(sys.argv))])

    end = time.time()
    print('Time taken: {}'.format(end-start))
