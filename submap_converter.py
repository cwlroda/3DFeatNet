import os
import sys
import time
import numpy as np
from numpy import matlib as mb
from scipy import spatial
import multiprocessing as mp
from multiprocessing import Pool
import csv

FEATURE_DIM = 32

def writeBin(file, data, count):
    parent_dir = file.split('/')[-2]
    # filename = os.path.basename(file).split('.')[0] + '_3dfeatnet.bin'
    outdir = os.path.join('data_output/', parent_dir)

    try:
        os.makedirs(outdir)
    except:
        pass

    # outfile = os.path.join(outdir, filename)
    outfile = os.path.join(outdir, str(count) + '.bin')

    data.tofile(outfile)
    # np.savetxt(outfile, data, delimiter=',')

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
    C = np.zeros((len(points), 6))
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

def createINS(file, vals):
    header = ['timestamp',
              'ins_status',
              'latitude',
              'longitude',
              'altitude',
              'northing',
              'easting',
              'down',
              'utm_zone',
              'velocity_north',
              'velocity_east',
              'velocity_down',
              'roll',
              'pitch',
              'yaw']

    timestamp = vals[0]
    valid = 'INS_SOLUTION_GOOD' if vals[3] else 'INS_BAD_GPS_AGREEMENT'
    longitude = vals[4]
    latitude = vals[5]
    altitude = vals[6]
    roll = vals[7]
    pitch = vals[8]
    yaw = vals[9]

    data = [timestamp,
            valid,
            latitude,
            longitude,
            altitude,
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            roll,
            pitch,
            yaw]

    parent_dir = os.path.join(file.split('/')[-2], 'gps/')
    outdir = os.path.join('data_output/', parent_dir)
    outfile = os.path.join(outdir, 'ins.csv')

    try:
        os.makedirs(outdir)
    except:
        pass

    file_exists = os.path.isfile(outfile)

    with open(outfile, 'a') as out:
        writer = csv.writer(out)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(data)

def createTimestamp(file, timestamp):
    parent_dir = file.split('/')[-2]
    outdir = os.path.join('data_output/', parent_dir)
    outfile = os.path.join(outdir, 'lms_front.timestamps')

    with open(outfile, 'a') as out:
        out.write(str(timestamp))
        out.write(' 1')
        out.write('\n')

def createMetadata(file, vals, count):
    parent_dir = file.split('/')[-2]
    outdir = os.path.join('data_output/', parent_dir)
    outfile = os.path.join(outdir, 'metadata.txt')

    header = 'Idx\tDataset\tStartIdx\tEndIdx\tNumPts\tX\tY\tZ\n'
    data = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                count,
                parent_dir,
                '', '',
                vals[17],
                vals[10],
                vals[11],
                vals[12]
            )

    file_exists = os.path.isfile(outfile)

    with open(outfile, 'a') as out:
        if not file_exists:
            out.write(header)

        out.write(data)
        out.write('\n')

def convert(args):
    file, count = args
    points = []
    vals = []

    if file.endswith('bin'):
        with open(file, 'r') as f:
            dt = np.dtype('i8,i4,i8,?,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,i4,i4')
            vals = list(np.fromfile(f, dtype=dt, count=1)[0])

        # createINS(file, vals)
        # createTimestamp(file, vals[0])

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

        writeBin(file, data, count)
        createMetadata(file, vals, count)

        print('Succesfully converted {}'.format(file))

if __name__ == '__main__':
    numCores = mp.cpu_count()
    start = time.time()

    with Pool(numCores) as p:
        p.map(convert, [(sys.argv[i], i) for i in range(1, len(sys.argv))])

    # for i in range(1, len(sys.argv)):
    #     convert(sys.argv[i])

    end = time.time()
    print('Time taken: {}'.format(end-start))
