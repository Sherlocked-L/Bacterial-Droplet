import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import sys
import os
import csv

# path1 = '/home/LYuLin/software/2Drops/conf-data/characteristic_velocity/v_max-0.0/conf_200.dat'
# path2 = '/home/LYuLin/software/2Drops/conf-data/characteristic_velocity/v_max-1/conf_100.dat'

#path2 = '/home/LYuLin/software/2Drops/conf-data/test2_gammaphi1-Aphi1-Kphi1-wType1-wmax0.1-kwx1-kwy1-rw10-Nx128-Ny128-h1-rate0/conf_200.dat'

N = 64
N0 = 50
N1 = 12500

def calculate_centroid(field):
    sum_x = 0
    sum_y = 0
    sum_z = 0
    total_mass = 0

    size = len(field)

    for i in range(size):
        for j in range(size):
            for k in range(size):
                phi = field[i][j][k]
                mass = phi  
                sum_x += phi * i
                sum_y += phi * j
                sum_z += phi * k
                total_mass += mass

    centroid_x = sum_x / total_mass
    centroid_y = sum_y / total_mass
    centroid_z = sum_z / total_mass

    return int(centroid_x), int(centroid_y), int(centroid_z)

def calculate_centroid_2d(field):
    sum_x = 0
    sum_y = 0
    total_mass = 0

    size = len(field)

    for i in range(size):
        for j in range(size):
            phi = field[i][j]
            mass = phi  
            sum_x += phi * i
            sum_y += phi * j
            total_mass += mass

    centroid_x = sum_x / total_mass
    centroid_y = sum_y / total_mass

    return centroid_x, centroid_y



def read_data(path):
    data = pd.read_csv(path,sep=" ", names = ["1", "2", "3", "4","5","6","7"])

    phi = data["7"].to_numpy().reshape(N, N, N)
    vx = data["1"].to_numpy().reshape(N, N, N)
    vy = data["2"].to_numpy().reshape(N, N, N)

    idx1, idx2, idx3 = calculate_centroid(phi)

    return phi[:,:,idx3], vx[:,:,idx3], vy[:,:,idx3] , idx1, idx2

def extract_contour_polar(phi, center):

    contour_set = plt.contour(phi, levels=[0.5], colors='black')
    plt.close()
    contour_paths = contour_set.collections[0].get_paths()  # 获取等势线
    
    points = []
    
    for path in contour_paths:
        vertices = path.vertices
        cx, cy = np.mean(vertices, axis=0)
        points.extend(vertices.tolist())
    

    polar_r = np.zeros(len(points))
    polar_theta = np.zeros(len(points))

    for i in range(len(points)):
        point = points[i]
        dx = point[0] - cx  # 相对于 [32, 32] 的横坐标差值
        dy = point[1] - cy  # 相对于 [32, 32] 的纵坐标差值
        r = np.sqrt(dx**2 + dy**2)  # 极径
        theta = np.arctan2(dy, dx)  # 极角

        polar_r[i] = r
        polar_theta[i] = theta
    
    for i in range(len(points)):
        if polar_theta[i] < -np.pi:
            polar_theta[i] += 2*np.pi
        elif polar_theta[i] > np.pi:
            polar_theta[i] -= 2*np.pi
    
    R = []
    X = []
    Y = []
    for theta in range(-180, 180, 3):
        tempDelta = polar_theta - theta/180*np.pi

        targetTheta = np.argmin(np.abs(tempDelta))
        targetR = polar_r[targetTheta]

        R.append(targetR)
        X.append(points[targetTheta][0])
        Y.append(points[targetTheta][1])

        
    return np.array(R), np.array(X), np.array(Y)


# center = [32, 32]


def SaveR(Path, N0, N1):
    R = []
    for i in range(N0, N1+1):
        path = Path + '/conf_'+str(i)+'.dat'
        phi, vx, vy, _, _ = read_data(path)
        c_x, c_y = calculate_centroid_2d(phi)
        r, _, _ = extract_contour_polar(phi, [c_x,c_y])
        R.append(r)
    
    save_path = Path + '/a_R.txt'
    np.savetxt(save_path, R)

def SaveV(Path, N0, N1):

    os.makedirs(Path+"/Vv")

    for i in range(N0, N1+1):
        path = Path + '/conf_'+str(i)+'.dat'
        _, vx, vy, _, _ = read_data(path)
        
        # np.save(Path + '/Vv/Vx_'+str(i)+'.npy', vx)
        # np.save(Path + '/Vv/Vy_'+str(i)+'.npy', vy)

        with open(Path + '/Vv/Vx_'+str(i)+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(vx)

        with open(Path + '/Vv/Vy_'+str(i)+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(vy)
        
def plotProfile(Path,N):

    path = Path + '/conf_'+str(N)+'.dat'
    phi, _, _, _, _ = read_data(path)
    
    plt.plot(phi[:,32])
    plt.savefig("profile.png")

path = sys.argv[1]
# plotProfile(path,50)
SaveR(path, N0, N1)
# SaveV(path, N0, N1)


'''
phi1 = read_data(path1)
phi2 = read_data(path2)

R1, X1, Y1 = extract_contour_polar(phi1, center)

R2, X2, Y2 = extract_contour_polar(phi2, center)

mse = np.mean(np.square(R1 - R2))
#print(R1)

print(mse)
'''
# def shape_mse(path1, path2):
#     phi1 = read_data(path1)
#     phi2 = read_data(path2)

#     R1, _, _ = extract_contour_polar(phi1, center)

#     R2, _, _ = extract_contour_polar(phi2, center)

#     mse = np.mean(np.square(R1 - R2))
#     return mse

'''
MSE = []
V = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]

for v in V:
    path1 = '/home/LYuLin/software/2Drops/conf-data/characteristic_velocity/max_v-0.0/conf_100.dat'
    path2 = '/home/LYuLin/software/2Drops/conf-data/characteristic_velocity/max_v-'+str(v)+'/conf_100.dat'
    mse = shape_mse(path1, path2)
    MSE.append(mse)

plt.plot(V, MSE)
plt.xlabel('v_max')
plt.ylabel('MSE')
plt.xlim(0, 2)
plt.savefig('MSE.png')
'''

# MSE = []
# x = []
# N = 1000
# # it = sys.argv[1]
# it = '0.01_T-1M_every-10'
# #path0 = '/home/LYuLin/software/2Drops/conf-data/characteristic_velocity/max_v-0.0/conf_100.dat'
# #path = '/home/LYuLin/software/2Drops/conf-data/characteristic_velocity/max_v-'+it
# #path = '/home/LYuLin/software/2Drops/conf-data/max_v-'+it
# path0 = '/home/LYuLin/software/2Drops/conf-data/max_v-0.0/conf_10.dat'
# path = '/home/LYuLin/software/2Drops/conf-data/temp_velocity'
# for i in range(0,N+1):
#     path1 = path + '/conf_'+str(i)+'.dat'
#     mse = shape_mse(path0, path1)
#     MSE.append(mse)
#     x.append(i)

# plt.plot(x,MSE)
# plt.xlabel('time')
# plt.ylabel('MSE')
# plt.savefig('MSE_v-'+it+'_last-20k.png')

# #arr = array.array('i', MSE)
# #file_path = "./MSE_max_v-0.01_T-1M_every-10.bin"
# #with open(file_path, "wb") as file:
# #    arr.tofile(file)

# '''
# # read
# file_path = "path/to/your/file.bin"
# arr = array.array('i')

# with open(file_path, "rb") as file:
#     arr.fromfile(file, len(arr))

# data = list(arr)
# '''
