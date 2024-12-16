# -*- coding: gbk -*
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from tqdm import tqdm
import matplotlib.animation as animation
import pandas as pd
import time
import os
import math
import platform
# if platform.system() == "Linux":
import cv2
import shutil
from scipy import ndimage
import sys
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from op import d1xO4, d1yO4

def delete_extra_zero(n):
    if isinstance(n, int):
        return n
    if isinstance(n, float):
        n = str(n).rstrip('0')  
        n = int(n.rstrip('.')) if n.endswith('.') else float(n) 
        return n

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

parameter_columns_name = ["TimeScheme", "RKStage", "ExplImpi", "AdapTime", "ExpoData", "InitCond",
                        "isSimplified", "checkUnStable", "Nx", "Ny", "Nz", "Nb", "h", "dt0", "T0", "Ts", 
                        "dte", "dtVarC", "dtMax", "mu", "cA", "alpha","beta", "eta","La", "kappa", "lambda0", 
                        "aphi", "kphi", "gamma", "A", "init0", "ReadSuccess"]

output_columns_name = ["vx", "vy", "vz","px","py","pz","phi"]
# output_columns_name = ["vx", "vy", "vz", "4", "5","6","phi","8","9"]

run_cuda = 0                      # draw animation, 0 as dir name 
cut_scale = 2                     # decide the thickness of cut. Bigger Denser
cut_rate = 2                      # quiver interval 
draw_v = 1                        #  v imshow and quiver, 1 means yes
draw_phi = 1                      # phi imshow , 1 means yes
draw_phi_drop_size = 0            # phi_drop by time, 1 means yes
draw_vx = 0                       # vx imshow , 1 means yes
draw_vy = 0                       # vy imshow , 1 means yes
draw_vz = 0                       # vz imshow , 1 means yes
draw_dvy = 0                      # dvy imshow , vy - vy(addtConvt), 1 means yes
draw_vortex = 0                   # vortex imshow��1 means yes
draw_dir = 0                      # direction imshow��1 means yes
draw_all = 0                      # draw all files under conf-data-f-field, 1 means yes
draw_term = 0                     # term imshow, include term1��2��3 �� 1 means yes
draw_dtvortex = 0                 # dtvortex = vortex(t) - vortex(t-1)
draw_p = 1
draw_v_star = 0
draw_dtphi = 0

draw_convection = 0
draw_mu = 0                       # mu imshow
draw_J = 0                        # J imshow and quiver
draw_flux = 0                     # flux imshow and flux_sum over time
draw_phi_v = 0                    # phi and v imshow

draw_wk = 0
draw_dwk = 0
draw_dvortex = 0 # dvortex = vortex(t) - vortex(0)

if_save_last = 1 # save last frame
if_save_gif = 1 # save gif

pre_filename = './'  #########################################################
args = sys.argv
if run_cuda:
    input1 = pd.read_csv("input.dat", names=['0']).T.to_numpy()
    input1 = pd.DataFrame(input1, columns=parameter_columns_name)
    T0 = int(input1.loc[0, "T0"])
    Ts = int(input1.loc[0, "Ts"])
    t = input1.loc[0, "Ts"] - input1.loc[0, "T0"]
    dt = input1.loc[0, "dt0"]
    dte = input1.loc[0, "dte"]
    h = int(input1.loc[0, "h"])
    Lx = int(input1.loc[0, "Nx"])
    Ly = int(input1.loc[0, "Ny"])
    kwx=int(input1.loc[0, "kwx"]) 
    kwy=int(input1.loc[0, "kwy"]) 
    ExpoData = input1.loc[0, "ExpoData"]
    InitCond = int(input1.loc[0, "InitCond"])
    alpha = delete_extra_zero(input1.loc[0, "alpha"])
    gammaphi=delete_extra_zero(input1.loc[0, "gammaphi"])
    Aphi = delete_extra_zero(input1.loc[0, "Aphi"])
    Kphi = delete_extra_zero(input1.loc[0, "Kphi"])
    phi0 = delete_extra_zero(input1.loc[0, "phi0"])
    rho1=delete_extra_zero(input1.loc[0, "rho1"])
    rho2=delete_extra_zero(input1.loc[0, "rho2"])
    tauv=delete_extra_zero(input1.loc[0, "tauv"]) 
    wType=delete_extra_zero(input1.loc[0, "wType"]) 
    wmax=delete_extra_zero(input1.loc[0, "wmax"]) 
    rw=delete_extra_zero(input1.loc[0, "rw"])
    filename = "gammaphi"+str(gammaphi)+"-Aphi"+str(Aphi)+"-Kphi"+str(Kphi)+"-phi0"+str(phi0)+"-wType"+str(wType)+"-wmax"+str(wmax)+"-kwx"+str(kwx)+"-kwy"+str(kwy)+"-rw"+str(rw)+"-Nx"+str(Lx)+"-Ny"+str(Ly)+"-h"+str(h)
else:
    if len(args) == 1:
        filename = ['6'] 
        #filename = ['gammaphi1-Aphi1-Kphi1-phi00.01-wType1-wmax0-kwx1-kwy1-rw10-Nx64-Ny64-h2']
    elif draw_all == 1:
        filename = os.listdir(pre_filename)
        print(filename)
        print("numbel of file {}".format(len(filename)))
    else:
        filename = args[1:]


class Draw:
    def __init__(self, filename, input):
        self.ExpoData = input.loc[0, "ExpoData"]
        #  set parameter
        self.T0 = int(input.loc[0, "T0"])
        self.Ts = input.loc[0, "Ts"]
        self.t = self.Ts - self.T0
        self.dte = input.loc[0, "dte"]
        self.filename = filename
        self.dx = input.loc[0, "h"]
        self.dy = input.loc[0, "h"]
        self.dz = input.loc[0, "h"]
        # self.gammaphi=delete_extra_zero(input.loc[0, "gammaphi"])
        # self.alpha = delete_extra_zero(input.loc[0, "alpha"])
        # self.beta = delete_extra_zero(input.loc[0, "beta"])
        # self.gamma0 = delete_extra_zero(input.loc[0, "gamma0"])
        # self.gamma2 = delete_extra_zero(input.loc[0, "gamma2"])
        # self.s = delete_extra_zero(input.loc[0, "s"])
        # self.d = delete_extra_zero(input.loc[0, "d"])
        # self.Kphi = delete_extra_zero(input.loc[0, "Kphi"])
        # self.phi0 = delete_extra_zero(input.loc[0, "phi0"])
        self.Lx = int(input.loc[0, "Nx"])
        self.Ly = int(input.loc[0, "Ny"])
        self.Lz = int(input.loc[0, "Nz"])
        self.h = input.loc[0, "h"]
        # self.kwx=int(input.loc[0, "kwx"]) 
        # self.kwy=int(input.loc[0, "kwy"]) 
        self.Init = int(input.loc[0, "InitCond"])
        # self.rho1=delete_extra_zero(input.loc[0, "rho1"])
        # self.rho2=delete_extra_zero(input.loc[0, "rho2"])
        # self.tauv=delete_extra_zero(input.loc[0, "tauv"]) 
        # self.wType=delete_extra_zero(input.loc[0, "wType"]) 
        # self.wmax=delete_extra_zero(input.loc[0, "wmax"]) 
        # self.rw=delete_extra_zero(input.loc[0, "rw"]) 
        self.flux_x_list, self.flux_y_list, self.flux_l_list= [self.Lx//4, 3*self.Lx//4], [self.Ly//4, 3*self.Ly//4], [14, 12]


        if self.ExpoData == 1:
            self.phi = np.zeros((self.Ly, self.Lx))
            self.vortex = np.zeros((self.Ly, self.Lx))
            self.vx = np.zeros((self.Ly, self.Lx))
            self.vy = np.zeros((self.Ly, self.Lx))
            self.mu = np.zeros((self.Ly, self.Lx))
        else:
            1  ##���������м�������д

    def getData(self, i):
        if isinstance(i, str):
            i = int((int(i) - self.T0) / self.dte)
        if self.ExpoData == 1:
            data = pd.read_csv("{}/conf_{}.dat".format(self.filename, i), sep=" ",
                                names=output_columns_name)
            self.phi = data["phi"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            # self.vortex = data["vortex"].to_numpy().reshape((self.Ly, self.Lx))
            # self.p = data["p"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            self.vx = data["vx"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            self.vy = data["vy"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            self.vz = data["vz"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            self.px = data["px"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            self.py = data["py"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            self.pz = data["pz"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            # self.vx_star = data["vx_star"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            # self.vy_star = data["vy_star"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            # self.dtphi = data["dtphi"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))
            # self.div_v_star = data["div_v_star"].to_numpy().reshape((self.Ly, self.Lx, self.Lz))


            idx1, idx2, idx3 = calculate_centroid(self.phi)

            # self.p = self.p[:,:,idx3]
            self.vx = self.vx[:,:,idx3]
            self.vy = self.vy[:,:,idx3]
            self.vz = self.vz[:,:,idx3]
            self.px = self.px[:,:,idx3]
            self.py = self.py[:,:,idx3]
            self.pz = self.pz[:,:,idx3]
            # self.vx_star = self.vx_star[:,:,idx3]
            # self.vy_star = self.vy_star[:,:,idx3]
            self.phi = self.phi[:,:,idx3]
            # self.dtphi = self.dtphi[:,:,idx3]
            # self.vz_star = self.vz_star[:,:,int(self.Lz/2)]
            self.vortex = d1xO4(self.vy, self.h) - d1yO4(self.vx, self.h)
            self.Pvortex = d1xO4(self.py, self.h) - d1yO4(self.px, self.h)
            # self.mu = data["stream"].to_numpy().reshape((self.Ly, self.Lx))
            # self.Jx = -d1xO4(self.mu, self.h)
            # self.Jy = -d1yO4(self.mu, self.h)
        if self.ExpoData == 2:
            data = pd.read_csv("{}/conf_{}.dat".format(self.filename, i), sep=" ",
                    names=output_columns_name)
            self.phi = data["phi"].to_numpy().reshape((self.Ly, self.Lx))
            self.vortex = data["vortex"].to_numpy().reshape((self.Ly, self.Lx))
            self.vx = data["vx"].to_numpy().reshape((self.Ly, self.Lx))
            self.vy = data["vy"].to_numpy().reshape((self.Ly, self.Lx))
            self.term1_t = data["term1"].to_numpy().reshape((self.Ly, self.Lx))
            self.term2_t = data["term2"].to_numpy().reshape((self.Ly, self.Lx))
            self.term3_t = data["term3"].to_numpy().reshape((self.Ly, self.Lx))
        else:
            1  ##���������м�������д

    def cutMatrix(self, matrix, cut_row, cut_col):
        matrix_cut = np.zeros((int(matrix.shape[0] / cut_row), int(matrix.shape[1] / cut_col)))
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                if r % cut_row == 0 and c % cut_col == 0:
                    matrix_cut[int(r / cut_row)][int(c / cut_col)] = matrix[r][c]
        return matrix_cut

    def quarterMatrix(self, matrix):
        half_x = int(matrix.shape[1]/2)
        half_y = int(matrix.shape[0]/2)
        return matrix[-half_y-1:, -half_x-1:]
    
    def find_couple(self, N, k):
        x = np.arange(-N, N + 1)
        y = np.arange(-N, N + 1)
        X, Y = np.meshgrid(x, y)
        tmp = np.transpose(np.where(X + Y == k))
        tmp = tmp - N
        return tmp

    def K_2(self, kx, ky):
        lhs = (ky[:, np.newaxis] ** 2 + kx[np.newaxis, :] ** 2)
        lhs[lhs == 0] = 1
        return lhs

    def convert_png_to_video(self, png_folder, output_video, fps=10):
        def extract_t_end(filename):
            t_end_str = filename.split("_t_end")[-1].split(".png")[0]
            return float(t_end_str)
        
        png_files = [file for file in os.listdir(png_folder) if file.endswith(".png")]

        image_files = sorted(png_files, key=extract_t_end)

        first_image = cv2.imread(os.path.join(png_folder, image_files[0]))
        height, width, _ = first_image.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for image_file in image_files:
            image_path = os.path.join(png_folder, image_file)
            image = cv2.imread(image_path)
            video_writer.write(image)

        video_writer.release()
    
    def drawGeneral(self, t_string, val, valName, interpla='gaussian', c='jet', addGrid = False, addFluxBoundry = False):
        xy_pro = math.ceil(self.Lx / self.Ly)
        base_size = 5 * math.ceil((self.Ly / 64))
        fig, ax = plt.subplots(figsize=(xy_pro * base_size, base_size))
        # val = abs(self.wk - self.wk0)
        # hmax, hmin = np.max(val), np.min(val)
        im = ax.imshow(val, origin='lower', interpolation=interpla, cmap=c)
        clim = max(max(val.flatten()), -min(val.flatten()))
        im.set_clim(-clim, clim)
        # im.set_clim(-200, 200)
        # xticks = np.arange(0, 70, 2)
        # ax.set_xticks(xticks)
        counter_lines = ax.contour(self.phi, colors='red', linewidths=0.5, levels=[0.5])

        # mass center
        idx1, idx2 = calculate_centroid_2d(self.phi)

        paths = counter_lines.collections[0].get_paths()

        distances = []
        for path in paths:
            vertices = path.vertices
            path_distances = [np.linalg.norm(point - [idx1, idx2]) for point in vertices]
            cx, cy = np.mean(vertices, axis=0)
            distances.extend(path_distances)

        average_radius1 = np.mean(distances)

        circle2 = Circle([cx, cy], average_radius1, edgecolor='green', facecolor='none', linewidth=0.5)
        ax.add_patch(circle2)

        title = ax.text(0.5, 1.05,
                         "{}_".format(valName)  + "T=" + t_string,
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes)
        cbar = fig.colorbar(im, ax=ax)
        # colorbar set clim
        # cbar.set_clim(-40, 40)
        if addGrid:
            ax.grid(addGrid, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(np.arange(0, val.shape[1], 5))
            plt.yticks(np.arange(0, val.shape[0], 5))
        if addFluxBoundry:
            for i in range(len(self.flux_x_list)):
                rec = Rectangle((self.flux_x_list[i] - self.flux_l_list[i], self.flux_y_list[i] - self.flux_l_list[i])
                                , 2*self.flux_l_list[i], 2*self.flux_l_list[i], linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rec)

        if not os.path.exists('{}/{}'.format(self.filename, valName)):
            os.makedirs('{}/{}'.format(self.filename, valName))
        fig.savefig('{}/{}/{}_t_end{}.png'.format(self.filename, valName, valName, t_string))
        # fig.savefig('{}/{}/{}_{}_t_end{}.png'.format(self.filename, valName, valName, self.midname, t_string))
        plt.close(fig)
    
    def drawGeneralQuiver(self, t_string, valx, valy, valName, cut_row, cut_col, interpla='gaussian', c='jet', addGrid = False, normalization = False, imshow = None):
        xy_pro = math.ceil(self.Lx / self.Ly)
        if self.Ly <=64:
            base_size = 10 * math.ceil((self.Ly / 64))
        else:
            base_size = 5 * math.ceil((self.Ly / 64))
        row = np.arange(0, self.Ly, cut_row)
        clo = np.arange(0, self.Lx, cut_col)
        X, Y = np.meshgrid(clo, row)
        fig, ax = plt.subplots(figsize=(xy_pro * base_size, base_size))
        if imshow is None:
            val = np.sqrt(valx ** 2 + valy ** 2)
            # val /= np.max(val)
        else:
            val = imshow
        valx = valx[::cut_row, ::cut_col]
        valy = valy[::cut_row, ::cut_col] 
        if normalization:
            val_cut = val[::cut_row, ::cut_col]
            valx = valx / val_cut
            valy = valy / val_cut
            qu = ax.quiver(X, Y, valx, valy,units='xy', scale_units='xy', animated=True, pivot='middle')
        else:
            # max_v = np.max(np.sqrt(valx ** 2 + valy ** 2))
            qu = ax.quiver(X, Y, valx, valy, color=(0.3, 0.3, 0.3), scale=2, animated=True, pivot='middle')
        hmax, hmin = np.max(val), np.min(val)
        im = ax.imshow(val, origin='lower', interpolation=interpla, cmap=c)
        clim = max(max(val.flatten()), -min(val.flatten()))
        im.set_clim(-clim, clim)

        counter_lines = ax.contour(self.phi, colors='blue', linewidths=2, levels=[0.5])

        # idx1, idx2 = calculate_centroid_2d(self.phi)

        # paths = counter_lines.collections[0].get_paths()

        # distances = []
        # for path in paths:
        #     vertices = path.vertices
        #     path_distances = [np.linalg.norm(point - [idx1, idx2]) for point in vertices]
        #     cx, cy = np.mean(vertices, axis=0)
        #     distances.extend(path_distances)

        # average_radius1 = np.mean(distances)

        # circle2 = Circle([cx, cy], average_radius1, edgecolor='red', facecolor='none', linewidth=2)
        # ax.add_patch(circle2)

        #################

        # points = []
    
        # for path in contour_paths:
        #     vertices = path.vertices
        #     cx, cy = np.mean(vertices, axis=0)
        #     points.extend(vertices.tolist())
    

        # polar_r = np.zeros(len(points))
        # polar_theta = np.zeros(len(points))

        # for i in range(len(points)):
        #     point = points[i]
        #     dx = point[0] - cx  
        #     dy = point[1] - cy 
        #     r = np.sqrt(dx**2 + dy**2)  
        #     theta = np.arctan2(dy, dx)  

        #     polar_r[i] = r
        #     polar_theta[i] = theta
    
        # for i in range(len(points)):
        #     if polar_theta[i] < -np.pi:
        #         polar_theta[i] += 2*np.pi
        #     elif polar_theta[i] > np.pi:
        #         polar_theta[i] -= 2*np.pi
    
        # # R = []
        # X = []
        # Y = []
        # for theta in range(-180, 180, 3):
        #     tempDelta = polar_theta - theta/180*np.pi

        #     targetTheta = np.argmin(np.abs(tempDelta))
        #     targetR = polar_r[targetTheta]

        #     R.append(targetR)
            # X.append(points[targetTheta][0])
            # Y.append(points[targetTheta][1])
        ##################


    #    qu = ax.quiver(X, Y, valx, valy,units='xy', scale_units='xy', animated=True, pivot='middle')
        title = ax.text(0.5, 1.05,
                        "{}_".format(valName)  + "T=" + t_string,
                        # size=plt.rcParams["axes.titlesize"],
                        fontsize = 32,
                        ha="center", transform=ax.transAxes)
        # fig.colorbar(im, ax=ax)
        # colorbar set clim
        cbar = fig.colorbar(im, ax=ax)

        # cbar.set_clim(-20, 20)

        if addGrid:
            ax.grid(addGrid, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(np.arange(0, val.shape[1], 5))
            plt.yticks(np.arange(0, val.shape[0], 5))
        if not os.path.exists('{}/{}'.format(self.filename, valName)):
            os.makedirs('{}/{}'.format(self.filename, valName))
        fig.savefig('{}/{}/{}_t_end{}.png'.format(self.filename, valName, valName, t_string))
        # fig.savefig('{}/{}/{}_{}_t_end{}.png'.format(self.filename, valName, valName, self.midname, t_string))
        plt.close(fig)

    def savePng(self):
        xy_pro = math.ceil(self.Lx / self.Ly)
        base_size = 5 * math.ceil((self.Ly / 64))

        if cut_scale == 1:
            cut_row = cut_rate  # math.ceil((self.Ly / 127))
            cut_col = cut_rate  # math.ceil((self.Lx / 127))
        else:
            cut_row = math.ceil((self.Ly / 32))
            cut_col = math.ceil((self.Lx / 32))
        row = np.arange(0, self.Ly, cut_row)
        clo = np.arange(0, self.Lx, cut_col)
        t_interval = self.dte

        run_times = int(self.t/t_interval) + 1
        
        for i in tqdm(range(run_times)):
            self.getData(i + int(self.T0/t_interval))
            # if t_interval * i + self.T0 >= self.tAddConvt:
            #     is_on = "on"
            X, Y = np.meshgrid(clo, row)
            ######################################################################
            v = np.sqrt(self.vx ** 2 + self.vy ** 2)
            # v_star = np.sqrt(self.vx_star ** 2 + self.vy_star ** 2)
            # vx_cut = self.cutMatrix(self.vx, cut_row, cut_col)
            # vy_cut = self.cutMatrix(self.vy, cut_row, cut_col)
            # v_cut = np.sqrt(vx_cut ** 2 + vy_cut ** 2)
            # rate = ((v_cut - np.min(v_cut)) / (np.max(v_cut) - np.min(v_cut)))/v_cut
            # vx_cut = vx_cut * rate
            # vy_cut = vy_cut * rate
            quiver_scale = 5
            arrow_control=5
            hmax , hmin = np.max(v) , np.min(v)
            #v = (v - hmin) / (hmax - hmin)  # ���Թ�һ�� 
            t_string = "{:0.3f}".format(self.T0+i*t_interval)
            ######################################################################
            if draw_v:    
                # fig_v, ax_v = plt.subplots(figsize=(xy_pro*base_size, base_size))   
                # im = ax_v.imshow(v, origin='lower', interpolation='gaussian', animated=True, cmap='summer') #���ٶȷֲ�
                # qu = ax_v.quiver(X, Y, vx_cut, vy_cut, units='xy', scale_units='xy', animated=True, pivot='middle') ###########10
                # title = ax_v.text(0.5, 1.05,
                #                 "vmin=" + str(hmin) + ", vmax=" + str(hmax) + "\n" + "t=" + str(t_interval * i + self.T0),
                #                 size=plt.rcParams["axes.titlesize"],
                #                 ha="center", transform=ax_v.transAxes)
                # fig_v.colorbar(im, ax=ax_v)
                # if not os.path.exists('{}/{}'.format(self.filename, 'v')):
                #     os.makedirs('{}/{}'.format(self.filename, 'v'))
                # fig_v.savefig('{}/v/v_t_end{}.png'.format(self.filename, t_string))
                # # fig_v.savefig('{}/v/v_{}_t_end{}.png'.format(self.filename, self.midname, t_string))
                # plt.close(fig_v)
                self.drawGeneralQuiver(t_string, self.vx, self.vy, 'v', 2, 2, normalization=False, c='jet', imshow=self.vortex)
            ######################################################################
            
            if draw_v_star:
                self.drawGeneralQuiver(t_string, self.vx_star, self.vy_star, 'v_star', 4, 4, normalization=True, c='summer')
            
            if draw_phi:
                self.drawGeneral(t_string, self.phi, 'phi', c='seismic', addFluxBoundry=False)
            
            if draw_p:
                self.drawGeneralQuiver(t_string, self.px, self.py, 'p', 2, 2, normalization=False, c='jet', imshow=self.Pvortex)
            ######################################################################

            if draw_dtphi:
                self.drawGeneral(t_string, self.dtphi, 'dtphi', c='seismic', addFluxBoundry=True)

            if draw_vx:
                self.drawGeneral(t_string, self.vx, 'vx')

            ######################################################################
            if draw_vy:
                self.drawGeneral(t_string, self.vy, 'vy')
                
            if draw_vz:
                self.drawGeneral(t_string, self.vz, 'vz')

            ######################################################################
            if draw_dvy:
                if t_interval * i + self.T0 < self.tAddConvt:
                    vy_init = self.vy
                dvy_abs = np.abs(self.vy-vy_init)
                self.drawGeneral(t_string, dvy_abs, 'dvy')


            ######################################################################
            if draw_vortex:
                self.drawGeneral(t_string, self.vortex, 'vortex')
            
            ######################################################################
            if draw_dir:
                fig_w, ax_w = plt.subplots(figsize=(xy_pro * base_size, base_size))
                theta = np.arctan2(self.vy, self.vx)
                theta = np.where(theta >= 0, theta, 2*np.pi-np.abs(theta))
                im = ax_w.imshow(theta, origin='lower', interpolation='gaussian', animated=True, vmax=2*np.pi , vmin=0,
                                cmap='hsv')  
                title = ax_w.text(0.5, 1.05,
                        "t=" + str(t_interval * i + self.T0),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax_w.transAxes)
                cbar = fig_w.colorbar(im, ax=ax_w, ticks=[0, np.pi, 2*np.pi])
                cbar.set_ticklabels(['0','��','2��'])
                cbar.ax.tick_params(labelsize=14)  
                fig_w.savefig('{}/dir/dir_t_end{}.png'.format(self.filename, t_string))
                # fig_w.savefig('{}/dir/dir_{}_t_end{}.png'.format(self.filename, self.midname, t_string))


            ######################################################################
            if draw_dtvortex and self.ExpoData == 2:  
                dtvortex = self.term1_t + self.term2_t + self.term3_t
                self.drawGeneral(t_string, dtvortex, 'dtvortex')
            
                        ######################################################################
            if draw_term and self.ExpoData == 2:
                self.drawGeneral(t_string, self.term1_t, 'term1')
                self.drawGeneral(t_string, self.term2_t, 'term2')
                self.drawGeneral(t_string, self.term3_t, 'term3')
            
            central = [(int)(self.Ly / 2), (int)(self.Lx / 2)]
            Nky, Nkx = 5, 64
            self.wk = np.fft.fftshift(np.fft.ifft2(self.vortex))[central[0] - Nky:central[0] + Nky + 1,
                central[1] - Nkx:central[1] + Nkx + 1]
            if draw_wk:
                self.drawGeneral(t_string, self.quarterMatrix(abs(self.wk)), 'wk', interpla='none', addGrid=True)

            if draw_dwk:
                if i == 0:
                    self.wk0 = self.wk
                self.drawGeneral(t_string, self.quarterMatrix(abs(self.wk - self.wk0)), 'dwk', interpla='none', addGrid=True)

            if draw_dvortex:
                if i == 0:
                    self.vortex0 = self.vortex
                self.drawGeneral(t_string, self.vortex - self.vortex0, 'dvortex')
            
            if draw_mu:
                self.drawGeneral(t_string, self.mu, 'mu', interpla='none' )

            if draw_J:
                #self.drawGeneral(t_string, J_stream, 'J')
                self.drawGeneralQuiver(t_string, self.Jx, self.Jy, 'J', cut_row, cut_col, normalization=True, c='Greens')

            if draw_flux:
                # flux_normal = self.get_flux_area(np.zeros((self.Lx, self.Lx, 2)), self.flux_x_list[0], self.flux_y_list[0], self.flux_l_list[0]) 
                # flux_normal += self.get_flux_area(np.zeros((self.Lx, self.Lx, 2)), self.flux_x_list[1], self.flux_y_list[1], self.flux_l_list[1]) 
                flux_normal = self.get_flux_area(np.zeros((self.Lx, self.Lx, 2)))
                flux_J = self.Jx * flux_normal[:, :, 0] + self.Jy * flux_normal[:, :, 1]
                self.drawGeneral(t_string, flux_J, 'flux', interpla='none' )
                #self.drawGeneral(t_string, flux_normal[:, :, 0] ** 2 + flux_normal[:, :, 1] ** 2, 'flux', interpla='none' )
            
            if draw_phi_v:
                self.drawGeneralQuiver(t_string, self.vx, self.vy, 'phi', 4, 4, normalization=False, c='seismic', imshow=self.phi)

            if draw_convection:
                conv = - (self.vx * d1xO4(self.phi, self.h) + self.vy * d1yO4(self.phi, self.h))
                self.drawGeneral(t_string, conv, 'conv', interpla='none')
    def saveMp4(self, valName):
        png_folder = '{}/{}'.format(self.filename, valName)
        output_video = '{}/a_{}.mp4'.format(self.filename, valName)
        self.convert_png_to_video(png_folder, output_video)
        # shutil.rmtree(png_folder)  delete folder
    
    def drawDynamic(self):
        self.savePng()

        kx, ky = np.array([6]), np.array([1])
        t_string = "{:0.0f}".format(10130)
        # self.drawTerm3Item(kx[0], ky[0], t_string)
        # self.drawVk(kx[0], ky[0], t_string)
        
        if platform.system() == "Linux":
            if draw_v:  
                self.saveMp4('v')

            if draw_phi:  
                self.saveMp4('phi')

            if draw_phi_drop_size:
                run_times = self.get_conf_number()
                for i in range(run_times):
                    self.getData(i)
                    self.draw_phi_drop_size_png(i, run_times, self.dte, 0.5, 0)
            
            if draw_vx:  
                self.saveMp4('vx')

            if draw_vy:  
                self.saveMp4('vy')

            if draw_dvy:  
                self.saveMp4('dvy')

            if draw_vortex:  
                self.saveMp4('vortex')
                
            if draw_dir:  
                self.saveMp4('dir')    
            
            if draw_dtvortex:
                self.saveMp4('dtvortex')

            if draw_term:
                self.saveMp4('term1')
                self.saveMp4('term2')
                self.saveMp4('term3')

            if draw_wk:
                self.saveMp4('wk')
            
            if draw_dwk:
                self.saveMp4('dwk')
            
            if draw_dvortex:
                self.saveMp4('dvortex')
            
            if draw_mu:
                self.saveMp4('J')

            if draw_flux:
                run_times = self.get_conf_number()
                
                for i in range(run_times):
                    self.getData(i)
                    self.draw_flux_sum_png(i, run_times, self.dte, self.flux_x_list, self.flux_y_list, self.flux_l_list, 2)
            
            if draw_phi_v:
                self.saveMp4('phi')

            if draw_convection:
                self.saveMp4('conv')

            if draw_p:
                self.saveMp4('p')
            


    def draw_phi_drop_size_png(self, i, run_times, t_interval, threshold=0.5, boundary_width=0):
        # binary_image = self.phi > threshold
        # binary_image_with_boundary = np.pad(binary_image, boundary_width, mode='wrap')  # ���������Ա߽�
        # labeled_array, num_features = ndimage.label(binary_image_with_boundary) 
        # #sizes = ndimage.sum(binary_image, labeled_array, range(num_features + 1))
        # sizes = ndimage.sum(binary_image_with_boundary, labeled_array, range(num_features + 1))
        left_bottom_count = np.sum(self.phi[:self.Ly//2, :] > 0.5)
        right_top_count = np.sum(self.phi[self.Ly//2:, :] > 0.5)
        if(i == 0):
            self.phi_size_array = np.zeros((2, run_times))

        # for label, size in enumerate(sizes[1:], 1):
        #     self.phi_size_array[label-1, i] = size    

        self.phi_size_array[0, i], self.phi_size_array[1, i] = left_bottom_count, right_top_count

        if(i == run_times-1):
            fig, ax = plt.subplots()
            t_array = np.arange(run_times) * t_interval 
            phi_size_label_array = np.array(["large","small"])
            for j in range(self.phi_size_array.shape[0]):
                ax.plot(t_array, self.phi_size_array[j, :], label=f'{phi_size_label_array[j]}')
                last_x = t_array[-1]
                last_y = self.phi_size_array[j, -1]
                ax.annotate('({}, {})'.format(last_x, last_y), (last_x, last_y), textcoords="offset points", xytext=(0, 10), ha='center')

            fig.suptitle('wMax={}'.format(self.wmax))
            ax.legend(loc='best')
            ax.set_xlabel('t')
            ax.set_ylabel('area')
            valName = "phi_size"
            fig.savefig('{}/a_{}_gammaphi{}_Aphi{}_Kphi{}_phi0{}_wType{}_wmax{}_kwx{}_kwy{}_rw{}_Nx{}_Ny{}_h{}.png'.format(self.filename, valName, self.gammaphi, self.Aphi,self.Kphi, self.phi0,
                                                                            self.wType, self.wmax, self.kwx, self.kwy, self.rw, self.Lx, self.Ly, self.h))
            plt.close(fig)

    # get boundry that caculates flux 
    def get_flux_area(self, binary_image, x=0, y=0, L=0,  shape='phi'):  
        if shape == 'rec':
            binary_image[y-L:y+L+1, x-L, :] = [-1, 0]
            binary_image[y-L:y+L+1, x+L, :] = [1, 0]
            binary_image[y-L, x-L:x+L, :] = [0, -1]
            binary_image[y+L, x-L:x+L, :] = [0, 1]
            sqrt_2 = math.sqrt(2) / 2
            binary_image[y-L, x-L] = [-sqrt_2, -sqrt_2]
            binary_image[y-L, x+L] = [sqrt_2, -sqrt_2] 
            binary_image[y+L, x-L] = [-sqrt_2, sqrt_2] 
            binary_image[y+L, x+L] = [sqrt_2, sqrt_2] 
        
        if shape == 'phi':
            pos = (self.phi> 0.2) & (self.phi < 0.8)
            dphix = - d1xO4(self.phi, self.h)
            dphiy = - d1yO4(self.phi, self.h)
            dphi = np.sqrt(dphix ** 2 + dphiy ** 2)
            dphi_nonzero = np.where(dphi == 0, 1, dphi)
            dphix, dphiy = dphix /dphi_nonzero, dphiy /dphi_nonzero

            binary_image[pos==1, :] = np.array([dphix[pos==1],dphiy[pos==1]]).T

        return binary_image

    def draw_flux_sum_png(self, i, run_times, t_interval, x_list, y_list, l_list, num=2):
        
        if(i == 0):
            self.flux_sum = np.zeros((run_times, num))

        Jx = -d1xO4(self.mu, self.h) 
        Jy = -d1yO4(self.mu, self.h)
        
        flux_area = self.get_flux_area(np.zeros((self.Lx, self.Lx, 2)))
        # left_bottom
        self.flux_sum[i, 0] = np.sum(Jx[:self.Ly//2, :self.Lx//2] * flux_area[:self.Ly//2, :self.Lx//2, 0] + Jy[:self.Ly//2, :self.Lx//2] * flux_area[:self.Ly//2, :self.Lx//2, 1])
        # right_up
        self.flux_sum[i, 1] = np.sum(Jx[self.Ly//2:, self.Lx//2] * flux_area[self.Ly//2:, self.Lx//2:, 0] + Jx[self.Ly//2:, self.Lx//2:] * flux_area[self.Ly//2:, self.Lx//2:, 0] )
        # for j in range(num):
        #     tmp = self.get_flux_area(np.zeros((self.Lx, self.Lx, 2)), x_list[j], y_list[j], l_list[j])
        #     self.flux_sum[i, j] = np.sum(Jx * tmp[:, :, 0] + Jy * tmp[:, :, 1])

        if(i == run_times-1):
            fig, ax = plt.subplots()
            t_array = np.arange(run_times) * t_interval 
            label_array = np.array(["large","small"])
            for j in range(self.flux_sum.shape[1]):
                ax.plot(t_array[1:], self.flux_sum[1:, j], label=f'{label_array[j]}')

            fig.suptitle('wMax={}'.format(self.wmax))
            ax.legend(loc='best')
            ax.set_xlabel('t')
            ax.set_ylabel('flux_sum')
            valName = "flux_sum"
            fig.savefig('{}/a_{}_gammaphi{}_Aphi{}_Kphi{}_phi0{}_wType{}_wmax{}_kwx{}_kwy{}_rw{}_Nx{}_Ny{}_h{}.png'.format(self.filename, valName, self.gammaphi, self.Aphi,self.Kphi, self.phi0,
                                                                            self.wType, self.wmax, self.kwx, self.kwy, self.rw, self.Lx, self.Ly, self.h))
            plt.close(fig)



    def get_conf_number(self):
        file_names = os.listdir(self.filename)
        conf_files = [file for file in file_names if file.startswith("conf_")]
        return len(conf_files)

    # ��#                  
    # def plt_gammav(self):
    #     self.getData(0)
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(self.gammav, origin='lower')
    #     fig.colorbar(im)
    #     fig.savefig("{}/gammav.png".format(self.filename))
    

    # �ҵ�������kx��ky����Ӧ�ĵ�������������������������Ч����ǿ#
    def drawTerm3Item(self, kx, ky, t_string='0'):
        self.getData(t_string)
        central = [(int)(self.Ly / 2), (int)(self.Lx / 2)]
        Nky, Nkx = 16, 64
        self.wk = np.fft.fftshift(np.fft.ifft2(self.vortex))[central[0] - Nky:central[0] + Nky + 1,
                  central[1] - Nkx:central[1] + Nkx + 1]
        dx, dy = 2 * np.pi / self.Lx, 2 * np.pi / self.Ly
        list_3_kx, list_3_ky = self.find_couple(Nkx, kx), self.find_couple(Nky, ky)
        y_pos = list_3_ky[:, np.newaxis, :] + Nky
        x_pos = list_3_kx[np.newaxis, :, :] + Nkx
        y_val = list_3_ky * dy
        x_val = list_3_kx * dx
        tmp = self.wk[y_pos[:, :, 0], x_pos[:, :, 0]] * self.wk[y_pos[:, :, 1], x_pos[:, :, 1]]
        lhs = (y_val[:, np.newaxis, 0] ** 2 + x_val[np.newaxis, :, 0] ** 2)
        lhs[lhs == 0] = 1
        tmp *= (x_val[np.newaxis, :, 0] * y_val[:, np.newaxis, 1] - y_val[:, np.newaxis, 0] * x_val[np.newaxis, :,
                                                                                              1]) / lhs
        indices = np.argsort(-np.abs(tmp), axis=None)
        sorted_y, sorted_x = np.unravel_index(indices, tmp.shape)
        for i in range(10):
            print(f"kx: {list_3_kx[sorted_x[i], 0]}, ky:{list_3_ky[sorted_y[i], 0]} and kx: {list_3_kx[sorted_x[i], 1]}"
                  f", ky:{list_3_ky[sorted_y[i], 1]}, value is {np.abs(tmp)[sorted_y[i], sorted_x[i]]}")
        self.drawGeneral(t_string, abs(tmp), 'term3Item', interpla='none')

    # ��������kx��ky����Ӧ��ʱ��͸���Ҷ�ռ��vx��vy#
    def drawVk(self, kx, ky, t_string='0'):
        self.getData(t_string)
        central = [(int)(self.Ly / 2), (int)(self.Lx / 2)]
        Nky, Nkx = 31, 127
        self.wk = np.fft.fftshift(np.fft.ifft2(self.vortex))[central[0] - Nky:central[0] + Nky + 1,
                  central[1] - Nkx:central[1] + Nkx + 1]
        dx, dy = 2 * np.pi / self.Lx, 2 * np.pi / self.Ly
        kx, ky = np.arange(-Nkx, Nkx + 1) * dx, np.arange(-Nky, Nky + 1) * dy
        K2 = self.K_2(kx, ky)
        vxk_k = 1j * ky[:, np.newaxis] * K2 * self.wk
        vyk_k = -1j * kx[np.newaxis, :] * K2 * self.wk
        vx_k = np.fft.fft2(np.fft.ifftshift(vxk_k)).real  # k-wavenumber of vx
        vy_k = np.fft.fft2(np.fft.ifftshift(vyk_k)).real
        self.drawGeneral(t_string, abs(vxk_k), 'vxk_k', interpla='none')
        self.drawGeneral(t_string, abs(vyk_k), 'vyk_k', interpla='none')
        self.drawGeneral(t_string, abs(vx_k), 'vx_k')
        self.drawGeneral(t_string, abs(vy_k), 'vy_k')


def run():
    if run_cuda == 1:
        print("start draw")
        d = Draw(pre_filename+filename, input=input1)
        d.drawDynamic()
        print("finish") 
    else:
        for i in range(len(filename)):
            file = filename[i]
            print("start draw " + file)
            input2 = pd.read_csv("{}/{}/input.dat".format(pre_filename,file), names=['0']).T.to_numpy()
            input2 = pd.DataFrame(input2, columns=parameter_columns_name)
            d = Draw(pre_filename+file, input=input2)
            d.drawDynamic()
            print("finish " + file)


if __name__ == '__main__':
    time_start = time.time()
    run()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')