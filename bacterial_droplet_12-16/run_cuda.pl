#!/usr/bin/perl -w
$sharefile = ".";
$CuName = "TTSH3D.cu";
system("nvcc -std=c++14 -I /usr/include/cuda/ -L /usr/lib64/ --cudart=shared -lcufft $sharefile/$CuName -o TTSH3D.out");
# system("nvcc -std=c++14 -I /usr/include/cuda/ -L /usr/lib64/ --cudart=shared -lcufft $sharefile/TTSH3D_noActive.cu -o TTSH3D.out");
# Simulation parameters
$TimeScheme  = 1;             # 1-Euler, 2-Predictor Corrector 3-RKC2
$RKStage     = 16;
$ExplImpi    = 1;	#  0-Implicit, 1-Explicit 
$AdapTime    = 0;             # 0-dt=dt0, 1-Adpative time step.
$ExpoData    = 1;             # 1-important fields, -1-Incl Other variables.
$isSimplified   = 1;          # 1-Uniform, 2-Sin
$checkUnStable  = 0;
$Nx = 64;
$Ny = 64;
$Nz = 64;
$Nb          = 3;            # Boundary width.
$h = 1;
$dt0 = 0.002;
$InitCond    = 1;       # when InitCond is 0, you can use set_init.py to set Init
$T0          = 0;       # when T0>0, InitFileName in active_nematics.cu should be selected
$Ts = 1000;
$dte = 10;
$dtVarC      = 0.01;
$dtMax       = 0.000005;       # Maximum dt allowed.
$ReadSuccess = -8848;             # Used to check if input file is correctly read.
$gpu_node       = 1;

# Parameters for velocity field
$mu = 1;
$cA = -1;
$A = 0;
$eta = -0.187;
$La = 1;
$alpha = -0.02;
$beta = 4.366;
$lambda_0 = 9;
$kappa = -0.364;

$aPhi = 1;
$kPhi = 1;
$gamma = 1;

$init0 = 0.0001;
# $CuName = "TTSH3D_noActive.cu";

            print"running\n";

            # $direData="test14_gammaphi$gammaphi-Aphi$Aphi-Kphi$Kphi-wType$wType-wmax$wmax-kwx$kwx-kwy$kwy-rw$rw-Nx$Nx-Ny$Ny-h$h-rate0";
           $direData='eta_-0.187-LA_1-alpha_-0.02-beta_4.366-mu_1-cA_-1-A_0-lambda0_9-aPhi_1-kPhi_1-gamma_1-N_64-h_1-T_1000-DT_10';
            # $direData='aPhi_$aPhi-kPhi_$kPhi-gamma_$gamma-A_$A-N_$Nx-h_$h-Ts_$Ts-dte_$dte';

            open (FILE, '>input.dat');  #simulation parameters
            print FILE "$TimeScheme\n";
            print FILE "$RKStage\n";
            print FILE "$ExplImpi\n";
            print FILE "$AdapTime\n";
            print FILE "$ExpoData\n";
            print FILE "$InitCond\n";
            print FILE "$isSimplified\n";
            print FILE "$checkUnStable\n";
            print FILE "$Nx\n";
            print FILE "$Ny\n";
            print FILE "$Nz\n";                                                                                             
            print FILE "$Nb\n";
            print FILE "$h\n";
            print FILE "$dt0\n";
            print FILE "$T0\n";
            print FILE "$Ts\n";
            print FILE "$dte\n";
            print FILE "$dtVarC\n";
            print FILE "$dtMax\n";
            print FILE "$mu\n";
            print FILE "$cA\n";
            print FILE "$alpha\n";
            print FILE "$beta\n";
            print FILE "$eta\n";
            print FILE "$La\n";
            print FILE "$kappa\n";
            print FILE "$lambda_0\n";
            print FILE "$aPhi\n";
            print FILE "$kPhi\n";
            print FILE "$gamma\n";
            print FILE "$A\n";
            print FILE "$init0\n";
            print FILE "$ReadSuccess\n";
            close (FILE);

            $datafile = "data_$gpu_node";
            $destination_dir = "./conf-data/$direData";
            if ($InitCond == -1)  {
                system("mv ../conf-data/$direData $datafile");  
            }          
            else {
                system("mkdir $datafile");
            }

            # read .cu file
            open(my $cu_file, '<', $CuName) or die "Cannot open file: $!";
            my @lines1 = <$cu_file>;
            close($cu_file);

            # modifiy the content
            my $a_gpu_node ="int gpu_node = $gpu_node;";
            $lines1[41] = $a_gpu_node."\n";

            # re-write the .cu file
            open($cu_file, '>', $CuName) or die "Cannot open file: $!";
            print $cu_file @lines1;
            close($cu_file);

            print "$direData\n";

            system("cp run_cuda.pl $datafile");
            system("cp $sharefile/TTSH3D.cu $datafile");
            # system("cp $sharefile/set_init.py $datafile");
            open (FILE, '>readme.dat');  #simulation parameters
            if ($InitCond == 0) {
                print FILE "0\n";
            } else {
                print FILE "$T0\n";
            };
            print FILE "$Ts\n";
            print FILE "$dte\n";
            print FILE "$Nx\n";
            print FILE "$Ny\n";
            close(FILE);
            system("mv readme.dat $datafile");
            system("mv input.dat $datafile");
            # system("cp *.m $datafile");
            # system("mv InitCond.dat $datafile");

            if ($InitCond == 0){
                system("python ../src_share/set_init.py")
            }
            system("CUDA_VISIBLE_DEVICES=$gpu_node ./TTSH3D.out");
            
            if($T0 > 0){  # cp file in $datafile to $destination_dir
                system("cp -rf $datafile/* $destination_dir");
                system("rm -r $datafile");
            }
            else{
                system("mv $datafile ./conf-data/$direData");
            }

             system("python $sharefile/readDraw.py conf-data/$direData");

            #  system("python $sharefile/characteristicVelocity.py conf-data/$direData");

