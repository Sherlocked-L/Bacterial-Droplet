#!/usr/bin/perl -w

use strict;
use warnings;

# define something
my $FileName = "run_cuda.pl";
my $CuName = "TTSH3D.cu";
my $gpu_node = 1;


my @a_mu = (1);
my @a_cA = (-1);
my @a_A = (0);
my @a_eta = (-0.187);
my @a_LA = (1);
my @a_alpha = (-0.02);
my @a_beta = (4.366);
my @a_lambda_0 = (9);
my @a_kappa = (-0.364);
my @a_aPhi = (1);
my @a_kPhi = (1);
my @a_gamma = (1);

my $Tend = 1000;
my $dT = 0.002;
my $dTe = 10;
my $init0 = 0.0001;

my $N = 64;
my $h = 1;

for my $mu (@a_mu) {
for my $cA (@a_cA) {
for my $A (@a_A) {
for my $eta (@a_eta) {
for my $LA (@a_LA) {
for my $alpha (@a_alpha) {
for my $beta (@a_beta) {
for my $lambda_0 (@a_lambda_0) {
for my $kappa (@a_kappa) {
for my $aPhi (@a_aPhi) {
for my $kPhi (@a_kPhi) {
for my $gamma (@a_gamma) {

    my $t_mu = "\$mu = $mu;";
    my $t_cA = "\$cA = $cA;";
    my $t_A = "\$A = $A;";
    my $t_eta = "\$eta = $eta;";
    my $t_LA = "\$La = $LA;";
    my $t_alpha = "\$alpha = $alpha;";
    my $t_beta = "\$beta = $beta;";
    my $t_lambda_0 = "\$lambda_0 = $lambda_0;";
    my $t_kappa = "\$kappa = $kappa;";
    my $t_aPhi = "\$aPhi = $aPhi;";
    my $t_kPhi = "\$kPhi = $aPhi;";
    my $t_gamma = "\$gamma = $gamma;";

    my $t_Tend = "\$Ts = $Tend;";
    my $t_dTe = "\$dte = $dTe;";
    my $t_dT = "\$dt0 = $dT;";
    my $t_init0 = "\$init0 = $init0;";

    my $t_Nx = "\$Nx = $N;";
    my $t_Ny = "\$Ny = $N;";
    my $t_Nz = "\$Nz = $N;";
    my $t_h = "\$h = $h;";
    
    my $output = "eta_$eta-LA_$LA-alpha_$alpha-beta_$beta-mu_$mu-cA_$cA-A_$A-lambda0_$lambda_0-aPhi_$aPhi-kPhi_$aPhi-gamma_$gamma-N_$N-h_$h-T_$Tend-DT_$dTe";

    my $OutPut = "           \$direData='$output';";

    my $Gpu_node ="\$gpu_node       = $gpu_node;";

# read .cu file
open(my $pl_file, '<', $FileName) or die "Cannot open file: $!";
my @lines = <$pl_file>;
close($pl_file);

# modifiy the content
$lines[29] = $t_mu."\n";
$lines[30] = $t_cA."\n";
$lines[31] = $t_A."\n";
$lines[32] = $t_eta."\n";
$lines[33] = $t_LA."\n";
$lines[34] = $t_alpha."\n";
$lines[35] = $t_beta."\n";
$lines[36] = $t_lambda_0."\n";
$lines[37] = $t_kappa."\n";

$lines[39] = $t_aPhi."\n";
$lines[40] = $t_kPhi."\n";
$lines[41] = $t_gamma."\n";

$lines[21] = $t_Tend."\n";
$lines[22] = $t_dTe."\n";
$lines[18] = $t_dT."\n";
$lines[43] = $t_init0."\n";

$lines[13] = $t_Nx."\n";
$lines[14] = $t_Ny."\n";
$lines[15] = $t_Nz."\n";
$lines[17] = $t_h."\n";

$lines[49] = $OutPut."\n";
$lines[26] = $Gpu_node."\n";

# re-write the .cu file
open($pl_file, '>', $FileName) or die "Cannot open file: $!";
print $pl_file @lines;
close($pl_file);

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

if (-e "TTSH3D.out") {
system("rm TTSH3D.out");
}

system("perl run_cuda.pl");

}}}}}}}}}}}}
