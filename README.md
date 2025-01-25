# On the Fly Atomistic Calculator

This is an On-the-Fly Atomistic Calculator based on Gaussian Process Regression (GPR), designed as an ASE add-on calculator. It incorporates a hybrid approach by combining:


1. `Base calculator`: A high-fidelity ab initio calculator (e.g., DFT) that serves as the reference (“gold standard”) for accurate but computationally expensive calculations.
2. `Surrogate GPR calculator`: A computationally inexpensive model trained on-the-fly to approximate the base calculator and accelerate simulations.

## Motivation

Many atomistic simulations—such as geometry optimization, barrier calculations, molecular dynamics (MD), and equation-of-state (EOS) simulations—require sampling a large number of atomic configurations in a compact phase space. The workhorse method for such calculations is often Density Functional Theory (DFT), which provides a balance between accuracy and computational cost. However, DFT can still be prohibitively expensive, particularly for large systems or simulations for many snapshots.


## How It Works?

1.	Initially, the calculator invokes the base calculator (e.g., a DFT code) to compute energy, forces, and stress tensors for each atomic configuration.
2.	These computed data points are used to train an internal surrogate model based on Gaussian Process Regression (GPR).
3.	As the surrogate model accumulates training data, it gradually learns the energy and force landscape of the system.
4.	Once the model achieves sufficient accuracy, it begins predicting the mean and variance of energy, forces, and stress tensors, enabling efficient on-the-fly calculations.
5.	If the predicted uncertainty is low, the model replaces the expensive DFT evaluations. Otherwise, the base calculator is called again to refine the model.

## Installation
```
export CC=g++ && pip install .
```

## A quick example

Below illustrates an example to run NEB calculation with the hybrid calculator

```python
from gpr_calc.GPRANEB import GP_NEB
from gpr_calc.calculator import GPR
from ase.optimize import BFGS
from ase.mep import NEB
from ase.calculators.emt import EMT

initial_state = 'database/initial.traj'
final_state = 'database/final.traj'
num_images = 5
fmax = 0.05

print("\nInit the model")
neb_gp = GP_NEB(initial_state, 
                final_state, 
                num_images=num_images)

print("\nGet the initial images")
images = neb_gp.generate_images(IDPP = False)
    
# Set Base calculator and Run NEB
for image in images: 
    image.calc = EMT()
neb = NEB(images)
opt = BFGS(neb) 
opt.run(fmax=fmax)
neb_gp.plot_neb_path(images, figname='Ref.png')

# Test gpr calculator
kernel = 'Dot'
images = neb_gp.generate_images(IDPP = False)

print("\nCreate the initial GPR model")
neb_gp.set_GPR(kernel=kernel, noise_e=fmax/10)
neb_gp.train_GPR(images)
print(neb_gp.model)

# Set hybrid calculator
for image in images:
    image.calc = GPR(base_calculator=neb_gp.useCalc,
                     ff=neb_gp.model,
                     freq=10, #update frequency
                     return_std=True)

print("\nRun actual NEB")
neb = NEB(images)
opt = BFGS(neb) 
opt.run(fmax=fmax)

# Plot results
neb_gp.plot_neb_path(images, figname=kernel+'.png')

print(neb_gp.model)
print("\nTotal number of base calls", neb_gp.model.count_use_base)
print("Total number of surrogate calls", neb_gp.model.count_use_surrogate)
print("Total number of gpr_fit calls", neb_gp.model.count_fits)
```

The output should look like the following. In the process of NEB calculation, the base calculators were used frequently in the beginning. 
Some of the representative data points were then added to the GPR model. After the model becomes more accurate, the predictions from surrogate model were useded more frequently. 


```
------Gaussian Process Regression------
Kernel: 2.000**2 *Dot(length=2.000) 5 energy (0.005) 15 forces (0.050)

Loss:      -64.796  2.000  2.000  0.005
Loss:     5526.244  0.010  1.606  0.003
Loss:      -76.615  1.336  1.869  0.004
Loss:      -91.056  0.673  1.737  0.003
Loss:      -83.325  0.236  1.651  0.003
Loss:      -94.218  0.490  1.701  0.003
Loss:     -100.400  0.414  1.686  0.003
Loss:     -100.516  0.421  1.688  0.003
Train Energy [   5]: R2 0.9997 MAE  0.000 RMSE  0.000
Train Forces [  45]: R2 0.9999 MAE  0.005 RMSE  0.009
------Gaussian Process Regression------
Kernel: 0.421**2 *Dot(length=1.688) 5 energy (0.003) 15 forces (0.025)


Run actual NEB
From base model, E: 0.002/3.721/3.727, F: 0.171/1.639/1.632
From surrogate,  E: 0.002/0.003/4.219, F: 0.104/0.030/3.517
From base model, E: 0.002/3.718/3.725, F: 0.170/1.630/1.629
      Step     Time          Energy          fmax
BFGS:    0 19:18:53        4.219058        3.517425
From base model, E: 0.001/3.642/3.650, F: 0.171/1.261/1.172
From surrogate,  E: 0.002/0.003/3.917, F: 0.107/0.030/2.568
From base model, E: 0.001/3.639/3.648, F: 0.170/1.253/1.169
BFGS:    1 19:18:54        3.916727        2.568004
From base model, E: 0.002/3.515/3.546, F: 0.170/0.421/0.466
====================== Update the model =============== 14
From base model, E: 0.003/3.666/3.724, F: 0.152/0.220/0.315
From base model, E: 0.002/3.538/3.545, F: 0.073/0.449/0.462
BFGS:    2 19:18:57        3.723818        0.332303
From base model, E: 0.002/3.535/3.542, F: 0.074/0.434/0.448
From base model, E: 0.003/3.664/3.720, F: 0.151/0.203/0.289
From base model, E: 0.002/3.534/3.541, F: 0.073/0.432/0.444
BFGS:    3 19:18:59        3.720366        0.304576
From base model, E: 0.002/3.518/3.529, F: 0.074/0.355/0.363
====================== Update the model =============== 11
From base model, E: 0.002/3.679/3.702, F: 0.040/0.172/0.138
From surrogate,  E: 0.001/0.003/3.525, F: 0.021/0.030/0.370
BFGS:    4 19:19:02        3.701898        0.257257
From surrogate,  E: 0.001/0.003/3.516, F: 0.028/0.030/0.345
From base model, E: 0.002/3.676/3.699, F: 0.040/0.322/0.268
From surrogate,  E: 0.001/0.003/3.517, F: 0.028/0.030/0.344
BFGS:    5 19:19:04        3.699291        0.284075
From surrogate,  E: 0.002/0.003/3.505, F: 0.035/0.030/0.374
From base model, E: 0.002/3.672/3.699, F: 0.045/0.378/0.310
From surrogate,  E: 0.002/0.003/3.507, F: 0.035/0.030/0.376
BFGS:    6 19:19:05        3.699091        0.348929
From base model, E: 0.002/3.476/3.531, F: 0.051/0.413/0.450
====================== Update the model =============== 11
From base model, E: 0.002/3.683/3.706, F: 0.039/0.290/0.288
From surrogate,  E: 0.001/0.003/3.526, F: 0.034/0.030/0.448
BFGS:    7 19:19:09        3.705957        0.417406
From surrogate,  E: 0.001/0.003/3.516, F: 0.030/0.030/0.326
From base model, E: 0.002/3.676/3.696, F: 0.030/0.178/0.176
From surrogate,  E: 0.001/0.003/3.516, F: 0.029/0.030/0.326
BFGS:    8 19:19:11        3.696143        0.286430
From surrogate,  E: 0.001/0.003/3.510, F: 0.026/0.030/0.378
From surrogate,  E: 0.002/0.003/3.672, F: 0.022/0.030/0.073
From surrogate,  E: 0.001/0.003/3.511, F: 0.026/0.030/0.378
BFGS:    9 19:19:11        3.672378        0.079445
From surrogate,  E: 0.001/0.003/3.509, F: 0.027/0.030/0.379
From surrogate,  E: 0.002/0.003/3.672, F: 0.024/0.030/0.070
From surrogate,  E: 0.001/0.003/3.509, F: 0.027/0.030/0.379
BFGS:   10 19:19:12        3.671673        0.077038
From surrogate,  E: 0.001/0.003/3.506, F: 0.031/0.030/0.382
From surrogate,  E: 0.002/0.003/3.670, F: 0.028/0.030/0.061
From surrogate,  E: 0.001/0.003/3.506, F: 0.031/0.030/0.382
BFGS:   11 19:19:13        3.670466        0.084920
From surrogate,  E: 0.001/0.003/3.504, F: 0.032/0.030/0.383
From surrogate,  E: 0.002/0.003/3.670, F: 0.030/0.030/0.059
From surrogate,  E: 0.001/0.003/3.505, F: 0.031/0.030/0.382
BFGS:   12 19:19:14        3.670026        0.084529
From surrogate,  E: 0.001/0.003/3.500, F: 0.034/0.030/0.381
From base model, E: 0.002/3.668/3.692, F: 0.035/0.080/0.082
From surrogate,  E: 0.001/0.003/3.500, F: 0.034/0.030/0.378
BFGS:   13 19:19:15        3.691848        0.081847
From surrogate,  E: 0.001/0.003/3.498, F: 0.034/0.030/0.384
From base model, E: 0.002/3.667/3.691, F: 0.033/0.057/0.066
From surrogate,  E: 0.001/0.003/3.499, F: 0.033/0.030/0.380
BFGS:   14 19:19:16        3.690697        0.068272
From surrogate,  E: 0.001/0.003/3.498, F: 0.032/0.030/0.380
From surrogate,  E: 0.002/0.003/3.666, F: 0.030/0.030/0.041
From surrogate,  E: 0.001/0.003/3.499, F: 0.032/0.030/0.376
BFGS:   15 19:19:17        3.666124        0.036292
From base model, E: 0.001/3.315/3.315, F: 0.059/0.083/0.038
From base model, E: 0.001/3.316/3.316, F: 0.059/0.088/0.047
------Gaussian Process Regression------
Kernel: 0.948**2 *Dot(length=1.723) 5 energy (0.003) 60 forces (0.025)


Total number of base calls 21
Total number of surrogate calls 29
Total number of gpr_fit calls 4
```
