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
Create the initial GPR model
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
From base model, E: 0.002/4.219/4.220, F: 0.104/3.517/3.521
From base model, E: 0.002/3.718/3.725, F: 0.170/1.630/1.629

      Step     Time          Energy          fmax
BFGS:    0 19:02:13        4.219952        3.520758
From base model, E: 0.001/3.642/3.650, F: 0.171/1.261/1.172
From base model, E: 0.002/3.917/3.937, F: 0.107/2.563/2.176
====================== Update the model =============== 11
From base model, E: 0.002/3.642/3.648, F: 0.081/1.168/1.169
BFGS:    1 19:02:16        3.938718        2.178510
From base model, E: 0.002/3.544/3.555, F: 0.085/0.543/0.522
From base model, E: 0.002/3.718/3.720, F: 0.112/0.404/0.423
From base model, E: 0.002/3.543/3.554, F: 0.084/0.538/0.517
BFGS:    2 19:02:18        3.720113        0.422757
From base model, E: 0.002/3.526/3.541, F: 0.104/0.475/0.442
====================== Update the model =============== 13
From base model, E: 0.002/3.712/3.711, F: 0.044/0.169/0.189
From surrogate,  E: 0.002/0.003/3.536, F: 0.019/0.030/0.451
BFGS:    3 19:02:21        3.710666        0.268977
...
...
...
...
...
BFGS:   16 19:02:46        3.692232        0.163631
From surrogate,  E: 0.001/0.003/3.477, F: 0.026/0.030/0.346
From base model, E: 0.002/3.689/3.691, F: 0.027/0.135/0.130
From surrogate,  E: 0.001/0.003/3.476, F: 0.027/0.030/0.344
BFGS:   17 19:02:47        3.691261        0.130144
From surrogate,  E: 0.001/0.003/3.477, F: 0.027/0.030/0.339
From base model, E: 0.002/3.687/3.690, F: 0.026/0.080/0.061
From surrogate,  E: 0.001/0.003/3.476, F: 0.028/0.030/0.339
BFGS:   18 19:02:49        3.689780        0.060540
From surrogate,  E: 0.001/0.003/3.477, F: 0.027/0.030/0.342
From base model, E: 0.002/3.687/3.689, F: 0.024/0.080/0.028
From surrogate,  E: 0.001/0.003/3.476, F: 0.028/0.030/0.344
BFGS:   19 19:02:50        3.689174        0.048480

------Gaussian Process Regression------
Kernel: 1.162**2 *Dot(length=1.738) 5 energy (0.003) 63 forces (0.025)

Total number of base calls 33
Total number of surrogate calls 29
Total number of gpr_fit calls 5
```
