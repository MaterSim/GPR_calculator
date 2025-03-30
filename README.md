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
pip install .
```

## A quick example

Below illustrates an example to run NEB calculation with the hybrid calculator

```python
from ase.calculators.emt import EMT
from gpr_calc.NEB import neb_calc, neb_generate_images, neb_plot_path
from gpr_calc.gaussianprocess import GaussianProcess as GP
from gpr_calc.calculator import GPR

# Set parameters
initial, final = 'database/initial.traj', 'database/final.traj'
num_images = 5
fmax = 0.05

# Run NEB with EMT calculator
images = neb_generate_images(initial, final, num_images)
images, energies = neb_calc(images, EMT(), fmax=fmax, steps=100)
data = [(images, energies, 'EMT')]

# Run NEB with gpr calculator in different etols
for etol in [0.02, 0.1, 0.2]:
    images = neb_generate_images(initial, final, num_images)

    # initialize GPR model
    gp_model = GP.set_GPR(images, EMT(),
                          kernel='RBF',
                          noise_e=etol/len(images[0]),
                          noise_f=0.1)
    # Set GPR calculator
    calc = GPR(base_calculator=EMT(), ff=gp_model)

    # Run NEB calculation
    images, energies = neb_calc(images, calc, fmax=fmax, steps=100)
    print(gp_model)
    data.append((images, energies, f'GPR-{etol:.2f}'))

neb_plot_path(data, figname='NEB-test.png')
```

The output should look like the following. In the process of NEB calculation, the base calculators were used frequently in the beginning.
Some of the representative data points were then added to the GPR model. After the model becomes more accurate, the predictions from surrogate model were useded more frequently.

```
Calculate E/F for image 0: 3.314754
Calculate E/F for image 1: 3.727147
Calculate E/F for image 2: 4.219952
Calculate E/F for image 3: 3.724974
Calculate E/F for image 4: 3.316117
------Gaussian Process Regression (0/2)------
Kernel: 1.00000**2 *RBF(length=0.10000) 5 energy (0.00769) 15 forces (0.10000)


Update GP model => 20
Loss:       -2.916  1.000  0.100
Loss:     1821.480  0.010 10.000
Loss:       48.811  0.527  4.827
Loss:      -51.328  0.835  1.750
Loss:      -52.140  0.898  1.717
Loss:      -53.035  1.025  1.634
Loss:      -53.163  1.078  1.589
From Surrogate ,  E: 0.054/0.100/3.729, F: 0.102/0.120/1.660
From Surrogate ,  E: 0.066/0.100/4.215, F: 0.155/0.120/3.489
From Surrogate ,  E: 0.054/0.100/3.725, F: 0.102/0.120/1.651
      Step     Time          Energy          fmax
BFGS:    0 20:02:22        4.214882        3.489110
From Surrogate ,  E: 0.054/0.100/3.647, F: 0.103/0.120/1.284
From Surrogate ,  E: 0.066/0.100/3.918, F: 0.154/0.120/2.609
From Surrogate ,  E: 0.053/0.100/3.644, F: 0.103/0.120/1.278
BFGS:    1 20:02:24        3.917948        2.608887
From Base model , E: 0.053/3.500/3.546, F: 0.182/0.350/0.400
From Base model , E: 0.100/3.512/3.738, F: 0.315/0.423/0.434
From Base model , E: 0.053/3.499/3.545, F: 0.183/0.349/0.399
BFGS:    2 20:02:29        3.737970        0.488517
From Base model , E: 0.053/3.504/3.544, F: 0.174/0.377/0.421
From Base model , E: 0.093/3.527/3.723, F: 0.303/0.594/0.309
...
...
...
From Surrogate ,  E: 0.081/0.200/3.532, F: 0.091/0.120/0.404
From Surrogate ,  E: 0.095/0.200/3.695, F: 0.061/0.120/0.073
From Surrogate ,  E: 0.081/0.200/3.529, F: 0.090/0.120/0.388
BFGS:   38 20:08:23        3.695437        0.074676
From Surrogate ,  E: 0.081/0.200/3.531, F: 0.092/0.120/0.404
From Surrogate ,  E: 0.095/0.200/3.695, F: 0.062/0.120/0.059
From Surrogate ,  E: 0.081/0.200/3.528, F: 0.092/0.120/0.388
BFGS:   39 20:08:27        3.695121        0.065271
From Surrogate ,  E: 0.081/0.200/3.529, F: 0.095/0.120/0.400
From Surrogate ,  E: 0.095/0.200/3.694, F: 0.062/0.120/0.051
From Surrogate ,  E: 0.081/0.200/3.526, F: 0.094/0.120/0.390
BFGS:   40 20:08:31        3.694400        0.052724
From Surrogate ,  E: 0.081/0.200/3.529, F: 0.097/0.120/0.393
From Surrogate ,  E: 0.095/0.200/3.694, F: 0.062/0.120/0.039
From Surrogate ,  E: 0.081/0.200/3.526, F: 0.097/0.120/0.391
BFGS:   41 20:08:35        3.693717        0.039118
From Surrogate ,  E: 0.082/0.200/3.344, F: 0.069/0.120/0.041
From Surrogate ,  E: 0.082/0.200/3.346, F: 0.067/0.120/0.048
------Gaussian Process Regression (0/2)------
Kernel: 2.80314**2 *RBF(length=1.52921) 7 energy (0.01538) 55 forces (0.10000)
Total number of base calls: 22
Total number of surrogate calls: 106
Total number of gpr_fit calls: 4
```

The generated model will be stored as

- a `json` file to store the parameters
- a `db` file to store training structure, reference energy and forces

The final result is shown as follows
 <img src="https://raw.githubusercontent.com/MaterSim/GPR_calculator/master/database/NEB-test.png" alt="NEB"/>


It can be reused as follows

```python
from gpr_calc.gaussianprocess import GaussianProcess as GPR

gpr = GPR.load('test-RBF-gpr.json')
print(gpr)
gpr.validate_data(show=True)
gpr.fit(opt)
```
For more productive example using VASP as the base calculator, please check the [Examples](https://github.com/MaterSim/GPR_calculator/tree/main/examples).
