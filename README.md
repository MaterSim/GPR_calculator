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
opt.run(fmax=0.01)
neb_gp.plot_neb_path(images, figname='Ref.png')

# Test gpr calculator
images = neb_gp.generate_images(IDPP = False)
for kernel in ['Dot', 'RBF']:
    print("\nCreate the initial GPR model")
    neb_gp.set_GPR(kernel=kernel, noise_e=0.002)
    neb_gp.train_GPR(images)
    print(neb_gp.model)

    # Set hybrid calculator
    for image in images:
        image.calc = GPR(base_calculator=neb_gp.useCalc,
                         ff=neb_gp.model,
                         return_std=True)

    print("\nRun actual NEB")
    neb = NEB(images)
    opt = BFGS(neb, trajectory='neb.traj') 
    opt.run(fmax=0.01)

    # Plot results
    neb_gp.plot_neb_path(images, figname=kernel+'.png')
```


