# MPM-Dambreak

![License](https://img.shields.io/badge/license-MIT-blue.svg)

**MPM-Dambreak** is a fluid simulation implementation of the classic ''Dam Break'' scenario using the **Moving Least Squares Material Point Method (MLS-MPM)**. This project demonstrates the coupling of fluid dynamics with rigid boundaries using robust numerical schemes.


---

## 1. Introduction and How to Run

### Overview
This repository contains the source code for simulating a column of water collapsing under gravity and collides with a rigid body (the Dam Break problem). 

### How to Run

To run the simulations, follow the command-line instructions located at the bottom of each `.cpp` source file. These commands will generate particle data files (`.ply`).

Use **[ParaView](https://www.paraview.org/download/)** to visualize the results and watch the animation.

#### Example: Running on Windows
For instance, to run `dambreak_stationary_obstacle.cpp` on Windows, execute the following commands in your terminal:

```bash
# 1. Compile (ensure you have a C++ compiler like MinGW)
g++ dambreak_stationary_obstacle.cpp -o dambreak_stationary_obstacle.exe -std=c++14 -O3 -lgdi32 -fopenmp

# 2. Run the executable
.\dambreak_stationary_obstacle.exe
```
The generated `.ply` files will be saved in the `ply_stationary_obstacle` directory.

---

## 2. Brief Introduction to MLS-MPM
<!-- 
The **Material Point Method (MPM)** is a hybrid Lagrangian-Eulerian method that uses both particles and a background grid.

### Why MLS-MPM?
Standard MPM can suffer from "grid crossing errors" where particles crossing cell boundaries cause numerical instability. **MLS-MPM (Moving Least Squares MPM)** improves upon this by using a continuous polynomial reconstruction of the velocity field, significantly improving energy conservation and stability.



### The Algorithm Cycle
1.  **P2G (Particles to Grid):** Transfer mass and momentum from particles to the background grid using B-spline shape functions.
2.  **Grid Update:** Solve the equations of motion on the grid nodes (gravity, boundary conditions).
3.  **G2P (Grid to Particles):** Interpolate updated velocities back to particles and update their positions and deformation gradients. -->


APIC    

---

## 3. Equation of State for Water


### Equation of State

To simulate water as a weakly compressible fluid, we define its pressure-density relationship using an **Equation of State (EOS)**. Since we do not solve the expensive Poisson equation for pressure, we use **[Tait–Murnaghan Equation of State](https://en.wikipedia.org/wiki/Tait_equation#Tait%E2%80%93Murnaghan_equation_of_state)**, which allows small density variations to generate pressure that resists compression.

The pressure $p$ for a particle is calculated as:

$$
p = K \left[  J^{-\gamma} - 1 \right]
$$

Where:
* $K$: The bulk modulus (controls incompressibility), set to $K=50.0$ in code.
* $J$: The determinant of the deformation gradient $\mathbf{F}$ ( $J = \det(\mathbf{F})$ ). It represents the **volume ratio** ( $J = \frac{V_{\text{current}}}{V_{\text{initial}}}$ ).
    * $J < 1$ indicates compression (density increases).
    * $J = 1$ indicates no volume change.
* $\gamma$: The stiffness parameter, set to $\gamma=7.0$ in code.

<!-- > **Implementation Note:** In the code, density $\rho$ is estimated based on the particle's deformation gradient $J$ (where $J = \det(\mathbf{F})$). -->


The Cauchy stress is modeled through the **[Linear stress constitutive equation](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#Compressible_flow)** (viscocity terms neglected for now)

$$
\mathbf{\sigma} = -p\mathbf{I}
$$

### Determinant Update
According to the continuity equation, the rate of change of volume is characterized by the **divergence** of the fluid velocity field $\mathbf{v}$ ($\nabla \cdot \mathbf{v}$) through the following equation:

$$
\frac{DJ}{Dt} = J(\nabla \cdot \mathbf{v})
$$

In APIC or MLS-MPM, the matrix `p.C` carried by the particle approximates the **Velocity Gradient**, denoted as $\nabla \mathbf{v}$.

$$
C \approx \nabla \mathbf{v} = \begin{bmatrix}
\frac{\partial v_x}{\partial x} & \frac{\partial v_x}{\partial y} & \frac{\partial v_x}{\partial z} \\
\frac{\partial v_y}{\partial x} & \frac{\partial v_y}{\partial y} & \frac{\partial v_y}{\partial z} \\
\frac{\partial v_z}{\partial x} & \frac{\partial v_z}{\partial y} & \frac{\partial v_z}{\partial z}
\end{bmatrix}
$$

This is because, if we know the velocity $\mathbf{v}(\mathbf{x}_p)$ at a particle $\mathbf{x}_p$ and want to find the velocity $\mathbf{v}(\mathbf{x}_i)$ at a nearby grid node $\mathbf{x}_i$ , we can perform a first-order Taylor expansion of the velocity field:

$$
\mathbf{v}(\mathbf{x}_i) \approx \mathbf{v}(\mathbf{x}_p) + \nabla \mathbf{v}(\mathbf{x}_p) \cdot (\mathbf{x}_i - \mathbf{x}_p)
$$

while in APIC, the local velocity represented by a particle at the grid node $\mathbf{x}_i$ is (APIC paper, Eq.(8))

$$
 \mathbf{v}(\mathbf{x}_p) + \mathbf{C}_p (\mathbf{x}_i - \mathbf{x}_p)
$$

Combining the two equations, we have that $C \approx \nabla \mathbf{v}$. Thus, divergence of the velocity field $\nabla \cdot \mathbf{v}$ is given by

$$
\nabla \cdot \mathbf{v} = \frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} + \frac{\partial v_z}{\partial z} = \text{Trace}(C)
$$

Use the first-order Taylor expansion to discretize the time derivative $\frac{DJ}{Dt}$:

$$
J_{n+1} \approx J_n + \Delta t \cdot \frac{DJ}{Dt}
$$

Substituting $\frac{DJ}{Dt} = J(\nabla \cdot \mathbf{v})$:

$$
\begin{aligned}
J_{n+1} &\approx J_n + \Delta t \cdot (J_n \cdot \nabla \cdot \mathbf{v}) \\
&= J_n \cdot (1 + \Delta t \cdot \nabla \cdot \mathbf{v}) \\
&= J_n \cdot (1 + \Delta t \cdot \text{Trace}(C))
\end{aligned}
$$



---

## 4. Coulomb Friction Model
