# Flexibility Function

## Key Information
- **License:** MIT License  
- **Language:** Python 3.x  
- **Dependencies:** numpy, pandas, scipy, matplotlib  

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Getting Started](#getting-started)  
3. [Usage](#usage)  
4. [Data Requirements](#data-requirements)  
5. [Visualization](#visualization)  
6. [Results](#results)  
7. [License](#license)  

---

## Introduction
The **Flexibility Function (FF)** algorithm is designed to represent flexibility dynamics. This utilizes a stochastic nonlinear differential equation for modeling dynamic demand in response to price deviations.
It uses **I-spline** basis functions to construct functions *g(U)*, which capture the relation between price and flexibility state. It also utilizes a polynomial of order 6 to construct the function *f(X)* for state equations.  

Key features include:  
- Maximum likelihood approach for parameter estimation with **SciPy optimization**
- Solvers: SLSQP, Nelder-Mead, L-BFGS-B
- Using multiple initial values for optimization problem solving   
- Validation with R² and RMSE metrics  
- Visualization of intermediate functions and final predictions  

---

## Getting Started

### 1. Install Dependencies
Install the required Python libraries:

```bash
pip install numpy pandas scipy matplotlib

