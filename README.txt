#Deep Learning and Heuristic Exploration for Risk-Aware Scheduling in Construction Projects

This repository provides the standardized experimental protocols, dependency management details, and procedural workflows for the risk-aware scheduling framework proposed in our paper. 

To protect intellectual property prior to final publication, the complete executable source code is currently abstracted. However, the comprehensive guidelines below ensure that independent researchers can strictly reconstruct, verify, and reproduce the proposed two-phase (Offline DL Training + Online Heuristic Scheduling) algorithmic framework.

## 1. Environment and Dependency Management

To prevent issues arising from library deprecation and to ensure strict reproducibility of tensor computations and heuristic sampling, the experimental framework is locked to the following environment and dependencies.

**Base Environment:**
* Python == 3.12

**Core Dependencies (`requirements.txt` equivalent):**
* `torch == 2.8.0` (For tensor operations and neural network backends)
* `torch_geometric == 2.5.1` (For Heterogeneous Graph Neural Network modeling)
* `blitz-bayesian-pytorch == 0.2.8` (For Bayesian Neural Network layers and uncertainty quantification)
* `pymoo == 0.6.1.5` (Standardized framework for Multi-Objective Evolutionary Algorithms: NSGA-II, NSGA-III, MOEA/D, and performance metrics)
* `numpy == 2.3.2`
* `pandas == 2.3.1`

## 2. Version Control and Determinism

A major challenge in reproducing stochastic scheduling and deep learning algorithms is the inherent sampling noise. We enforce strict deterministic version control via the following mechanisms:
* **Global Random Seeds:** The random seed is explicitly hardcoded across all mathematical modules (`np.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)`) before any training or evolutionary loops begin.
* **Common Random Numbers (CRN):** In the final high-fidelity Monte Carlo validation phase, CRN arrays are pre-generated to ensure that all compared metaheuristic algorithms face the exact same stochastic duration scenarios, thereby eliminating sampling bias.

## 3. Standardized Experimental Workflow

The experiments are executed strictly following the proposed Two-Phase architecture:

### Phase A: Offline Training of the Risk-Aware Evaluator
1. **Data Preparation:** Download the standard PSPLIB datasets (j10, j20, j30). Parse the JSON instances to extract nominal durations, resource limits, and precedence relations.
2. **Graph Construction:** Convert each project instance into a HeteroData graph (nodes: `task`, `resource`; edges: `precedes`, `consumes`). Apply zero-padding to align feature vector dimensions uniformly across different project scales.
3. **Dynamic Data Augmentation:** For each training epoch, inject additive Gaussian noise (uncertainty level = 0.3) into the nominal durations to dynamically generate synthetic actual durations as ground truth.
4. **Model Training:** Train the `GT-BNN` (GNN + Transformer + BNN) architecture. Optimize the model using the Negative Log-Likelihood (NLL) loss alongside a KL-divergence regularization term.
5. **Solidification:** Upon convergence, freeze the model weights and export the checkpoint .

### Phase B: Online Scheduling and High-Fidelity Validation
1. **Guided Search Deployment:** Load the frozen Evaluator. For a new, unseen test instance, perform a single forward pass to output the predicted expected duration multiplier ($\mu$) and standard deviation ($\sigma$) for each task.
2. **Metaheuristic Optimization Execution:** 
   * The framework uniformly executes a suite of specific Multi-Objective Evolutionary Algorithms to ensure a fair comparison. This suite includes the proposed **RA-NSGA-III**, alongside state-of-the-art peer algorithms: **NSGA-II**, **MA** (Memetic Algorithm), and **MOEA/D** (Multi-Objective Evolutionary Algorithm based on Decomposition).
   * **Population Initialization:** Set Population Size = 50 (strictly aligned with the reference-point generation logic of NSGA-III for a 3-objective space). For RA-NSGA-III, apply the DL-guided dual-scoring mechanism.
   * **Evolutionary Loop:** Execute the selected algorithm for a maximum budget of 100 generations (a widely accepted budget for computationally expensive stochastic scheduling).
   * **Decoding:** Apply the standard Serial Generation Scheme (SGS) to map chromosome representations into feasible project schedules.
3. **Post-Validation (The 5,000 MC Runs):**
   * Extract the final non-dominated Pareto front solutions.
   * Subject **each** solution to a high-fidelity Monte Carlo simulation with **5,000** independent runs based on a Log-Normal duration distribution.
   * Calculate the ultimate robust metrics: $E[Makespan]$, $E[Energy]$, and $CVaR_{0.95}[Makespan]$.
4. **Performance Evaluation:** Normalize the objectives and compute the Hypervolume (HV) and Inverted Generational Distance (IGD) metrics strictly utilizing the `pymoo` built-in evaluation modules.
