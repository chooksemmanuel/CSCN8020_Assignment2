# CSCN8020 Assignment 2: Taxi Gymnasium Environment with Q-Learning

**Student:** Emmanuel Ihejiamaizu  
**Student ID:** 9080005  
**Course:** CSCN8020 - Reinforcement Learning Programming  

## Project summary
This project implements tabular Q-learning for Gymnasium's Taxi-v3 environment. The agent learns to navigate a 5 x 5 grid, pick up a passenger, and complete a correct drop-off while minimizing wasted movement and illegal pickup or drop-off actions. The notebook uses an object-oriented design, records an execution log, and reports real outcomes for the required baseline, learning-rate, exploration-factor, and best-combination experiments. It also explains Taxi's 500-state / 6-action representation, maps each Python stage to Q-learning pseudocode, presents metrics and plots, and evaluates the final learned policy greedily to separate learned performance from training-time exploration.

## Repository links
- Repository: https://github.com/chooksemmanuel/CSCN8020_Assignment2
- Cloneable URL: https://github.com/chooksemmanuel/CSCN8020_Assignment2.git

## How to run
1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Jupyter Notebook or JupyterLab.
5. Open `CSCN8020_Assignment2_Q_Learning_Taxi.ipynb`.
6. Run all cells from top to bottom.

The notebook regenerates plots in `assets/plots/`, writes execution details to `logs/qlearning_experiments.log`, and saves the final experiment table to `results/experiment_summary.csv`.

## Major files
- `CSCN8020_Assignment2_Q_Learning_Taxi.ipynb` - full implementation, explanations, controlled experiments, plots, and final evaluation.
- `assets/plots/` - generated training and experiment-comparison plots.
- `logs/qlearning_experiments.log` - readable log of hyperparameters, progress checkpoints, and final metrics.
- `results/experiment_summary.csv` - final experiment metrics in a compact reusable format.
- `requirements.txt` - required Python packages.
- `.gitignore` - excludes virtual environments, caches, and unnecessary runtime files.
- `submission/Emmanuel_Ihejiamaizu_CSCN8020_Assignment2_Summary.pdf` - one-page Brightspace submission file.

## Q-learning implementation
The agent stores values in a 500 x 6 Q-table. During training it uses epsilon-greedy exploration, then applies the temporal-difference update:

`Q(S,A) <- Q(S,A) + alpha * [R + gamma * max_a Q(S',a) - Q(S,A)]`

At evaluation time, the agent acts greedily without exploration. This makes the final performance results reflect the learned policy rather than deliberately random training actions.
