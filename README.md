# SARMI: Safety-Augmented RL and MPC Integration

This repository will contain the official implementation of our paper:

**"Risk-Constrained On-Ramp Merging via Safety-Augmented Reinforcement Learning and Model Predictive Control"**

🚧 **Code Release Status:**  
The code is currently being cleaned and documented, and will be **publicly released in May 2026**.

---

## 📌 Overview
SARMI is a hierarchical framework that integrates:
- Combines safety-augmented RL with MPC to address autonomous on-ramp merging in dynamic traffic environments.
- Leverages a SAC-Discrete algorithm enhanced with an Augmented Lagrangian method incorporating barrier-like quadratic penalties to ensure adherence to safety constraints and mitigate oscillations during optimization.
- By integrating a Gaussian-based risk field model into the cost function, the RL agent proactively assesses collision risks using MPC-predicted future states.
- A dual safety mechanism is introduced, comprising action masking to eliminate invalid actions during exploration and action shielding to replace unsafe actions during execution.
- Theoretical analysis proves the equivalence between the optimal solutions of the primal and dual problems under the proposed framework.

The framework aims to enable **safe, efficient, and interpretable autonomous on-ramp merging**.

---

## 🎥 Simulation Video
More details about the simulation video can be found at:  
👉 https://github.com/JianLi000/Visualization

---

## 🔜 Coming Soon
- Full source code
- Training and evaluation scripts
- Pretrained models
- Detailed documentation

---

## 📄 Citation
If you find this work useful, please consider citing our paper (to be updated upon publication).

---

## 📬 Contact
For questions or collaboration, please feel free to contact lijian000@hnu.edu.cn or lyxc56@gmail.com.
