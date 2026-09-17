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
If you find this work useful, please consider citing our paper.

@ARTICLE{11454581,
  author={Li, Yang and Li, Jian and Huang, Wenjie and Yang, Qisong and Qin, Hongmao and Jiang, Xiaolong and Bian, Yougang and Hu, Manjiang and Hu, Yingbai},
  journal={IEEE Internet of Things Journal}, 
  title={Risk-Constrained On-Ramp Merging via Safety-Augmented Reinforcement Learning and Model Predictive Control}, 
  year={2026},
  volume={13},
  number={13},
  pages={28121-28137},
  keywords={Safety;Planning;Merging;Decision making;Reinforcement learning;Predictive control;Motion control;Autonomous vehicles;Markov decision processes;Costs;Action masking and shielding;augmented Lagrangian;autonomous on-ramp merging;model predictive control (MPC);reinforcement learning (RL)},
  doi={10.1109/JIOT.2026.3676898}}


Li, Yang, Jian Li, Wenjie Huang, Qisong Yang, Hongmao Qin, Xiaolong Jiang, Yougang Bian, Manjiang Hu, and Yingbai Hu. "Risk-Constrained On-Ramp Merging via Safety-Augmented Reinforcement Learning and Model Predictive Control." IEEE Internet of Things Journal (2026).


---

## 📬 Contact
For questions or collaboration, please feel free to contact lijian000@hnu.edu.cn or lyxc56@gmail.com.
