## AutoOptLib (Automated Design of Metaheuristic Optimization Algorithms)
[![GNU General Public License v3.0](https://img.shields.io/badge/license-GNU%20GPL--v3.0-green.svg)](https://github.com/qz89/AutoOpt/blob/main/LICENSE)
![](https://img.shields.io/badge/Matlab-%3E%3D%202018a%20-blue.svg)
![](https://img.shields.io/badge/Windows-Pass-brightgreen.svg)
![](https://img.shields.io/badge/MacOS-Pass-brightgreen.svg)

## What does AutoOptLib provide?
AutoOptLib is a MATLAB library for automatically designing metaheuristic optimizers. It provides:

*  **Rich library of design choices** - Over 40 representative metaheuristic components for designing algorithms for continuous, discrete, and permutation problems with/without constraints and uncertainties.
*  **Flexibility to designing diverse algorithms** - Design algorithms with diverse structures in a single run, enables great possibility to find novel and efficient algorithms.
*  **Various design objectives and techniques** - Various design objectives, e.g., solution quality, runtime, and anytime performance. Different design techniques, e.g., racing, intensification, and surrogate.
*  **Good accessibility** - GUI for users to input problems, manage the algorithm design process, make experimental comparisons, and visualize results with simple one-clicks. 
*  **Easy extensibility** - Easily add new algorithmic components, objectives, and techniques by a uniform interface. 

##  What are AutoOptLib's benefits:
* **Save labor resources and time** - Human experts may cost days or weeks to conceive, build up, and verify the optimizers; AutoOptLib would saves such labor resources and time costs with today's increasing computational power. 
* **Democratize metaheuristic optimizers** - Through automated algorithm design techniques, AutoOptLib would democratize the efficient and effective use of metaheuristic optimizers. This is significant for researchers and practitioners with complicated optimization problem-solving demands but without the expertise to distinguish and manage suitable optimizers among various choices. 
* **Surpass human algorithm design** - By fully exploring potential design choices and discovering novelties with computing power, AutoOptLib would go beyond human experience and gain enhanced performance regarding human problem-solving.
* **Promote metaheuristic research** - With a uniform collection of related techniques, AutoOptLib would promote research of the automated algorithm design and metaheuristic fields and be a tool in the pursuit of autonomous and general artificial intelligence systems.

## How to use AutoOptLib?
Follow the steps below to use AutoOptLib:

1. Download source code and add it to MATLAB path.

2. Implement the targeted optimization problem.

3. Define the space for designing algorithms.

4. Run AutoOptLib by command or GUI. 

Refer to the Documentation [Web](https://AutoOpt.readthedocs.io/) / [PDF](https://github.com/qz89/AutoOpt/blob/main/AutoOptLib%20for%20Matlab/Documentation.pdf) for user guide.

AutoOptLib is developed and maintained by Qi Zhao at the Department of Computer Science and Engineering, Southern University of Science and Technology. Users should reference the following papers if using AutoOptLib in their publications:
```
@article{zhao2023autooptlib,
  title={AutoOptLib: Tailoring Metaheuristic Optimizers via Automated Algorithm Design},
  author={Zhao, Qi and Yan, Bai and Hu, Taiwei and Chen, Xianglong and Duan, Qiqi and Yang, Jian and Shi, Yuhui},
  journal={arXiv preprint arXiv:2303.06536},
  year={2023}
}
```

## Contact
Users may ask question in [Issues block](https://github.com/qz89/AutoOpt/issues) and upload contributions by [Pulling request](https://github.com/qz89/AutoOpt/pulls). 

For any question, comment, or suggestion, please contact Dr. Qi Zhao, email: zhaoq@sustech.edu.cn.
