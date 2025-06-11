<h1 align="center">
  <b>SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending</b>
</h1>

This is the official code release of paper SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending. This repository contains code and ckpts of our framework **SkillBlender** and benchmark **SkillBench**.

**[[paper (coming soon)]]() [[project page]](https://usc-gvl.github.io/SkillBlender-web/)**

<div align=center>
    <img src="docs/img/teaser.jpg" width=100%>
</div>

## Installation

Our code is easy to install and run. Please refer to [1.INSTALLATION.md](./docs/1.INSTALLATION.md) for detailed instructions.

## Training, playing, and evaluation

Please refer to [2.RUNNING.md](./docs/2.RUNNING.md) for how to train, play, and evaluate the skills and tasks of SkillBlender in our SkillBench.
We also include some troubleshooting tips in this document.

## Changing observations

We support both state-based and vision-based observations. The default observation is state-based, which includes joint positions, velocities, etc. You can also use ego-centric vision observations for high-level tasks.
Please refer to [3.OBSERVATION.md](./docs/3.OBSERVATION.md) for more details on how to change the observation types and structures.

## Changing humanoid

We support three distinct humanoid robots: H1, G1, and H1-2. Each robot has its own set of skills and tasks. However, you can also easily change the humanoid robot in the codebase to your own robot by modifying the config files and the robot class.
For more details, please refer to [4.HUMANOID.md](./docs/4.HUMANOID.md).

## Release status

- [x] Release code for **SkillBlender** and **SkillBench**.
- [x] Release ckpts for H1 skills and tasks.
- [ ] Release ckpts for G1 and H1-2 skills and tasks.
- [ ] Release sim2sim and sim2real code for primitive skill deployment.
- [ ] More to come... (Feel free to open issues and PRs!)

**Please stay tuned for any updates of the dataset and code!**

## Acknowledgement

This project is based on [legged_gym](https://github.com/leggedrobotics/legged_gym), [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [humanoid-gym](https://github.com/roboterax/humanoid-gym), [humanplus](https://github.com/MarkFzp/humanplus), and [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym). We thank the authors for their contributions.

## Citation

If you find this work helpful, please consider citing:

```
BibTeX
```