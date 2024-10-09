# Franka Walker
Preliminary project for humanoid locomotion using RL
</br></br>

## Gymnasium Environments
For this project, the '[Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/)', '[Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/)' environments from [Gymnasium](https://gymnasium.farama.org/) are used.

<a href="https://gymnasium.farama.org/_images/gymnasium-text.png">
  <img src="https://gymnasium.farama.org/_images/gymnasium-text.png" alt="Gymnasium" style="width: 250px; height: auto;"/>
</a>
</br>

<img src="https://github.com/user-attachments/assets/cb9fe43d-a024-4212-8e44-383165a56c19" width="30%"/>
<img src="https://github.com/user-attachments/assets/ccfbfded-d52c-4a3d-af64-eac77a5689e3" width="30%"/>
</br></br>

## Installation
__Step 1.__ Create conda environment
```shell
$ conda env create -f environment.yaml
```
__Step 2.__ Activate conda environment
```shell
$ conda activate walker
```
__Step 3.__ Install dependencies
``` shell
$ pip install -r requirements.txt
```
</br>

## Usage
### Train
> Walker2D
``` shell
$ python src/walker.py
```
> Humanoid
``` shell
$ python src/humanoid.py
```
### Test
> Walker2D
``` shell
$ python src/viz_walker.py
```
> Humanoid
``` shell
$ python src/viz_humanoid.py
```
### Plot
You can check the plots of rewards and constraints for each environment!
> Reward
``` shell
$ python src/plot_reward.py
```
> Constraints
``` shell
$ python src/plot_constraint.py
```

</br>

## References
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [IPO: Interior-point Policy Optimization under Constraints](https://arxiv.org/abs/1910.09615)
- [Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion](https://arxiv.org/abs/2308.12517)
