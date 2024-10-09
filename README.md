# Franka Walker
Preliminary project for humanoid locomotion using RL
</br>

## Gymnasium Environments
For this project, the '[Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/)', '[Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/)' environments from [Gymnasium](https://gymnasium.farama.org/) are used.

<a href="https://gymnasium.farama.org/_images/gymnasium-text.png">
  <img src="https://gymnasium.farama.org/_images/gymnasium-text.png" alt="Gymnasium" style="width: 250px; height: auto;"/>
</a>

<a href="https://gymnasium.farama.org/_images/walker2d.gif">
  <img src="https://gymnasium.farama.org/_images/walker2d.gif" alt="Walker2D GIF" style="width: 200px; height: auto;"/>
</a>
<a href="https://gymnasium.farama.org/_images/humanoid.gif">
  <img src="https://gymnasium.farama.org/_images/humanoid.gif" alt="Humanoid GIF" style="width: 200px; height: auto;"/>
</a>
</br>

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