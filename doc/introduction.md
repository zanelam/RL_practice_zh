### 强化学习是什么？

强化学习Reinforcement Learning解决的问题：（不完全可知的）Markov决策过程，交互式问题（interactive problem）

强化学习不同于监督学习supervised learning，也不同于无监督学习unsupervised learning

one challenge：the trade-of between exploration and exploitation. 

> The agent has to ***exploit*** what it has already experienced in order to obtain reward, but it also has to ***explore*** in order to make better action selections in the future.   

反馈被延迟

### 例子

共同特征：

- involve ***interaction*** between an active decision-making ***agent*** and its ***environment***

- to achieve a ***goal*** despite ***uncertainty*** about environment
- agent's actions are permitted to affect the future state of the environment
- require foresight and planning

### 要素

agent、environment、以及四个核心要素：policy、reward signal、value function、model of the environment（option）

- policy：the core of a reinforce learning agent. 可以理解为state到action的映射

- reward signal：定义了强化学习问题的目标。每个time step，环境向agent发送一个标量数值（reward），而agent的目标就是最大化长期总收益。reward signal对应短期收益。

- value function：对应长期收益。但确定价值很难，环境能提供的是reward，而value需要综合评估。a
  method for efficiently estimating values（价值评估方法）是几乎所有强化学习算法最重要的组成部分。  

  > the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state.  

- model of the environment：有model-based和model-free两类方法，model-free method不依赖模型，是通过试错学习的。

### 关于环境

- fully observable environments：对应马尔可夫决策过程（Markov Decision Process，MDP），

  agent state = environment state = information state = MDP state

- partially observable environments：对应部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process，POMDP）

  此时agent state $\ne$ environment state
  
  