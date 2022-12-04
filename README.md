# Self Delusion
This notebook replicates the results of the paper [Shaking the foundations: delusions in sequence models for interaction and control](https://arxiv.org/pdf/2110.10819.pdf).

## Assumptions (see Remark 11 in paper)
The paper assumes a [multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) environment. The bandit has 5 arms and the best arm is `l=1`. The expert interacts with the bandit using the following policy:
``` Python
action_probs = [0.6, 0.1, 0.1, 0.1, 0.1]
```

Based on the expert action the bandit returns a reward (observation):
``` Python
if action == latent:
    observation_probs = [0.25, 0.75]
else:
    observation_probs = [0.75, 0.25]
```

Depending on whether the environment configuration (best arm) can be observed, two cases can be distinguished:
* Fully observable: Environment configuration `l` is observable
* Partially observable: Environment configuration `l` is not observable.

We assume that we cannot observe the environment configuration `l` when training the model.

## Model
A model is trained based on the interactions between the expert and the environment (5-armed bandit). This model is then used to interact with the environment instead of the expert.

Model – fully observable
```
P(l,a1,o1,a2,o2,a3) = P(a3|a1,o1,a2,o2) * P(a2,o2|a1,o1) * P(a1,o1|l) * P(l)
```

Model – partially observable
```
P(a1,o1,a2,o2,a3) = P(a3|a1,o1,a2,o2) * P(a2,o2|a1,o1) * P(a1,o1)
```


```python
from model import Model

# (a1,o1) = (1,1)
# (a2,o2) = (2,0)
model: Model = Model()
action_probs = model.predict(past_interactions=[(1,1), (2,0)])
print("P(a3|a1=1,o1=1,a2=2,o2=0) =", action_probs)

# Probability of action being 1
print("P(a3=1|a1=1,o1=1,a2=2,o2=0) =", action_probs[0])
```

    P(a3|a1=1,o1=1,a2=2,o2=0) = [0.491304347826087, 0.14347826086956525, 0.12173913043478261, 0.12173913043478261, 0.12173913043478261]
    P(a3=1|a1=1,o1=1,a2=2,o2=0) = 0.491304347826087



```python
import random
import config

SEQ_LENGTH: int = 20

def sample_observation(latent: int, action: int) -> int:
    probs = config.prob_observation(latent, action)
    return random.choices(config.OBSERVATION_SPACE, weights=probs, k=1)[0]

def sample_action(probs: list[float]) -> int:
    return random.choices(config.ACTION_SPACE, weights=probs, k=1)[0]

def run_agent(latent: int, seed: int) -> list[int]:
    if seed is not None:
        random.seed(seed)
    past_interactions: list[tuple[int, int]] = []
    actions: list[int] = []
    model = Model()
    for _ in range(0, SEQ_LENGTH):
        probs = model.predict(past_interactions)
        action = sample_action(probs)
        actions.append(action)
        observation = sample_observation(latent, action)
        past_interactions += [(action, observation)]
    return actions

print("Actions generated by the model over 20 interactions:")
for index in range(3):
    actions: list[int] = run_agent(1, index)
    action_groups = [actions[i:i+5] for i in range(0, len(actions), 5)]
    text: str = f"example {index + 1} – actions "
    for action_group in action_groups:
        text += " ".join([str(x) for x in action_group]) + "  "
    print(text)
```

    Actions generated by the model over 20 interactions:
    example 1 – actions 5 4 5 5 5  5 3 5 5 5  4 5 5 5 5  5 3 5 5 5  
    example 2 – actions 1 4 2 3 1  4 4 2 1 5  1 5 1 1 1  1 1 1 4 2  
    example 3 – actions 5 1 5 5 5  5 5 5 5 5  1 5 4 5 3  4 5 5 5 5  


The expert interacts with the bandite regardless of the past actions. However, if we use our model to imitate the expert, the past actions suddenly have an impact on the probability of the next action.
