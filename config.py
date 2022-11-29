
ACTION_PROB: float = 0.6
REWARD_PROB: float = 0.75
ACTION_SPACE: list[int] = [1,2,3,4,5]
OBSERVATION_SPACE: list[int] = [0,1]
PROB_LATENT: float = 0.20

def prob_action(latent: int) -> list[float]:
    action_probs = [(1.0-ACTION_PROB)/(len(ACTION_SPACE)-1)] * len(ACTION_SPACE)   
    action_probs[latent-1] = ACTION_PROB
    return action_probs

def prob_observation(latent: int, action: int) -> list[float]:
    if latent == action:
        return [1.0 - REWARD_PROB, REWARD_PROB]
    else:
        return [REWARD_PROB, 1.0 - REWARD_PROB]