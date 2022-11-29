import config

class Model():

    def _joint_probability(self, interactions, latents) -> float:
        joint_probs: float = 0
        for latent in latents:
            prob: float = 1.0
            next_action_prob: float = 1.0
            for interaction in interactions:
                action = interaction[0]
                if len(interaction) == 1:
                    next_action_prob = config.prob_action(latent)[action-1]
                    break
                observation = interaction[1]
                prob *= config.prob_action(latent)[action-1] * config.prob_observation(action, latent)[observation]
            joint_probs += prob * next_action_prob
        return joint_probs * config.PROB_LATENT

    def _cond_probability(self, prob_before, prob_after) -> float:
        return prob_after / prob_before

    def predict(self, past_interactions: list[tuple[int, int]]) -> list[float]:
        probs: list[float] = []
        latent: int = config.ACTION_SPACE[0]
        remaining_latents: list[int] = config.ACTION_SPACE[1:]
        for action in config.ACTION_SPACE:
            next_action: list[(int, )] = [(action,)]
            prob_before_latent: float = self._joint_probability(interactions=past_interactions, latents=[latent])
            prob_after_latent: float = self._joint_probability(interactions=past_interactions + next_action, latents=[latent])
            prob_before_remaining_latent: float = self._joint_probability(interactions=past_interactions, latents=remaining_latents)
            prob_after_remaining_latent: float = self._joint_probability(interactions=past_interactions + next_action, latents=remaining_latents)
            prob_before: float = prob_before_remaining_latent + prob_before_latent
            prob_after: float = prob_after_remaining_latent + prob_after_latent
            probs.append(self._cond_probability(prob_before, prob_after))
        return probs


if __name__ == "__main__":
    model = Model()
    probs = model.predict(past_interactions=[(1,1), (2,0)])
    assert probs == [0.491304347826087, 0.14347826086956525, 0.12173913043478261, 0.12173913043478261, 0.12173913043478261]

    probs = model.predict(past_interactions=[(1,1), (2,0), (3,1)])
    assert probs == [0.325, 0.125, 0.32500000000000007, 0.1125, 0.1125]