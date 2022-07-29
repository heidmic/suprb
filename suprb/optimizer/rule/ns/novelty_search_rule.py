from suprb.rule import Rule


class NoveltySearchRule():
    def __init__(self, rule: Rule, novelty_score: float):
        self.rule = rule
        self.novelty_score = novelty_score
