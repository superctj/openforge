class ExperimentState:
    def __init__(self):
        self.default_f1_score = -1
        self.default_associated_accuracy = -1

        self.best_f1_score = -1
        self.best_associated_accuracy = -1
        self.best_hp_config = None

        self.worst_f1_score = -1
        self.worst_associated_accuracy = -1
        self.worst_hp_config = None
