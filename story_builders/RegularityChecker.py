import random

class RegularityChecker:
    def __init__(self):
        pass

    def is_regular(self, program: str, logger=None) -> bool:
        """
        Base implementation: simply returns True or False at random.
        (This method is expected to be overridden by subclasses.)
        """
        return random.choice([True, False])
    