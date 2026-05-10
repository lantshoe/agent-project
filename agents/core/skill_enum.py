from enum import Enum

class Skill(str, Enum):
    # Use models that are better at reasoning
    REASONING = "reasoning"
    # better at coding
    CODING = "coding"
    # fast and cheap
    SIMPLE = "simple"