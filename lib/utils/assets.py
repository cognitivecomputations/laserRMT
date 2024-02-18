from enum import Enum
class PromptTemplate(str, Enum):
    chatml = "chatml"
    instruct = "instruct"