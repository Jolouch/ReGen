from typing import List

from pydantic import BaseModel, Field


# diffusion output
class CompletedTruncatedSpecifications(BaseModel):
    """completed truncated specifications"""
    completed_truncated_specifications: str = Field(description="the completed truncated specifications")


class DiffusionSpecifications(BaseModel):
    completedSpecifications: List[CompletedTruncatedSpecifications]


class VanillaAnswer(BaseModel):
    incompleteness: str = Field(description="the detected incompleteness")
    new_specification: str = Field(
        description="the generated new requirement specification based on the incompleteness")


class VanillaGeneration(BaseModel):
    answers: List[VanillaAnswer]


# extract action output
class Action(BaseModel):
    """action"""
    action: str = Field(description="the action")


class Actions(BaseModel):
    actions: List[Action]


class Step(BaseModel):
    analysis: str = Field(description="the analysis")


class FinalAnswer(BaseModel):
    absent_element: str = Field(description="the absent situation or the absent action(objects)")
    new_specification: str = Field(description="the generated requirement specification related to the absent element")


class ReGeneration(BaseModel):
    steps: list[Step] = Field(description="each step of the analysis")
    final_answer: FinalAnswer = Field(description="the final answer")
