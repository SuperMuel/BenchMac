import uuid

from pydantic import BaseModel, Field


class PatchGenerationTask(BaseModel):
    """
    A task for generating patches using an agent configuration on a specific instance.

    This represents a single unit of work in the patch generation process.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the task.",
    )

    instance_id: str = Field(
        ...,
        description="Unique identifier for the benchmark instance to process.",
    )
    model_name: str = Field(
        ...,
        description="The name of the model to use for patch generation "
        "(e.g., 'mistral/devstral-medium-2507').",
    )
