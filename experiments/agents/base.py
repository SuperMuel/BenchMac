from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def run(
        self,
        *,
        submission_id: str,
    ) -> str:
        """Execute the agent and return the solution as a patch string."""
        pass
