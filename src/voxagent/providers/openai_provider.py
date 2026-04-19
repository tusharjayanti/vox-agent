from voxagent.llm import GenerateResult

_ROADMAP_MSG = (
    "OpenAIProvider is a placeholder. Implementation pending — see DESIGN.md §11 roadmap."
)


class OpenAIProvider:
    def __init__(self, **kwargs) -> None:
        pass

    async def generate(
        self,
        messages: list[dict],
        system: str,
        temperature: float,
    ) -> GenerateResult:
        raise NotImplementedError(_ROADMAP_MSG)

    async def complete(
        self,
        prompt: str,
        temperature: float,
    ) -> str:
        raise NotImplementedError(_ROADMAP_MSG)
