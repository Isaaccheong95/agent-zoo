from .config import SQLAgentSettings, load_settings
from .result import SQLAgentStructuredResult

__all__ = [
    "SQLAgent",
    "SQLAgentSettings",
    "SQLAgentStructuredResult",
    "ask_question",
    "ask_question_result",
    "build_root_agent",
    "load_settings",
    "root_agent",
]


def __getattr__(name: str):
    if name == "ask_question":
        from .runtime import ask_question

        return ask_question

    if name == "ask_question_result":
        from .runtime import ask_question_result

        return ask_question_result

    if name in {"SQLAgent", "build_root_agent", "root_agent"}:
        from .agent import SQLAgent, build_root_agent, root_agent

        return {
            "SQLAgent": SQLAgent,
            "build_root_agent": build_root_agent,
            "root_agent": root_agent,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
