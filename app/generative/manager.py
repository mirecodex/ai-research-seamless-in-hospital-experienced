import json
from functools import partial
from app.generative.engine import GenAI
from config.setting import env
from langchain_core.language_models.chat_models import BaseChatModel

VERTEX_CONFIG = {
    "gemini_regular": {
        "creator_method": "chatGgenai",
        "params": {"model": env.GEMINI_REGULAR_MODEL, "think": True},
    },
    "gemini_mini": {
        "creator_method": "chatGgenai",
        "params": {"model": env.GEMINI_MINI_MODEL, "think": False},
    },
    "gemini_thinking": {
        "creator_method": "chatGgenai",
        "params": {"model": env.GEMINI_THINKING_MODEL},
    },
    "openai_regular": {
        "creator_method": "chatAzureOpenAi",
        "params": {"model": env.OPENAI_REGULAR_MODEL},
    },
    "openai_mini": {
        "creator_method": "chatAzureOpenAi",
        "params": {"model": env.OPENAI_MINI_MODEL, "deployment": "002"},
    },
    "openai_thinking": {
        "creator_method": "chatAzureOpenAi",
        "params": {"model": env.OPENAI_THINKING_MODEL, "deployment": "002"},
    },
}

LITELLM_CONFIG = {
    "gemini_regular": {
        "creator_method": "chatLiteLLM",
        "params": {"model": env.GEMINI_REGULAR_MODEL},
    },
    "gemini_mini": {
        "creator_method": "chatLiteLLM",
        "params": {"model": env.GEMINI_MINI_MODEL},
    },
    "gemini_thinking": {
        "creator_method": "chatLiteLLM",
        "params": {"model": env.GEMINI_THINKING_MODEL},
    },
    "openai_regular": {
        "creator_method": "chatLiteLLM",
        "params": {"model": env.OPENAI_REGULAR_MODEL},
    },
    "openai_mini": {
        "creator_method": "chatLiteLLM",
        "params": {"model": env.OPENAI_MINI_MODEL},
    },
    "openai_thinking": {
        "creator_method": "chatLiteLLM",
        "params": {"model": env.OPENAI_THINKING_MODEL},
    },
}


class LLMManager:
    with open('app/generative/default.json', 'r') as f:
        DEFAULTS = json.load(f)

    def __init__(self):
        self._llms = {}
        self.gen_ai = GenAI()

        self.llm_configs = LITELLM_CONFIG if env.LLM_PROVIDER == "litellm" else VERTEX_CONFIG
        for name, config in self.llm_configs.items():
            if name in self.DEFAULTS and "default_params" in self.DEFAULTS[name]:
                config["default_params"] = self.DEFAULTS[name]["default_params"]

    def _get_llm(self, name: str, **override_params):
        if name in self._llms:
            return self._llms[name]

        config = self.llm_configs.get(name)
        if not config:
            raise AttributeError(f"No LLM named '{name}' is configured.")

        if config["params"].get("model"):
            base_params = config["params"].copy()
        else:
            base_params = config["default_params"].copy()

        base_params.update(override_params)
        final_params = base_params

        creator_method = getattr(self.gen_ai, config["creator_method"])

        llm_instance = creator_method(**final_params)
        self._llms[name] = llm_instance
        return llm_instance

    def __getattr__(self, name: str) -> BaseChatModel:
        if name in self.llm_configs:
            return partial(self._get_llm, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

manager = LLMManager()
