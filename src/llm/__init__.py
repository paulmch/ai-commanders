"""
LLM integration module for AI Commanders.

Provides LLM-controlled captains that make tactical decisions
during space combat simulations using OpenRouter/LiteLLM.
"""

from .client import CaptainClient, LLMResponse
from .captain import LLMCaptain, LLMCaptainConfig, CaptainPersonality
from .communication import CommunicationChannel, CaptainMessage, MessageType
from .victory import VictoryEvaluator, BattleOutcome
from .battle_runner import LLMBattleRunner, BattleConfig, BattleResult
from .tools import CAPTAIN_TOOLS
from .prompts import build_captain_prompt

__all__ = [
    # Client
    "CaptainClient",
    "LLMResponse",
    # Captain
    "LLMCaptain",
    "LLMCaptainConfig",
    "CaptainPersonality",
    # Communication
    "CommunicationChannel",
    "CaptainMessage",
    "MessageType",
    # Victory
    "VictoryEvaluator",
    "BattleOutcome",
    # Battle Runner
    "LLMBattleRunner",
    "BattleConfig",
    "BattleResult",
    # Tools
    "CAPTAIN_TOOLS",
    # Prompts
    "build_captain_prompt",
]
