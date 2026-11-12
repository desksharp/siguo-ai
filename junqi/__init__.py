"""junqi-ai：四国军旗模拟引擎与 AI。"""

from .engine import (  # noqa: F401
    AGENT_EXPLOIT,
    AGENT_RANDOM,
    AGENT_RULE,
    SETUP_HEURISTIC,
    SETUP_LEGACY,
    SETUP_RANDOM,
    TYPE_NAMES,
    Game,
    RuleConfig,
    board_info,
    dims,
    load_rules,
    make_config,
    selfplay,
)
