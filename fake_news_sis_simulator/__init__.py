# External
from importlib_metadata import version

# Project
from .edo import edo_sis_k1, edo_sis_k2
from .markov import q_matrix_k1, evolution_from_markov, markov_timeline_probability_matrix
from .simulator import Event, EventType, Simulator, EventOrigin, TimelineType, TopologyType

try:
    __version__: str = version(__name__)  # type: ignore
except Exception:  # pragma: no cover
    # Internal
    import traceback
    from warnings import warn

    warn(f"Failed to set version due to:\n{traceback.format_exc()}", ImportWarning)
    __version__ = "0.0a0"

__all__ = (
    "__version__",
    "Event",
    "TopologyType",
    "EventType",
    "Simulator",
    "EventOrigin",
    "TimelineType",
    "q_matrix_k1",
    "markov_timeline_probability_matrix",
    "evolution_from_markov",
)
