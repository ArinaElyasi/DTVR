"""DTV-REA - the DTV Requirements Agent.

An interviewing agent that elicits digital-twin requirements from a single
stakeholder and produces a structured requirements document aligned with the
DTV framework (Bitencourt et al., 2025, Int. J. Production Research,
DOI 10.1080/00207543.2025.2524516), steps 1-2 only.

Governing rule of the architecture:

    Decisions in code. Language in the model.

The LLM does exactly two jobs - phrase questions, and structure answers. What
to ask next, whether an extraction is acceptable, and whether the session is
finished are all deterministic Python.

Central research claim, enforced by :mod:`dtv_rea.validator` rather than by
prompt instructions: the agent never invents a number.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
