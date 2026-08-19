"""Agentic re-implementation of the EDA judging (agentic_workflow branch).

One `JudgeAgent` per judge model (config.yaml eda.judges), each wrapping a
LangChain LCEL chain around ChatOllama, with per-agent call logging
(`monitor.py`) and a plain function-based orchestrator (`orchestrator.py`) —
deliberately no LangGraph, per plan §1 ("if a step can be a function, it is
a function").

Author: Dr. Chakresh Kr. Singh
Prepared for Roland Berger (RB GenAI evaluation).
"""
