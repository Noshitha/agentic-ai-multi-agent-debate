# 🧠 Agentic AI: Exploring Multi-Agent Debate and Training-Free Self-Improvement  
*Analyzing reasoning dynamics and self-reflection strategies in large language models.*

---

## Overview

This repository is part of an ongoing research study on **Agentic AI**, focusing on how language models can **debate, self-evaluate, and improve their reasoning** without explicit fine-tuning.

We explore **Training-Free Group Relative Policy Optimization (GRPO)** as one of the mechanisms enabling **self-improvement through reflection**, and compare it to **multi-agent debate** frameworks where multiple models collaboratively reason to reach consensus.

---

## Core Ideas

| Concept | Description |
|----------|-------------|
| 🗣️ **Multi-Agent Debate** | Multiple agents (LLMs) argue for/against different reasoning paths to reach stable, explainable outcomes. |
| 🧭 **Training-Free GRPO** | A single-agent variant where the model learns contextually via group-relative feedback, not gradient updates. |
| 🔍 **Self-Reflection Loops** | The model summarizes its best reasoning and reuses it as "experience memory" for subsequent prompts. |
| 🧮 **Reward-Based Comparison** | Candidates are scored relative to ground truth or heuristic feedback — simulating reinforcement signals. |

---

## What This Repository Includes

- **Single-Agent GRPO Framework** (training-free)  
  → Generates multiple candidates, scores them, and learns via textual reflection.  
- **Evaluation Metrics for Agent Reasoning**  
  → Tracks how reasoning consistency, factual accuracy, and label precision evolve.  
- **Audit Traces for Interpretability**  
  → Stores reasoning steps, rewards, and extracted "experience rules" for transparency.  
- **HPC-Ready SLURM Pipelines**  
  → Fully compatible with UMass Unity GPU nodes for large-scale model experiments.

---

## Why This Matters:

Agentic AI represents the next leap beyond static LLMs — systems that:
- **Argue**, **reflect**, and **revise** their own reasoning.  
- **Adapt** to feedback without retraining.  
- Exhibit **goal-directed behavior** and **emergent cooperation**.

The GRPO layer acts as a **control experiment** — testing how far a *single agent* can self-improve *without external debate*.

