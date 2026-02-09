# LLM Tool Calling with Pydantic

This repository demonstrates how to build **structured, JSON-serializable LLM tool calls**
using **Pydantic** and **LangChain**.

## What This Shows
- Enforcing strict schemas on LLM outputs
- Validating and parsing tool inputs with Pydantic
- Dispatching tool calls using JSON
- Building reliable, production-style LLM pipelines

## Included Examples
- Weather extraction
- Spam classification
- Math operations with tool dispatching

## Why This Matters
Unstructured LLM output breaks systems.
Structured schemas make LLMs usable in real APIs, agents, and automation.

## Brutal Reality Check (Important)
- This is **NOT advanced agentic AI**.  
- This is **foundational plumbing**.
- Recruiters don’t care about tutorials — they care that you understand **why structure matters**.

If you can’t explain:
> “Why Literal + Pydantic prevents tool hallucination”

then you **don’t actually understand this yet**.

If you want, next we can:
- Turn this into a **LangGraph node**
- Convert it to **Gemini-only**
- Wrap it as a **FastAPI tool server**
- Or integrate it into your **DhruvaIQ agent stack**

Say the word.
