"""chronicle — Merkle-DAG agent traceability.

A content-addressed causal DAG of agent events, with integrity verification
and a semantic search overlay. Built as a research artifact; see the notebook
for motivation and design.
"""
from .event import Event, canonical_json
from .core import Chronicle
from .semantic import HashingEmbedder, SentenceTransformerEmbedder, SemanticIndex
from .integrity import verify, merkle_root, root
from .ergonomics import span, tool, active
from .branch import branch_at, diff

__version__ = "0.1.0"

__all__ = [
    "Chronicle",
    "Event",
    "canonical_json",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "SemanticIndex",
    "verify",
    "merkle_root",
    "root",
    "span",
    "tool",
    "active",
    "branch_at",
    "diff",
    "demo",
    "__version__",
]


def demo() -> "Chronicle":
    """Run a small end-to-end toy agent and return its Chronicle.

    No LLM dependency — a deterministic in-memory corpus stands in for a
    retrieval tool. The shape of the recorded DAG (prompt -> thought ->
    tool_call -> tool_result -> thought -> answer) is the canonical example
    used throughout the notebook.
    """
    c = Chronicle()
    prompt = c.record(
        "prompt",
        actor="user",
        payload={"text": "find me a vegan lasagna recipe"},
    )

    @tool("recipe_search")
    def recipe_search(query: str) -> list:
        corpus = {
            "vegan lasagna": ["tofu ricotta", "cashew bechamel", "spinach", "zucchini"],
            "chicken curry": ["onions", "garlic", "garam masala"],
        }
        for k, v in corpus.items():
            if all(w in k for w in query.lower().split()):
                return v
        return []

    with span(c, actor="agent:planner", parent=prompt.id) as s:
        s.thought("the user wants a vegan lasagna recipe; I'll search the corpus")
        hits = recipe_search("vegan lasagna")
        s.thought(f"found {len(hits)} ingredients; assembling the answer")
        s.answer(f"Here's a vegan lasagna using: {', '.join(hits)}")

    return c
