# chronicle

Merkle-DAG agent traceability. Built from `notebook/chronicle.ipynb`.

```python
import chronicle
c = chronicle.demo()
print("root:", c.root())
print("verify:", c.verify())
for evt, score in c.search("vegan recipe"):
    print(f"{score:+.3f}  {evt.kind:<12} {evt.actor}")
```

See the notebook for the design and the headline properties:
deterministic cross-run identity, tamper detection, causal `why()`,
semantic search, and counterfactual branching.
