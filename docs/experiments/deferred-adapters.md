# Deferred P10 Adapters

LLM/EvoQuant candidate generation and learned market world models remain
intentionally unimplemented while Gates C--F are open.

When enabled later, the contracts are:

- EvoQuant-style integration supplies typed `PolicyTransition` candidates; it
  does not replace PIVOT or reproduce an evolution system.
- An LLM runs outside the event-level execution path and persists prompt,
  response, typed edit, compilation result, and strategy artifact.
- A learned F3 world adapter reads a local result file and always emits
  `ground_truth: false`; it is an alternative interventional proxy.
- No adapter may support a core claim until the controlled, finance, and
  strategic gates have registered evidence.

This file is the explicit P10 stop marker. Absence of adapter code is a gate
decision, not an accidental implementation gap.
