# **Project Notes**

- **Project Note 1** - Identify the source of variability before selecting the adaptive mechanism.

- **Project Rule 1** - `Start from variability`, not from a technique. Identify what changes at runtime and why adapting to it should improve the system.

---

- **Project Note 2** - The architecture of the controller.
```text
SENSE:
What runtime information do we observe?

ESTIMATE:
How do we convert it into difficulty/resource information?

CHOOSE:
What policy selects the model configuration?

EXECUTE:
What exactly changes in the model?

UPDATE:
What runtime feedback do we record/use?
```

- **Project Rule 2** - `Define the complete adaptive loop`, not just the adaptive model.

---

- **Project Note 3** Create a small table like this.
```text
| Design axis     | Project answer |
| --------------- | -------------- |
| What varies?    | ?              |
| When decide?    | ?              |
| What signal?    | ?              |
| What objective? | ?              |
```
Do not start implementation until this table has convincing answers.
Example: 
```text
| Axis            | Example                                     |
| --------------- | ------------------------------------------- |
| What varies?    | ViT depth                                   |
| When decide?    | once per image                              |
| What signal?    | confidence + available latency budget       |
| What objective? | minimize latency while maintaining accuracy |
```
- **Project Rule 3** - Fill in the four-axis design table before committing to the architecture.

---

- **Project Note 4**  First explicit evaluation rule

We should not report only:
```text
accuracy = 84%
``` 
or only:
```text
latency = 18 ms
```
We should report the trade-off between the two.
```text
For example:

configuration A → 78% accuracy, 8 ms
configuration B → 82% accuracy, 15 ms
configuration C → 85% accuracy, 28 ms
```
Then plot something like:
```text
accuracy vs latency
accuracy vs FLOPs
accuracy vs energy
```
depending on what the project actually optimizes.

- **Project Rule 4** - Evaluate the adaptive system as a frontier of operating points, not as a single model result.

---

- **Project Note 5**  Edge-specific adaptivity

A common mistake would be to think:
```text
“Adaptive AI means modifying the neural network.”
```
But this is too narrow.

For an edge system, adaptation can happen at the sensor/input pipeline too.

- **Project Rule 5** - Consider whether the cheapest adaptation happens before the model.

If our eventual project uses imaging or sensor streams, we should compare things like:
```text
adapt model depth
vs
adapt input resolution
vs
adapt sensing frequency
```
Sometimes avoiding unnecessary data acquisition is much more valuable than making the model slightly cheaper.

---

- **Project Note 6** When we use a ViT, then token count itself is an adaptive dimension.

That means our project design space is broader than:
```text
small ViT
vs
large ViT
```
It could instead be:
```text
same ViT
+
dynamic token budget
```
or potentially:
```text
adaptive depth + adaptive tokens
```
- **Project Rule 6** - If the model is a transformer, include token budget as a candidate adaptive control knob.

---

- **Project Note 7** Reuse previous course material

This is useful because pruning and quantization are not separate islands from adaptivity.

They can become components of an adaptive system.

For example, a project can use:
```text
resource monitor
      ↓
precision policy
      ↓
INT8 / FP16 configuration
```

- **Project Rule 7** - Compression methods can become adaptive control dimensions rather than only offline preprocessing steps.

This is probably why the course teaches pruning and quantization before adaptivity.

---

- **Project Rule 8** - Prefer an adaptive mechanism that matches the actual source of redundancy.

If the redundancy is:
```text
easy vs hard samples → early exit
resource variation → width / precision
uninformative image regions → token pruning
slowly changing stream → caching
redundant blocks → layer skipping
```

---

- **Project Note 9** Granularity is a design decision

When we define our project, we should explicitly answer:
```text
At what granularity do we adapt?
```
For example:
```text
once per image
once per transformer block
once per token
```
A project does not become more sophisticated simply because the controller operates more frequently.

- **Project Rule 9** - Use the coarsest decision granularity that captures the useful variability.

That should help keep the system realistic.

---

- **Project Note 10**  Decide the operating mode

This is a design choice we should make early.

For example, if our future scenario says:
```text
inference must finish within a known latency budget
```
then `budgeted inference` probably matches naturally. If the system could be interrupted or timing varies unpredictably:
```text
return the best answer available so far
```
then anytime inference makes more sense.

- **Project Rule 10** - State whether the system is anytime, budgeted, or intentionally combines both.

---

- **Project Note 11**  Separate mechanism from controller

Suppose our model supports: depth = 6, 8, 10, 12

That is the model mechanism. But something still needs to decide:
```text
when should depth 6 be used?
when should depth 12 be used?
```
That belongs to the runtime controller. And the controller may itself be trying to satisfy:
```text
latency < 30 ms
```
which comes from the application policy. So we should keep these concepts separate in the project:
```text
Application objective
       ↓
Runtime controller
       ↓
Adaptive model mechanism
```

- **Project Rule 11** - Separate the adaptive mechanism from the policy/controller that decides when to use it.

This will make the project architecture much cleaner.

---

- **Project Note 12** Signal quality may matter more than mechanism sophistication

Suppose we design an amazing early-exit network. But our signal deciding whether to exit is unreliable. Then the whole system fails. So the controller's signal is not a minor implementation detail.

For our project, we should eventually test:
```text
Does the selected signal actually predict
when cheaper computation is sufficient?
```
That's a research question in itself.

- **Project Rule 12** - Validate the decision signal independently; an adaptive mechanism is only useful if the controller can reliably decide when to use it.

---

- **Project Note 13**  Don't use confidence blindly

If we use something like: `confidence threshold = 0.90` in our project, we cannot simply invent that value. We should determine it using validation data and examine whether confidence is reliable. There are some concrete calibration tools.

- **Project Rule 13** - A confidence threshold must be validated/calibrated; never assume softmax confidence is automatically a reliable difficulty score.

This could become especially important if the project has a safety-sensitive scenario.

---

- **Project Note 14**  We need system metrics, not just ML metrics

A weak project result would be:
```text
Depth 6: 80% accuracy
Depth 12: 84% accuracy
```
A stronger adaptive-edge result is something like:
```text
Depth | Accuracy | Latency | Energy | Memory
6       ...        ...       ...      ...
8       ...        ...       ...      ...
10      ...        ...       ...      ...
12      ...        ...       ...      ...
```
plus eventually switching/controller behavior. And later, if our scenario makes latency deadlines important, we might care more about: `deadline miss rate` than simply average latency.

- **Project Rule 14** - Evaluate the system metric that motivated adaptivity in the first place, alongside accuracy.

If we say the project adapts to latency but only report FLOPs, then the evaluation does not actually prove the claim.

---

- **Our project framework is becoming quite concrete**

```text
SOURCE OF VARIABILITY:
?

ADAPTIVE GRANULARITY:
input / layer / token

OPERATING MODE:
anytime / budgeted

MODEL CONTROL KNOB:
depth / width / tokens / precision / ...

POLICY SIGNAL:
confidence / resource state / features / history

APPLICATION OBJECTIVE:
latency / energy / accuracy / ...

CONTROLLER:
how does the signal map to a configuration?

EVALUATION:
accuracy + actual system metric + switching cost

CALIBRATION:
how do we know the policy signal is trustworthy?
```

---

- **Project Note 15** Training objective must include adaptivity

If our project supports several modes, we cannot train only the full model and assume the smaller modes will work.

- **Project Rule 15** - Every operating mode that may be used at runtime must be explicitly trained or supported by the training objective.

---

- **Project Note 16**  Hardware validation is mandatory

This one is very strong for the course project. We should not claim: `Our method is faster because FLOPs decreased`, unless we also measure actual runtime.

- ***Project Rule 16** - Treat theoretical efficiency and measured hardware efficiency as different quantities.

We should eventually measure on the target environment, not only calculate FLOPs.

---

- **Project Note 17** We must justify why adaptivity fits the scenario

Before implementing anything, we should be able to answer:
```text
Are there actually many easy cases?
Can we estimate difficulty?
Does runtime variability exist?
Will the hardware benefit?
```
- **Project Rule 17** - Demonstrate that the scenario contains exploitable variability; otherwise adaptivity may add complexity without producing value.

---

- **Project Note 18** Evaluation checklist

```text
1. strong static baseline?
2. controller overhead included?
3. real hardware measured?
4. tail/worst-case behavior examined?
```

- **Project Rule 18** - Compare against a strong static baseline and include controller/switching overhead in the measurements.

---

- **Project Note 19** Full model can become the teacher

This is particularly relevant if we end up with one elastic model with multiple configurations. We may not need an entirely separate external teacher. Conceptually, the full configuration can guide the cheaper ones:
```text
largest/full configuration
        ↓ teacher
medium configuration
small configuration
early exits
```
- **Project Rule 19** - If small adaptive modes lose too much accuracy, consider using the full model/configuration as a teacher during training.

---

- **Project Note 20** Early exit only makes sense if early representations are useful

This is a useful constraint. Adding exit heads mechanically does not guarantee a good adaptive model.

- **Project Rule 20** - If using early exit, verify that intermediate representations can support useful predictions; exits must be trained, not merely attached.

---

- **Project Note 21** Count exit-head overhead

This is directly measurable. If we ever implement early exit, we should not compare: `cost until exit point` against full inference while ignoring the cost of: `adapter + classifier + policy`

- **Project Rule 21** - Include auxiliary exit-head and policy cost when calculating early-exit savings.

---

- **Project Note 22** Early-exit training creates a multi-objective problem

If we use early exit, we should track at least:
```text
exit-1 accuracy
exit-2 accuracy
final accuracy
```
not only final-model accuracy.

- **Project Rule 22** - Evaluate each exit separately during training; improving shallow exits must not silently degrade the final model.

---

- **Project Note 23** Threshold becomes part of the trade-off curve

This is quite important. An early-exit model does not really have just one operating point. Suppose: `threshold = 0.70` Many samples exit early: `cheap, potentially less accurate`. But: `threshold = 0.99` very few exit: `expensive, potentially safer`. So the threshold itself can generate multiple accuracy/cost operating points.

- **Project Rule 23** - Treat exit threshold as a policy parameter and evaluate its accuracy–cost curve rather than selecting one arbitrary threshold.

---

- **Project Note 24** Distinguish single-sample latency from throughput

This matters a lot in experiments. Suppose early exit improves: `latency for one sample`. That does not automatically prove improvement in: `samples/second under batching`. These are different deployment objectives.

- **Project Rule 24** - Decide whether the project optimizes per-sample latency or batched throughput, and benchmark accordingly.

For an edge scenario with one sensor stream, per-input latency may be more relevant than server throughput—but that decision should come from the scenario, not from convenience.

---

`One broader question:`

```text
If we consider early exit: does the dataset actually contain enough “easy” inputs that many samples can leave early without significant accuracy loss?
```

---

- **Project Note 25** Discrete modes may be better than arbitrary widths

This is useful for our project. It may be tempting to say: Let's support every possible width.`. But supported widths should also match the hardware.

- **Project Rule 25** - Choose a small set of useful, hardware-friendly operating widths rather than assuming every mathematically possible width is worth supporting.

That will also reduce controller complexity.

---

- **Project Note 26** Shared weights can still need width-specific state

This is an implementation issue worth remembering:

- **Project Rule 26** - If multiple widths share weights, check whether normalization/statistical state must remain width-specific.

Otherwise a “shared model” may perform badly for reasons unrelated to the adaptive idea itself.

---

- **Project Note 27** Training cost is part of method feasibility

This is not runtime cost; it is design/training cost.

- **Project Rule 27** - When evaluating an adaptive architecture, consider whether training all supported configurations is computationally realistic.

Especially if we combine width, depth, resolution, etc., the number of subnets can explode.

---

- **Project Note 28** Don't open too many control dimensions at once

This is where project scope could easily explode. If we vary: `width × depth × resolution × precision` then the number of configurations becomes huge. So even though the design space allows it:

- **Project Rule 28** - Start with one dominant adaptive dimension; add additional dimensions only if the first one cannot explain the scenario adequately.

That is more of a project-design inference from the slide's expanding design space, but it is very practical.

---

- **Project Note 29** Width adaptation can react to system state or sample difficulty

This gives us an important design choice:
```text
resource-aware width selection
vs
input-aware width selection
```
or potentially both.

- **Project Rule 29** - If using width adaptation, state whether width is chosen from resource state, input difficulty, or a combination; these are different controller problems.

---

- **Project Note 30** Evaluate subnet quality separately from controller quality

This is a very important experimental design rule. Before testing an adaptive controller, we should know:
```text
accuracy of 0.25×
accuracy of 0.5×
accuracy of 0.75×
accuracy of 1.0×
```
Then, separately, test whether the controller selects them intelligently. Otherwise, if the adaptive system performs poorly, we won't know whether: `subnet training failed`, or `selection failed`.

- **Project Rule 30** - Benchmark each operating mode independently before evaluating the adaptive policy.

This is one of the strongest project notes from this block.

---

- **Project Note 31** One decision rule.

- **Project Rule 31** - Match the mechanism to the dominant variability: input difficulty naturally suggests early exit; changing resource budgets naturally suggest width adaptation.

---

- **Project Note 32** This is directly relevant

This gives us a possible architecture philosophy for the project:
```text
Do not necessarily train several independent ViTs. Try to expose several useful capacities inside one shared model.
```
That directly addresses the storage and switching problems.

---

- **Project Note 33** Switching mechanism must be concrete

If we say our project has: `small / medium / large ViT`, we eventually need to answer: `What physically changes between those modes?` For example:
```text
number of active attention heads?
hidden dimensions?
blocks?
tokens?
some combination?
```
A label like “small mode” is not enough.

- **Project Rule 33** - Define the exact active subgraph difference between operating modes.

---

- **Project Note 34** This is almost a complete project architecture

A possible end-to-end research workflow:
```text
pretrained backbone
        ↓
construct several capacities
        ↓
profile each capacity
        ↓
runtime resource monitoring
        ↓
controller chooses operating point
```
For our transformer-oriented project, this is much more interesting than merely benchmarking several independently trained model sizes.

- **Project Rule 34** - Separate an offline elastic-model construction phase from an online runtime-selection phase.

This distinction will make the methodology much cleaner.

---

- **Project Note 35** Prior pruning knowledge now becomes useful

This is exactly why the earlier pruning lab is relevant to adaptivity. Pruning is not necessarily the final deployment solution. It can be used as a model-construction tool:
```text
pruning
↓
discover/create smaller structural modes
↓
embed those modes into an elastic system
```

- **Project Rule 35** - Consider compression as a way to construct elastic operating modes, not only as the final endpoint.

---

- **Project Note 36** Switching frequency may be as important as switching mechanism

This is very relevant. Imagine our controller detects resource changes every frame:
```text
frame 1 → large
frame 2 → medium
frame 3 → large
frame 4 → medium
```
Even if each configuration is efficient, switching overhead could destroy the benefit.Instead, we might need:
```text
resource state changes
       ↓
require persistent change / threshold
       ↓
switch mode
       ↓
stay there for a period
```
That can involve concepts such as hysteresis or minimum dwell time, although those specific techniques are outside what this slide teaches.

- **Project Rule 36** - Measure mode-switch cost and choose adaptation frequency so that switching overhead does not erase inference savings.

This could itself become an interesting experimental dimension.

---

- **Project Note 37** This should be one of our core success criteria

For our eventual experiment, don't only ask: `Can we create several model modes?` Ask: `Are several of them actually useful?` That suggests a very nice evaluation:
```text
Mode 1  ─┐
Mode 2   │
Mode 3   ├→ plot accuracy vs real resource cost
Mode 4  ─┘
```
Then inspect whether we actually obtain graceful degradation / a useful frontier.

- **Project Rule 37** - Success means a useful family of operating points, not simply the existence of multiple modes.

---

**The entire Elastoformer story in one diagram**

```text

              PRETRAINED BACKBONE
                     │
                     ▼
             STRUCTURED PRUNING
                     │
                     ▼
             SMALLEST CORE MODEL
                     │
          ┌──────────┴──────────┐
          │ progressive regrowth│
          ▼                     ▼
      DN small → DN medium → DN large
          └─ shared/nested param.──┘
                     │
                     ▼
                DEPLOYMENT
                     │
        runtime resource monitoring
                     │
                     ▼
             choose appropriate DN

GOOD RESULT:
several useful accuracy/resource points

BAD RESULT:
modes too similar
or
small modes become unusable

SYSTEM REQUIREMENT:
switching cost must be worth it
```

---

- **Project Note 38** Transformer structure itself can provide the elastic dimension

This is highly relevant if our model is a ViT. We don't necessarily need to treat:
```text
small ViT
medium ViT
large ViT
```
as three separate architectures. We can investigate whether the transformer's internal structure can provide:
```text
shared base
   +
progressively activated attention/embedding capacity
```

- **Project Rule 38** - For a transformer project, investigate whether elasticity can be structured inside attention-head/embedding subspaces rather than using independent model variants.

---

- **Project Note 39** Self-distillation is natural for an elastic model

If we construct an elastic ViT, the full configuration could act as the teacher:
```text
full ViT
   ↓
medium
   ↓
small
```
That avoids requiring a completely separate teacher architecture.

- **Project Rule 39** - Consider full-to-small self-distillation when training nested transformer modes.

---

- **Project Note 40** Nested modes are much more interesting than independent modes

This is a major architecture principle. If we eventually design 3–4 operating points, we should ask: Can each larger configuration be an extension of the smaller one? That potentially improves:
```text
storage,
consistency,
switching,
knowledge transfer.
```

- **Project Rule 40** - Prefer nested operating modes when possible: small ⊂ medium ⊂ large.

---

- **Project Note 41** Switching architecture can be a research contribution itself

We've been talking a lot about selecting the best model configuration. But this slide reveals another dimension: `How do we physically transition between configurations?` Two projects might have identical operating modes but very different switching cost.

- **Project Rule 41** - Treat mode transition cost as a first-class architectural problem, not just an implementation detail.

For an actual edge deployment, that can be just as important as reducing FLOPs.

---

- **Project Note 42** Controller should decide whether to switch, not only where to switch

This is subtle but very important.

A naïve controller does:
```text
budget changed
↓
change model
```
A better controller asks:
```text
budget changed
↓
would switching now produce enough benefit?
          ↓
        yes/no
```
- **Project Rule 42** - The controller needs a “stay vs switch” decision in addition to choosing the target mode.

This is probably one of the strongest ideas we can take into our project.

---

### **The SHARD picture**

```text
              ONE TRANSFORMER

        shared smallest/base subspace
                    │
             + delta / extra
                    ▼
               medium mode
                    │
             + delta / extra
                    ▼
                large mode

small ⊂ medium ⊂ large

TRAINING:
largest subnet can teach smaller ones

SWITCHING:
base remains resident
only deltas move in/out

CONTROLLER:
resources change
      ↓
is switching worthwhile?
      ↓
stay / delta-up / delta-down
```

---

### **The big conceptual evolution across it is remarkably coherent**

```text
STATIC MODEL
same compute always
        ↓
ADAPTIVE INFERENCE
compute responds to input/system
        ↓
EARLY EXIT
adapt depth
        ↓
SLIMMABLE
adapt width
        ↓
ELASTOFORMER
shared family of elastic operating modes
        ↓
SHARD
nested transformer subspaces
        ↓
DELTA SWITCHING
make changing modes cheap enough to use in reality
```

---
---

## LECTURE 5 — PROJECT EXTRACTION

DO NOW
- Keep F002a as a fair backbone comparison:
  MobileNetV2 vs ResNet-18 vs DeiT-Tiny.
- Do not introduce KD, LoRA, adapters or NAS during backbone screening.
- Record task performance + actual latency + memory/footprint + FLOPs/MACs.
- Use the same data/split/preprocessing and comparable training budget.
- Preserve checkpoints, configs, split manifests, seeds and per-sample logits.
- Do not select a teacher later from aggregate accuracy alone; preserve enough
  outputs to evaluate its reliability on OpenPOCUS.

KEEP FOR LATER
- Knowledge distillation to recover accuracy in smaller/compressed modes.
- Self-distillation: full model/final exit can teach smaller modes/early exits.
- Distillation + pruning + quantization can be composed.
- LoRA/adapters are useful for cheap domain/site adaptation, NOT inference
  compression.
- Unlabeled-data KD is possible after a reliable teacher exists.

RESEARCH ONLY
- Hardware-aware NAS: unnecessary for the current backbone screen; small-data
  search can overfit.
- Online/test-time adaptation and continual learning.
- Federated learning.
- Split inference / cloud coordination.
- Multi-teacher/cross-modal and other sophisticated KD extensions.