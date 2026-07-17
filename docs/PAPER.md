# M2VG: Expression Satisfiability Modeling for Generalized Visual Grounding

## Abstract

Generalized visual grounding extends conventional visual grounding to
one-to-zero, one-to-one, and one-to-many scenarios. It requires models to
localize referred targets and determine whether a language expression is
satisfiable by the image content. However, existing methods mainly rely on
target matching between text and image regions, making them vulnerable to the
target-presence assumption. They may predict incorrect regions when local
semantic cues match the expression but the global language constraints are not
satisfied.

To address this problem, we propose M2VG, an expression satisfiability modeling
framework for generalized visual grounding. M2VG reformulates the task as
language constraint verification, where an expression is satisfiable only when
its category, attribute, spatial, and relational constraints are fully supported
by object instances in the image. Specifically, we introduce dynamic focus
negative sample mining to construct reliable unsatisfiable image-expression
pairs through target removal and semantic mismatch. We further design an
adaptive word-aware residual modulation module and an auxiliary existence branch
to enhance image-conditioned token representations and predict expression-level
existence probability. Experiments show that M2VG achieves competitive
performance on generalized referring expression comprehension, generalized
referring expression segmentation, and conventional referring expression
localization. It improves target-absence recognition while maintaining strong
localization and segmentation performance.

**Keywords:** visual grounding; generalized visual grounding; referring
expression comprehension; referring expression segmentation; expression
satisfiability.
