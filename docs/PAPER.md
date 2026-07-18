# M2VG: Expression Satisfiability Modeling for Generalized Visual Grounding

## Abstract

Generalized visual grounding extends conventional visual grounding to
one-to-zero, one-to-one, and one-to-many scenarios. Beyond localizing referred
instances, a model must determine whether a referring expression is supported by
the image. Existing approaches primarily optimize text-region matching and can
therefore return plausible but incorrect regions when a local category cue
matches while attribute, spatial, or relational conditions are violated.

We propose M2VG, an expression satisfiability modeling framework for generalized
visual grounding. M2VG treats a referring expression as a set of visual
conditions and learns to reject predictions when the image does not provide
sufficient joint evidence for these conditions. To construct informative
no-target supervision, Dynamic Focus Negative Sample Mining generates reliable
unsatisfiable image-expression pairs through target removal and semantic
mismatch under reliability constraints. We further introduce Adaptive
Word-aware Residual Modulation to refine token representations with image
context before query generation, and an auxiliary existence branch that
aggregates decoded instance-query evidence to estimate expression-level
existence. Experiments on generalized referring expression comprehension,
generalized referring expression segmentation, and conventional referring
expression comprehension show that M2VG improves no-target recognition while
maintaining competitive localization and segmentation performance.

**Keywords:** visual grounding; generalized visual grounding; referring
expression comprehension; referring expression segmentation; expression
satisfiability.
