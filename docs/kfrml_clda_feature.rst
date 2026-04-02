.. _kfrml_clda_feature:

KFRML CLDA feature
==================

Overview
--------
``CLDA_KFRML_IntendedVelocity`` is a task feature mixin that enables online
closed-loop decoder adaptation (CLDA) for Kalman-style decoders. It combines:

- a Learner (``OFCLearnerRotateIntendedVelocity``) that estimates intended
  kinematics from task goals, and
- an Updater (``KFRML``) that recursively updates decoder parameters from
  batches of intended kinematics and neural observations.

This feature is defined in ``features.clda_features`` and is intended to be
used via multiple inheritance in BMI task classes that already provide a
``decoder`` and BMILoop plumbing.

How it works
------------
1. ``create_learner`` builds a feedback map ``F_dict`` where ``target`` uses
   position-error feedback and non-target states use zero feedback.
2. The learner converts target direction into an intended velocity vector with
   magnitude set by ``clda_intended_speed``.
3. Samples are accumulated into batches of size
   ``int(clda_batch_time / decoder.binlen)``.
4. ``create_updater`` initializes ``KFRML`` with
   ``(batch_time=clda_batch_time, half_life=clda_update_half_life)``.
5. Each update combines old sufficient statistics with new batch data using
   exponential forgetting and updates decoder observation parameters.

Configurable traits
-------------------
- ``clda_batch_time`` (seconds): update cadence for learner/updater batches.
- ``clda_update_half_life`` (seconds): forgetting timescale in KFRML.
- ``clda_intended_speed`` (decoder velocity units): nominal intended velocity
  magnitude used by the learner.

Operational notes
-----------------
- CLDA is off by default and controlled by task ``learn_flag`` logic.
- Intended-kinematics samples are only informative during behaviorally relevant
  states (typically ``target``).
- NaN/Inf samples are filtered before KFRML sufficient-stat updates.

Practical tuning
----------------
- Increase ``clda_intended_speed`` if adaptation is too weak.
- Decrease ``clda_intended_speed`` if updates are too aggressive/noisy.
- Reduce ``clda_batch_time`` for more frequent, smaller updates.
- Increase ``clda_batch_time`` for smoother, less variable updates.
