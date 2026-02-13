CODES Benchmark
===============

**CODES** is an end-to-end benchmark for coupled ODE surrogates. It ships with curated datasets, reproducible tuning/training/evaluation scripts, and a comprehensive API so you can extend the stack with new models.

.. note::

   A typical workflow follows **Tune → Train → Evaluate**. Use the links below to jump straight into the relevant guide, or follow the quickstart links in *Getting Started* to run a toy experiment on your machine.

* **Guided Quickstart** — :doc:`getting-started` walks you through cloning, installing, and running a smoke-test benchmark.
* **Benchmark Workflow** — :doc:`guides/running-benchmarks/index` explains how tuning feeds training and how evaluations consolidate metrics.
* **Extend the Stack** — :doc:`guides/extending-benchmark` shows how to add datasets or surrogates without rewriting orchestration glue.
* **API Reference** — :doc:`api-reference` explains how the generated package docs are organized and links to each module.

Looking for a bird’s-eye view first? Start with the **User Guide**. Already configuring experiments or integrating your own model? Skip ahead to the **API Reference**. Either way, the sidebar mirrors the sections below so you are one click away from the next step.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Getting Started <getting-started>
   Running Benchmarks <guides/running-benchmarks/index>
   Extending The Benchmark <guides/extending-benchmark>
   Configuration Reference <reference/configuration>
   Dataset Catalog <reference/datasets>
   Tutorials <tutorials/index>

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   API Reference Overview <api-reference>
   codes.benchmark package <codes.benchmark>
   codes.surrogates package <codes.surrogates>
   codes.train package <codes.train>
   codes.tune package <codes.tune>
   codes.utils package <codes.utils>
