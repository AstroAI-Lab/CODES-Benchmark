CODES Benchmark
===============

**CODES** is an end-to-end benchmark for coupled ODE surrogates. It ships with curated datasets, reproducible tuning/training/evaluation scripts, and a comprehensive API so you can extend the stack with new models.

:doc:`View benchmark results <benchmarks/index>` to compare accuracy, inference
time, and uncertainty across the four A&A datasets. To run a benchmark yourself,
start with :doc:`getting-started`.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   Getting Started <getting-started>
   Running Benchmarks <guides/running-benchmarks/index>
   Extending The Benchmark <guides/extending-benchmark>
   Configuration Reference <reference/configuration>
   Dataset Catalog <reference/datasets>
   Benchmark Results <benchmarks/index>
   Tutorials <tutorials/index>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   API Reference Overview <api-reference>
   codes.benchmark package <codes.benchmark>
   codes.finetune package <codes.finetune>
   codes.surrogates package <codes.surrogates>
   codes.train package <codes.train>
   codes.tune package <codes.tune>
   codes.utils package <codes.utils>
