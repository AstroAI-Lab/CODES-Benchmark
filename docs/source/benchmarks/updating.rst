Updating the benchmark
======================

When a pull request changes or adds published benchmark results, include the
updated results and figures in the same pull request. Run the affected
comparisons from a fixed source commit. If the commit is on a branch that will be
squash-merged, give it a permanent benchmark tag so readers can retrieve it later.

Each snapshot should use a common scientific protocol and include
the dataset files and split provenance,
model configurations, code revision, seeds, and timing method. Keep hardware
and batching fixed within each group whose timings are ranked together. Keep
cluster accounts, local paths, and scheduler scripts outside the results bundle.

Add the validated, sanitized results bundle as
``benchmarks/snapshots/<snapshot-id>/results.json``. Add its display metadata
and the bundle's SHA-256 digest to ``benchmarks/snapshots/index.json``. Point
the catalog's ``current`` entry to the new snapshot when it is ready to
become the reference. Keep existing snapshot IDs and bundles available for
citations. Use a new ID if published values need a correction.

To include the error distribution and recall plots for a dataset, store
``plot-data.json``, ``error-distribution.svg``, and
``catastrophic-error-recall.svg`` together in the snapshot's
``diagnostics/<dataset>`` directory. Add their SHA-256 digests under
``diagnostics.<dataset>`` in the catalog entry. Generate the plots from
the same evaluation split, surrogate checkpoints, and seeds as the tables.
The compact plot data stores the per-seed curves and cutoff details.
Render the two SVGs with
``uv run python benchmarks/render_diagnostics.py path/to/plot-data.json``.

Build the docs with the normal project workflow:

.. code-block:: bash

   uv sync --locked
   uv run --no-sync sphinx-build -b html docs/source docs/_build/html
   uv run --no-sync python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html

Open `Benchmark Results on localhost <http://127.0.0.1:8000/benchmarks/index.html>`_.
The Sphinx build validates each bundle and generates the tables and CSV
downloads. The docs build performs no training or benchmarking.
