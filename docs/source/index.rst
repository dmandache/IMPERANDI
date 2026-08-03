Welcome to IMPERANDI’s documentation!
=====================================

.. figure:: https://raw.githubusercontent.com/dmandache/IMPERANDI/main/static/imperandi-logo.png
   :alt: Viewer interface example
   :width: 700px
   :align: center

IMPERANDI [IMaging PREprocessing And Normalization for Diagnostic
Interoperability] is a Python framework and command-line interface for
turning heterogeneous CT and MRI DICOM exports into traceable, analysis-ready
cohorts.

One typed project YAML controls DICOM discovery, identity resolution,
modality-aware curation, conversion, optional contrast-phase prediction,
segmentation, registration, radiomics, and publication.

.. note::

   IMPERANDI is research software. It is not a certified medical device and
   its outputs require appropriate validation before clinical use.

Start here
----------

* :doc:`installation` — install the base package and optional feature sets.
* :doc:`quickstart` — run a first end-to-end cohort.
* :doc:`architecture` — understand modules, evidence, routing, and tests.
* :doc:`workflow` — understand stage boundaries, resuming, and data flow.
* :doc:`cli` — find command and option details.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   architecture
   workflow
   cli
   outputs
   configuration
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Python API

   api/imperandi
   api/workflow
   api/ingest
   api/process
   api/extract
   api/qc

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
