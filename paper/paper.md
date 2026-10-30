---
title: "IMPERANDI: A Python framework for building analysis-ready CT and MRI cohorts from heterogeneous DICOM data, with applications in abdominal imaging"
tags:
  - Python
  - medical imaging
  - computed tomography
  - CT
  - magnetic resonance imaging
  - MRI
  - DICOM
  - NIfTI
  - radiomics
  - segmentation
  - cohort building
  - longitudinal imaging
authors:
  - name: Diana Mandache
    affiliation: "1"
    corresponding: true
    orcid: "0000-0002-7160-4093"
  - name: Valérie Vilgrain
    affiliation: "2"
  - name: Stéphanie Allassonnière
    affiliation: "1"
affiliations:
  - name: "HeKA Unit, UMR 1346, Université Paris Cité, Inria, Inserm, Paris, France" 
    index: 1
    ror: "00my0j309" 
  - name: "Beaujon Hospital, Assistance publique - Hôpitaux de Paris (AP-HP), Paris, France"
    index: 2
    ror: "03jyzk483"
date: 1 October 2026
bibliography: paper.bib
---

# Summary

Imaging data collected in clinical care provides a substantial resource for retrospective research, but turning hospital exports into analysis-ready cohorts requires extensive curation; this process is currently fragmented across many manual and error-prone steps.

A single computed tomography (CT) or magnetic resonance imaging (MRI) examination can produce many image series[^1]: positioning scans, acquisitions at different times after contrast injection[^2], alternative reconstructions and derived images. Differences in protocols, metadata and naming conventions across scanners and institutions complicate their interpretation. Which acquisitions are relevant and how they should be processed also depends on the pathology and research question. Building comparable datasets across patients and visits therefore requires imaging expertise expressed as explicit, reproducible decisions.

[^1]: An image series is a group of related images from an examination; it may represent one or several volumes.
[^2]: The timing of a scan relative to contrast injection changes which tissues and blood vessels are most visible.

`IMPERANDI` (**IM**aging **PRE**processing **A**nd **N**ormalization for **D**iagnostic **I**nteroperability) is a Python framework that translates these decisions into a configurable workflow for CT and MRI cohort construction. It combines ingestion and processing stages with modality-specific curation that classifies acquisitions, ranks candidate volumes by their suitability within each examination and records the evidence behind selections. Selected volumes are converted to a compact format for organ and tumor segmentation, image registration and quantitative feature extraction.

The core workflow runs as independent command-line stages connected by cohort tables that link source images to derived outputs. Human-readable configuration files encode selection rules and processing strategies. Logs and error reports record outcomes and failures, while checkpoints allow interrupted batch jobs to resume. The architecture requires no database or continuously running web service, simplifying deployment on restricted computing clusters. Lightweight notebook and browser-based viewers support visual inspection of CT/MRI volumes, segmentation masks and cohort metadata. in such limited stack environments.

# State of the field

Growing imaging collections [@clark2013tcia; @li2024abdomenatlas] and advances in high-throughput AI-based methods [@Wasserthal2023] expand the potential for medical-image analysis at scale. Realizing that potential depends on making images acquired in different settings interpretable and usable across studies. Digital Imaging and Communications in Medicine (DICOM) provides a widely adopted standard for storing and exchanging clinical images and metadata. Research reuse requires a further layer: shared conventions for organizing acquisitions and describing their meaning. Neuroimaging has a head start in this field: the Neuroimaging Informatics Technology Initiative (NIfTI) format established a common representation for multidimensional image data and spatial-coordinate transforms [@cox2004nifti], while the Brain Imaging Data Structure (BIDS) defines how datasets are organized and described [@gorgolewski2016bids]. Together, these standards underpin a mature ecosystem of interoperable resources, including data repositories such as OpenNeuro and processing platforms such as Clinica [@routier2021clinica].

Efforts to extend shared conventions to other body regions are under way: [MIDS](https://github.com/BIMCV-CSUSP/MIDS) (BIDS Extension Proposal BEP025) proposes a structure for broader medical imaging data. Abdominal cohorts, however, often begin as institution-specific DICOM exports with multi-modal, multi-phase, multi-sequence acquisitions. Manufacturers can also encode acquisition details differently, including in private tags [@rorden2025dicom], while user-entered free-text descriptions introduce further variability. Preparing such exports for analysis remains a substantial step between available data and the application of new methods, especially for cohorts spanning CT and MRI.

Open-source platforms provide important foundations for medical-image analysis: the [`Insight Toolkit (ITK)`](https://github.com/SimpleITK/SimpleITK) offers reusable image-processing algorithms [@mccormick2014itk], while [`3D Slicer`](https://github.com/slicer/slicer) integrates visualization and interactive image analysis in an extensible application [@fedorov2012slicer]. More specialized software tools address individual parts of the workflow: [`pydicom`](https://github.com/pydicom/pydicom) reads DICOM objects, [`dicom2nifti`](https://github.com/icometrix/dicom2nifti) converts series to NIfTI volumes, [`TotalSegmentator`](https://github.com/wasserth/totalsegmentator) segments anatomy in CT and MRI [@Wasserthal2023; @dantonoli2025totalsegmentatormri] and [`PyRadiomics`](https://github.com/AIM-Harvard/pyradiomics) extracts quantitative features [@vanGriethuysen2017] as defined by the Image Biomarker Standardisation Initiative (IBSI) [@zwanenburg2020ibsi]. `IMPERANDI` contributes a cohort-level workflow that connects these components through ingestion, curation and processing, with explicit records of identifiers, selections and failures.

# Statement of need

Imaging researchers, data engineers and clinical collaborators need reliable patient-, visit- and volume-level records from hospital exports with inconsistent identifiers and metadata.

- **Selection criteria.** Cohort construction must bridge the knowledge gap between data scientists and medical experts by translating clinical expertise into explicit criteria for selecting CT phases and MRI sequences. Uncertain or unresolved selections must remain available for review.

- **Consistent processing.** Segmentation tasks, mask refinement, registration references and feature-extraction settings must also be specified consistently across the cohort. These choices determine the anatomical regions measured and the spatial correspondence needed to compare acquisitions.

- **Computing environment.** Clinical imaging data are usually held in restricted computing environments where only a minimal software stack is available.

# Software design

`IMPERANDI` organizes parsing, cleaning, conversion, image type curation, segmentation, registration and radiomics extraction as stages that exchange cohort tables and derived files. Shared orchestration provides stable identifiers and auditable outputs, while CT- and MRI-specific rules interpret their different acquisition protocols. A YAML manifest defines ordered operations and policies within these stages; optional user-defined Python hooks handle specific logic that cannot be expressed declaratively.

![Overview of the IMPERANDI cohort-building pipeline.\label{fig:pipeline}](figure.pdf){width="100%"}

- **Ingestion and cleaning.** During `ingestion`, the framework reads DICOM folders, DICOMDIR references and nested archives; derives identifiers from metadata, filepath structure or both; groups individual slices into volumes, including multi-volume series. It harmonizes date and time from multiple (often incomplete) metadata fields to establish visit and acquisition order. Manifest-defined `cleaning.steps` then apply ordered _keep_ or _discard_ filters at the instance or volume level. Rules can combine metadata values, text or regular-expression matches, numeric thresholds and explicit handling of missing values. They can exclude, for example, localizers, secondary images, non-target anatomy, unsuitable acquisition planes, implausible geometry or insufficient resolution.

- **Curation.** After DICOM instances have been grouped into volumes, modality-specific `curation` identifies CT contrast-enhancement phases and distinguishes T1-weighted, T2-weighted and diffusion MRI sequences. MRI rules also recognize acquisition techniques (e.g. Dixon), image variants (e.g. water-only, fat-only) and acquisition conditions (e.g. breath-hold), using acquisition order to help resolve contrast-enhancement phases. The manifest defines an ordered fallback strategy combining site-specific mapping (ontology), metadata rules and optional image-based phase prediction through a TotalSegmentator backend (available only for CT). Suitability scores rank candidate volumes, favoring appropriate acquisitions and image variants while penalizing localizers and low-value derived images. All curated candidates are retained and the preferred volume is exported for each CT phase or MRI sequence–phase combination, together with selection scores, phase provenance and unresolved cases for review.

- **Conversion and segmentation.** Selected DICOM volumes are `converted` to NIfTI. The manifest specifies separate CT and MRI `segmentation` tasks (with TotalSegmentator backend) and defines ordered post-processing transformations, including mask union or intersection, morphological operations (e.g. opening, closing), hole filling and largest-component selection.

- **Registration.** For groups of anatomically comparable scans, `registration` selects a reference, aligns organ masks through configurable linear and elastic stages using mask- or intensity-based objectives and can combine organ and tumor annotations across acquisitions. The manifest controls grouping criteria (e.g. by date), reference priorities (e.g. preferred modality), the order and conditions of registration stages and consensus methods (e.g. majority voting, STAPLE [@staple]). Outputs include masks in native or reference space, per-scan Dice scores and registration status.

- **Radiomics.** Similarly, modality-specific `radiomics` settings control quantitative feature extraction from organ, tumor or organ-minus-tumor regions.

The modular architecture supports end-to-end execution, independent use of stages and integration of external processing steps, allowing workflows to adapt to existing tools and evolving research needs.

The execution layer records results and failures per volume, allowing unaffected scans to continue when individual tasks fail. Checkpoint files are written atomically and resumption checks execution settings and input fingerprints before reusing completed results. Segmentation and visualization backends are packaged as optional dependencies, keeping ingestion and curation independent of heavier processing components. \autoref{fig:pipeline} provides an overview of the workflow. [Documentation](https://imperandi.readthedocs.io/en/latest/) describes installation and workflow configuration, while [interactive notebooks](https://github.com/dmandache/IMPERANDI/tree/main/examples/demos) demonstrate ingestion, curation and downstream processing on public TCGA-LIHC data.

# Research impact statement

`IMPERANDI` was developed within the RHU OPERANDI project to support research on treatment response prediction and clinical outcomes in hepatic oncology. Applied to imaging data hosted on the *Clinical Data Warehouse of Greater Paris University Hospitals* (EDS AP-HP), it addresses the challenge of transforming thousands of heterogeneous CT and MRI examinations from hundreds of patients into analysis-ready cohorts, demonstrating a unified workflow across both modalities within a restricted hospital computing environment.

+------------------------------------+----------------------------------------------------+--------------------------+
| Cohort selection measure           | CT portal-venous \                                 | MRI multiphasic \        |
|                                    | [@sarfati2023weakly]                               | [@gross2023liverhccseg]  |
|                                    +----------------------+-----------------------------+--------------------------+
|                                    | Metadata only        | Metadata + image            | Metadata only            |
+====================================+:====================:+:===========================:+:========================:+
| Patients in reference              | 49                   | 49                          | 17                       |
+------------------------------------+----------------------+-----------------------------+--------------------------+
| **Patients selected**              | **40**               | **65**                      | **36**                   |
+------------------------------------+----------------------+-----------------------------+--------------------------+
| $\enspace\hookrightarrow$          | 33                   | 49                          | 16                       |
| _matched to reference_             |                      |                             |                          |
+------------------------------------+----------------------+-----------------------------+--------------------------+
| $\enspace\hookrightarrow$          | 7                    | 16                          | 20                       |
| _outside reference_                |                      |                             |                          |
+------------------------------------+----------------------+-----------------------------+--------------------------+
| Patient-level recall               | 67.3%                | 100%                        | 94.1%                    |
+------------------------------------+----------------------+-----------------------------+--------------------------+
| Patient–date recall                | 53.1%                | 89.8%                       | 76.5%                    |
+------------------------------------+----------------------+-----------------------------+--------------------------+
| Patient–date–phase recall          | 53.1%                | 89.8%                       | 25.0%                    |
+------------------------------------+----------------------+-----------------------------+--------------------------+

Table: Automated cohort recovery against two published TCGA-LIHC reference cohorts. \label{tab:benchmark}

The framework was [tested](https://github.com/dmandache/IMPERANDI/tree/main/tests/slow) on the public 3D-IRCADb-01 liver dataset [@soler2010ircad] and the more heterogeneous TCGA-LIHC imaging collection [@erickson2016tcgalihc; @clark2013tcia]. On TCGA-LIHC, we assessed automated selection against two published subsets: portal-venous CT scans from the 49 patient cohort described by [@sarfati2023weakly] and multiphasic MRI scans from 17 patients described by [@gross2023liverhccseg]. For CT, a declarative manifest selected the portal-venous acquisition from each patient's earliest eligible examination. Metadata rules recovered most reference patients and adding image-based phase prediction recovered all of them without manual series selection. For the more challenging MRI cohort, metadata-based curation recovered most reference patients but fewer of the expected examination–phase combinations (\autoref{tab:benchmark}). These comparisons assess out-of-the-box cohort recovery rather than exact replication of the published studies. The original studies included expert image review and study-specific clinical or annotation eligibility criteria that were not reproduced here. Moreover, the exact imaging series identifiers were not reported, preventing direct verification of series-level agreement. The [benchmark materials](https://github.com/dmandache/IMPERANDI/tree/main/examples/benchmarks) provide the selection manifests, reference cohort tables and scripts used to reproduce these comparisons.

These results highlight the potential of automated cohort selection while exposing challenges posed by inconsistent acquisition nomenclature, particularly in MRI. By preserving selection evidence and intermediate outputs, `IMPERANDI` allows experts to review selection decisions and refine the explicit, reproducible rules that guide them. Selected volumes can then undergo conversion, segmentation and registration within the same configurable workflow. These downstream stages require separate, task-specific evaluation, which is beyond the scope of this paper. Together, these capabilities make domain expertise reusable across cohorts, anatomical regions and research questions.

# AI usage disclosure

Generative AI tools (ChatGPT and Codex, OpenAI) were used to assist with manuscript drafting and language editing, code refactoring, unit-test design and documentation development. All AI-assisted contributions were critically reviewed, revised as needed and verified against relevant sources and implementation evidence by the authors, who take full responsibility for the accuracy and integrity of the final manuscript and software.

# Acknowledgements

This work was developed within RHU OPERANDI (ANR-21-RHUS-0012), with additional support from PR[AI]RIE-PSAI (ANR-23-IACL-0008), both partly funded by the French National Research Agency (ANR) under the _France 2030_ program. The public-dataset tests used data generated by the TCGA Research Network and made available through The Cancer Imaging Archive.

# References
