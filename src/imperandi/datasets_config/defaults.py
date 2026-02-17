DEFAULT_VOLUME_LOWERBOUND = 30.0
DEFAULT_VOLUME_UPPERBOUND = 500.0

DEFAULT_MAX_PIXEL_SPACING_MM = 1.25
DEFAULT_MAX_SLICE_THICKNESS_MM = 3.0

DEFAULT_DICOM_TAGS = [
    # ─────────────────────────
    # Identifiers & paths
    # ─────────────────────────
    "PatientID",
    "PatientName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    # ─────────────────────────
    # Modality & SOP
    # ─────────────────────────
    "Modality",
    "ModalitiesInStudy",
    "SOPClassUID",
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    # ─────────────────────────
    # Study-level metadata
    # ─────────────────────────
    "StudyDate",
    "StudyTime",
    "StudyDescription",
    "StudyID",
    "AccessionNumber",
    "ReferringPhysicianName",
    # ─────────────────────────
    # Series-level metadata
    # ─────────────────────────
    "SeriesDate",
    "SeriesTime",
    "SeriesDescription",
    "SeriesNumber",
    "ProtocolName",
    "BodyPartExamined",
    "Laterality",
    # ─────────────────────────
    # Instance-level metadata
    # ─────────────────────────
    "InstanceNumber",
    "AcquisitionNumber",
    "InstanceCreationDate",
    "InstanceCreationTime",
    "ContentDate",
    "ContentTime",
    # ─────────────────────────
    # Image geometry
    # ─────────────────────────
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceLocation",
    "FrameOfReferenceUID",
    # ─────────────────────────
    # Image type & acquisition
    # ─────────────────────────
    "ImageType",
    "AcquisitionDate",
    "AcquisitionTime",
    "AcquisitionDateTime",
    "AcquisitionMatrix",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "RepetitionTime",
    "EchoTime",
    "EchoNumbers",
    "FlipAngle",
    # ─────────────────────────
    # CT-specific (very common)
    # ─────────────────────────
    "KVP",
    "ExposureTime",
    "XRayTubeCurrent",
    "Exposure",
    "ExposureModulationType",
    "ConvolutionKernel",
    "ReconstructionDiameter",
    # ─────────────────────────
    # Pixel data interpretation
    # ─────────────────────────
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "RescaleIntercept",
    "RescaleSlope",
    "RescaleType",
    # ─────────────────────────
    # Patient info (non-sensitive subset)
    # ─────────────────────────
    "PatientSex",
    "PatientAge",
    "PatientBirthDate",
    # ─────────────────────────
    # Misc / QC helpers
    # ─────────────────────────
    "NumberOfFrames",
    "PositionReferenceIndicator",
    "BurnedInAnnotation",
]