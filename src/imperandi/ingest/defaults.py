"""Framework defaults for DICOM ingestion and metadata normalization."""

# Used by parse.py as the default DICOM tag read list, and by clean.py to keep
# matching metadata columns when loading parsed CSV files.
DEFAULT_DICOM_TAGS = [
    # Identifiers used to link instances into patients, studies, and series.
    "PatientID",
    "PatientName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    # Modality and SOP tags used to keep CT image storage and remove PET/NM.
    "Modality",
    "ModalitiesInStudy",
    "SOPClassUID",
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    # Study-level metadata retained for ordering and downstream audit context.
    "StudyDate",
    "StudyTime",
    "StudyDescription",
    "StudyID",
    "AccessionNumber",
    "ReferringPhysicianName",
    # Series-level metadata retained for volume grouping, phase mapping, and QC.
    "SeriesDate",
    "SeriesTime",
    "SeriesDescription",
    "SeriesNumber",
    "ProtocolName",
    "BodyPartExamined",
    "Laterality",
    # Instance-level metadata retained for acquisition ordering and timestamps.
    "InstanceNumber",
    "AcquisitionNumber",
    "TemporalPositionIdentifier",
    "InstanceCreationDate",
    "InstanceCreationTime",
    "ContentDate",
    "ContentTime",
    # Image geometry used for scan quality filters and volume reconstruction.
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceLocation",
    "FrameOfReferenceUID",
    # Image type and acquisition tags used to keep primary axial acquisitions.
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
    # Common CT acquisition/reconstruction settings retained for downstream QC.
    "KVP",
    "ExposureTime",
    "XRayTubeCurrent",
    "Exposure",
    "ExposureModulationType",
    "ConvolutionKernel",
    "ReconstructionDiameter",
    # Pixel value interpretation tags retained for conversion and audit context.
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "RescaleIntercept",
    "RescaleSlope",
    "RescaleType",
    # Limited patient metadata retained for cohort checks.
    "PatientSex",
    "PatientAge",
    "PatientBirthDate",
    # Extra QC helpers retained when present.
    "NumberOfFrames",
    "PositionReferenceIndicator",
    "BurnedInAnnotation",
]

# Used by clean.add_time in priority order to build the normalized time column.
# Earlier tags are preferred; later tags fill missing values.
TIME_CANDIDATES = [
    "AcquisitionTime",
    "ContentTime",
    "SeriesTime",
    "InstanceCreationTime",
    "StudyTime",
]

# Used by clean.add_date in priority order to build the normalized date column.
# Earlier tags are preferred; later tags fill missing values.
DATE_CANDIDATES = [
    "StudyDate",
    "AcquisitionDate",
    "ContentDate",
    "SeriesDate",
    "InstanceCreationDate",
]
