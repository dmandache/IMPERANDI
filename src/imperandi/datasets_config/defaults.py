# Used by clean.py after grouped volumes get a computed volume_length in mm.
# Volumes outside this inclusive range are dropped; missing lengths are kept.
DEFAULT_VOLUME_LENGTH_MIN_MM = 30.0  # Keep volumes with length >= 3 cm.
DEFAULT_VOLUME_LENGTH_MAX_MM = 1700.0  # Keep volumes with length <= 170 cm.

# Used by clean.py's pixel-spacing derivation. Rows with missing values are
# kept; rows with present values above this maximum are dropped.
DEFAULT_MAX_PIXEL_SPACING_MM = 1.25  # Max in-plane PixelSpacing[0] (XY), in mm.

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
