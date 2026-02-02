import re
import pandas as pd

center_id_dict = {
    1: "BJN",
    2: "Nantes",
    3: "Mondor",
    4: "Dijon",
    5: "Grenoble",
    6: "Bordeaux",
    7: "Strasbourg",
    8: "IGR",
    9: "Angers",
}

source_id_ict = {
    1: "TNE",
    2: "CIRSE",
    3: "proactif",
    4: "proactif_TNE",
    5: "proactif_CHC",
    6: "OMICS_CHC",
}

tumor_type_dict = {
    1: "TNE",  # tumeurs neuroendocrines
    2: "CHC",  # carcinome hépatocellulaire (cancer primitif du foie)
}


def check_operandi_patient_key(patient_key):
    # remove prefix if string starts with 3 digits + underscore
    patient_key = re.sub(r"^\d{3}_", "", patient_key)
    # patient_key = center_id - source_id - patient_id - tumor_type
    tokens = patient_key.split("-")
    tokens = [int(item) for item in tokens]
    assert tokens[0] in center_id_dict
    assert tokens[1] in source_id_ict
    # assert tokens[2].isdigit()
    assert tokens[3] in tumor_type_dict
    return True


def standardize_operandi_patient_key(patient_key):
    # remove prefix if string starts with 3 digits + underscore
    patient_key = re.sub(r"^\d{3}_", "", patient_key)
    # patient_key = center_id - source_id - patient_id - tumor_type
    try:
        """
        for BJN and HMN
        """
        check_operandi_patient_key(patient_key)
        tokens = patient_key.split("-")
        tokens = [s.lstrip("0") for s in tokens]
        return "-".join(tokens[0:4])
    except:
        """
        for AUTRES
        """
        return transform_operandi_patient_key(patient_key)


def transform_operandi_patient_key(patient_key):
    # patient_key = center_id - source_id - patient_id - tumor_type
    try:
        tokens = patient_key.split()
        center_id = int(tokens[-2])
        ptient_local_id = int(tokens[-1])
        return f"{center_id}-2-{ptient_local_id}-2"
    except:
        return None


def extract_from_patient_key(patient_key):
    patient_key = standardize_operandi_patient_key(patient_key)
    tokens = patient_key.split("-")
    tokens = [int(item) for item in tokens]
    # return [center_id_dict[tokens[0]], source_id_ict[tokens[1]], tokens[2], tumor_type_dict[tokens[3]]]
    return pd.Series(
        {
            "center": center_id_dict[tokens[0]],
            "source": source_id_ict[tokens[1]],
            "tumor_type": tumor_type_dict[tokens[3]],
        }
    )
