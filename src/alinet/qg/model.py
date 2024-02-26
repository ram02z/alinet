from strenum import StrEnum


class Model(StrEnum):
    BASELINE = "alinet/bart-base-squad-qg"
    BASELINE_NOISE = "alinet/bart-base-spoken-squad-qg"
    BALANCED = "alinet/bart-base-balanced-qg"
    BALANCED_RESOLVED = "alinet/bart-base-balanced-resolved-qg"
    BALANCED_RA = "alinet/bart-base-balanced-ra-qg"
