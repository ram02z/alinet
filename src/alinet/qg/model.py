from strenum import StrEnum


class Model(StrEnum):
    BASELINE = "alinet/bart-base-squad-qg"
    BASELINE_NOISE = "alinet/bart-base-spoken-squad-qg"
    BALANCED = "alinet/base-base-balanced-qg"
    BALANCED_RESOLVED = "alinet/base-base-balanced-resolved-qg"
    BALANCED_RA = "alinet/base-base-balanced-ra-qg"
