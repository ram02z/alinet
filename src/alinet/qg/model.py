from strenum import StrEnum


class Model(StrEnum):
    BASELINE = "alinet/t5-base-squad-qg"
    BASELINE_NOISE = "alinet/t5-base-spoken-squad-qg"
    BALANCED = "alinet/t5-base-balanced-qg"
