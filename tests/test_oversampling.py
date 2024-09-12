import logging

from collections import Counter

from imblearn.over_sampling import SMOTE

from openforge.utils.prior_model_common import load_openforge_icpsr_benchmark


def test_oversampling():
    data_dir = "/ssd/congtj/openforge/icpsr/artifact"
    logger = logging.getLogger(__name__)

    X_train, y_train, _, _, _, _, _, _, _ = load_openforge_icpsr_benchmark(
        data_dir, logger
    )

    smote = SMOTE(random_state=12345)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(sorted(Counter(y_train_resampled).items()))

    smote = SMOTE(
        sampling_strategy={1: 1000, 2: 1000, 3: 1000},
        random_state=12345,
    )
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(sorted(Counter(y_train_resampled).items()))
