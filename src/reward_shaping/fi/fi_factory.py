from src.reward_shaping.fi.t_student_based import TStudentHeightLowPenaltyShiftedFiveHundred, \
    PyBulletTStudentHeightLowPenaltyShiftedFiveHundred, TStudentHeightLowPenaltyNotShifted, LinearShiftedFiveHundred, \
    PyBulletLinearShiftedFiveHundred, TStudentHeightLowPenaltyBigDifferences, TStudentHeightLowPenaltyMediumDifferences, \
    JustFiveHundred, TStudentHeightLowPenaltyMediumDifferencesShifted


class FiFactory:
    FI_MAPPING = {
        "tStudentFromSeminary": TStudentHeightLowPenaltyShiftedFiveHundred,
        "tStudentFromSeminaryNotShifted": TStudentHeightLowPenaltyNotShifted,
        "tStudentFromSeminaryMediumDifferences": TStudentHeightLowPenaltyMediumDifferences,
        "tStudentFromSeminaryMediumDifferencesShifted": TStudentHeightLowPenaltyMediumDifferencesShifted,
        "tStudentFromSeminaryBigDifferences": TStudentHeightLowPenaltyBigDifferences,
        "tStudentFromSeminaryPyBullet": PyBulletTStudentHeightLowPenaltyShiftedFiveHundred,
        "linearSameScaleAsFromSeminary": LinearShiftedFiveHundred,
        "JustFiveHundred": JustFiveHundred,
        "linearSameScaleAsFromSeminaryPyBullet": PyBulletLinearShiftedFiveHundred,
    }

    @staticmethod
    def get(name: str):
        fi = FiFactory.FI_MAPPING.get(name, None)

        if fi is None:
            raise ValueError(f"Unknown fi: {name}, viable options are: {FiFactory.FI_MAPPING.keys()}")

        return fi()
