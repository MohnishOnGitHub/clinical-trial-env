import random

# ---------- NORMALIZATION ----------
def normalize(raw):
    score = (raw + 20) / 42
    return max(0.001, min(0.999, score))


# ---------- REASON CHECK ----------
def good_reason(reason, patient):
    if not reason:
        return False

    r = reason.lower()

    return (
        ("egfr" in r and str(patient["egfr"]) in r) or
        ("hba1c" in r and str(patient["hba1c"]) in r) or
        ("age" in r and str(patient["age"]) in r)
    )


# ---------- BASE ----------
class BaseTask:
    def __init__(self, task_id):
        self.task_id = task_id

    def get_criteria(self):
        raise NotImplementedError

    def generate_patient(self):
        raise NotImplementedError

    def grade(self, action, patient, questions_asked):
        raise NotImplementedError


# ---------- EASY ----------
class EasyTask(BaseTask):
    def __init__(self):
        super().__init__("single_criterion")

    def get_criteria(self):
        return {
            "age_min": 18,
            "age_max": 65
        }

    def generate_patient(self):
        return {
            "age": random.randint(10, 80)
        }

    def grade(self, action, patient, questions_asked):
        eligible = 18 <= patient["age"] <= 65
        correct = action.eligible == eligible

        raw = 20 if correct else -20
        raw -= min(questions_asked, 5)

        if action.reason:
            raw += 1

        if good_reason(action.reason, patient):
            raw += 1

        return normalize(raw)


# ---------- MEDIUM ----------
class MediumTask(BaseTask):
    def __init__(self):
        super().__init__("multi_criteria")

    def get_criteria(self):
        return {
            "age_min": 30,
            "age_max": 70,
            "egfr_min": 45,
            "hba1c_max": 8.0,
            "no_meds": ["warfarin", "insulin"]
        }

    def generate_patient(self):
        return {
            "age": random.randint(20, 80),
            "egfr": random.randint(20, 90),
            "hba1c": round(random.uniform(5.0, 10.0), 1),
            "medications": random.sample(
                ["warfarin", "insulin", "metformin", "none"], 1
            )
        }

    def grade(self, action, patient, questions_asked):
        meds = patient.get("medications", [])

        checks = [
            30 <= patient["age"] <= 70,
            patient["egfr"] >= 45,
            patient["hba1c"] <= 8.0,
            not any(m in ["warfarin", "insulin"] for m in meds)
        ]

        eligible = all(checks)
        correct = action.eligible == eligible

        raw = 20 if correct else -20
        raw -= questions_asked

        # partial credit per criterion
        raw += sum(checks)

        if good_reason(action.reason, patient):
            raw += 2

        return normalize(raw)


# ---------- HARD ----------
class HardTask(BaseTask):
    def __init__(self):
        super().__init__("edge_case")

    def get_criteria(self):
        return {
            "age_min": 45,
            "age_max": 75,
            "egfr_min": 45,
            "hba1c_min": 6.5,
            "hba1c_max": 9.5
        }

    def generate_patient(self):
        return {
            "age": random.randint(40, 80),
            "egfr": random.choice([44, 45, 46]),
            "hba1c": round(random.uniform(6.0, 10.0), 1),
            "medications": ["none"],
            "conditions": []
        }

    def grade(self, action, patient, questions_asked):
        eligible = (
            45 <= patient["age"] <= 75 and
            patient["egfr"] >= 45 and
            6.5 <= patient["hba1c"] <= 9.5
        )

        correct = action.eligible == eligible

        raw = 20 if correct else -20
        raw -= questions_asked

        if action.reason and str(patient["egfr"]) in action.reason:
            raw += 2

        return normalize(raw)


# ---------- REGISTRY ----------
TASKS = {
    "single_criterion": EasyTask(),
    "multi_criteria": MediumTask(),
    "edge_case": HardTask()
}