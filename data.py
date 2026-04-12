import random


def generate_patient_easy():
    return {
        "age": random.randint(10, 80)
    }


def generate_patient_medium():
    return {
        "age": random.randint(20, 80),
        "egfr": random.randint(20, 100),
        "hba1c": round(random.uniform(5.0, 10.0), 1),
        "medications": random.choice([
            ["none"],
            ["insulin"],
            ["warfarin"],
            ["metformin"]
        ])
    }


def generate_patient_hard():
    return {
        "age": random.randint(40, 80),
        "egfr": random.choice([44, 45, 46]),
        "hba1c": round(random.uniform(6.0, 10.0), 1),
        "medications": random.choice([
            ["none"],
            ["insulin"],
            ["warfarin"],
            ["metformin"]
        ]),
        "conditions": random.choice([
            ["diabetes"],
            ["hypertension"],
            []
        ])
    }


def generate_patient(task_id="single_criterion"):
    if task_id == "single_criterion":
        return generate_patient_easy()
    elif task_id == "multi_criteria":
        return generate_patient_medium()
    else:
        return generate_patient_hard()