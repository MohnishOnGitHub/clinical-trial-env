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
            [], 
            ["insulin"], 
            ["warfarin"], 
            ["metformin"]
        ])
    }


def generate_patient_hard():
    return {
        "age": random.randint(25, 75),
        "egfr": random.choice([44, 45, 46]),  # borderline
        "hba1c": random.choice([7.9, 8.0, 8.1]),  # borderline
        "medications": random.choice([
            ["insulin"],
            ["warfarin"],
            ["metformin"],
            []
        ]),
        "conditions": random.choice([
            ["diabetes"],
            ["hypertension"],
            []
        ])
    }


def generate_patient(task_id="easy"):
    if task_id == "easy":
        return generate_patient_easy()
    elif task_id == "medium":
        return generate_patient_medium()
    else:
        return generate_patient_hard()