#report.py
def generate_medical_report(state, confidence, cam, vid, voice, missing_media=None):
    note = ""
    if missing_media:
        note = "Note: Report limited due to missing " + ", ".join(missing_media) + ". "
    if state == "Stable":
        return note + (
            "The analysis indicates stable cognitive and emotional condition. "
            "Visual attention, motion stability, and vocal clarity are within normal ranges. "
            "No immediate signs of fatigue or stress detected."
        )

    if state == "Low Focus":
        return note + (
            "The scan shows mild reduction in focus. "
            "Visual stability remains acceptable, however slight attention drift is observed. "
            "A short rest or reduced task intensity is recommended."
        )

    if state == "Elevated Fatigue":
        return note + (
            "The analysis indicates elevated cognitive fatigue. "
            "Video and voice signals suggest reduced engagement and increased mental load. "
            "Continued activity may reduce performance accuracy."
        )

    return note + (
        "High risk indicators detected. "
        "Significant instability in attention and vocal clarity observed. "
        "Immediate rest and further assessment are strongly advised."
    )
