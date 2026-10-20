from datetime import datetime, timedelta, timezone
import math


def calculate_sm2(
    quality: int,
    repetition_count: int,
    interval_days: int,
    ease_factor: float,
) -> tuple[int, int, float, datetime]:
    """Calculate updated SM-2 parameters from a user rating (quality 0-5).

    Returns:
        A tuple of repetition count, interval days, ease factor, and next due date.
    """
    if type(quality) is not int or not (0 <= quality <= 5):
        raise ValueError("Quality score must be between 0 and 5.")
    if repetition_count < 0:
        raise ValueError("Repetition count cannot be negative.")
    if interval_days < 0:
        raise ValueError("Interval days cannot be negative.")
    if not math.isfinite(ease_factor) or ease_factor < 1.3:
        raise ValueError("Ease factor must be finite and at least 1.3.")

    q = float(quality)

    if q >= 3:
        # Correct answer.
        if repetition_count == 0:
            interval = 1
        elif repetition_count == 1:
            interval = 6
        else:
            interval = round(interval_days * ease_factor)

        repetition_count += 1
    else:
        # Incorrect answer: reset the repetition count.
        repetition_count = 0
        interval = 1

    # Adjust the ease factor.
    ease_factor = ease_factor + (
        0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
    )

    # The ease factor must not fall below 1.3.
    if ease_factor < 1.3:
        ease_factor = 1.3

    # Calculate the next review date.
    next_review_date = datetime.now(timezone.utc) + timedelta(days=interval)

    return (
        repetition_count,
        interval,
        round(ease_factor, 2),
        next_review_date,
    )