import numpy as np
def calculate_error(num_actual_detection, num_predicted_detection):
    return abs(num_actual_detection - num_predicted_detection)


def calculate_mae(y_true, y_pred):
    per_image_errors = []

    for actual, pred in zip(y_true, y_pred):

        per_image_errors.append(calculate_error(actual, pred))

    print(f"================================================")
    print(f"MAE: {np.mean(per_image_errors):.2f}")
    print(f"================================================")
    