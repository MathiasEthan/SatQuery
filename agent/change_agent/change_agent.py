import sys
import os

# Ensure the local Multi_change module can be imported
multi_change_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Multi_change"
)
if multi_change_path not in sys.path:
    sys.path.append(multi_change_path)

try:
    from predict import Change_Perception
except ImportError:
    from Multi_change.predict import Change_Perception


def run_model(path_a, path_b, task):
    """
    Run the Change Perception model.
    task: "analyze" (default), "caption", "detection"
    """
    model = Change_Perception()
    if task == "analyze":
        return model.analyze_changes(path_a, path_b)
    elif task == "caption":
        return model.generate_change_caption(path_a, path_b)
    elif task == "detection":
        return model.change_detection(path_a, path_b)
    else:
        raise ValueError("Unsupported task")
