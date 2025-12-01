import json


def landmarks_to_json(pose_landmarker_result) -> str:
    # Assuming pose_landmarker_result.pose_world_landmarks[0] is a list of landmarks
    if len(pose_landmarker_result.pose_world_landmarks) == 0:
        return "[]"
    landmarks = pose_landmarker_result.pose_world_landmarks[0]

    # Convert to a list of dicts
    landmarks_list = [
        {
            "x": l.x,
            "y": l.y,
            "z": l.z,
            "visibility": getattr(l, "visibility", None),
            "presence": getattr(l, "presence", None)
        }
        for l in landmarks
    ]

    result_json = json.dumps(landmarks_list, indent=2)
    print(result_json)
    return result_json

def get_landmarks_header(pose_landmarker_result) -> list[str]:
    if len(pose_landmarker_result.pose_world_landmarks) == 0:
        raise Exception("No pose landmarks")

    landmarks = pose_landmarker_result.pose_world_landmarks[0]
    header = ["frame"]
    for i in range(len(landmarks)):
        header += [f"x{i}", f"y{i}", f"z{i}"]

    return header # remove last comma