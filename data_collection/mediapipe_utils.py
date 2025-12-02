def get_landmarks_header(pose_landmarker_result) -> list[str]:
    if len(pose_landmarker_result.pose_world_landmarks) == 0:
        raise Exception("No pose landmarks")

    landmarks = pose_landmarker_result.pose_world_landmarks[0]
    header = ["frame"]
    for i in range(len(landmarks)):
        header += [f"x{i}", f"y{i}", f"z{i}"]

    return header