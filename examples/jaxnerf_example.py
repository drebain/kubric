# Copyright 2021 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

scene = kb.Scene(resolution=(256, 256), background=kb.Color(1.0, 1.0, 1.0))
renderer = KubricRenderer(scene, scratch_dir="output_tmp")

scene += kb.Cube(name="floor", scale=(100, 100, 0.1), position=(0, 0, -0.1))
cube = kb.Cube(name="cube", scale=0.5, position=(0, 0, 0.5))
cube.material = kb.PrincipledBSDFMaterial(color=(1.0, 0.2, 0.5))
scene += cube
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)

camera = kb.PerspectiveCamera(name="camera", look_at=(0, 0, 1))
scene += camera

def update_camera():
    position = np.random.normal(size=(3, ))
    position *= 4 / np.linalg.norm(position)
    position[2] = np.abs(position[2])
    camera.position = position
    camera.look_at((0, 0, 0))
    return camera.matrix_world

def output_split(split_name, n_frames):
    frames = []
    for i in range(n_frames):
        matrix = update_camera()

        frame = renderer.render_still()
        
        dataset_path = f"{split_name}/{i}"
        kb.write_png(frame["rgba"], f"output/nerf/{dataset_path}.png")

        frame_data = {
            "transform_matrix": matrix.tolist(),
            "file_path": dataset_path,
        }
        frames.append(frame_data)

    transforms_data = {
        "camera_angle_x": camera.field_of_view,
        "frames": frames,
    }

    with open(f"output/nerf/transforms_{split_name}.json", "w") as fd:
        json.dump(transforms_data, fd)

n_train_frames = 60
n_val_frames = 10
n_test_frames = 10

output_split("train", n_train_frames)
output_split("val", n_val_frames)
output_split("test", n_test_frames)
