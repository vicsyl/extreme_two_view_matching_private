from scene_info import SceneInfo

"""
These functions exist to automatically create code that populates the data structures similar 
to those in SceneInfo. This is to:
a) use the data also in places run by python 2 (megadepth) 
b) manually populate the data (clusters_map) - this is strictly not because of python 2
"""


def clusters():

    scene = "scene1"
    scene_info = SceneInfo.read_scene(scene)
    interesting_imgs = scene_info.imgs_for_comparing_difficulty(0, "")
    interesting_imgs = sorted(list(set(interesting_imgs)))

    for img in interesting_imgs:
        print("clusters_map['{}'] = '?'".format(img))


def prepare_diff_0_img():

    scene = "scene1"
    scene_info = SceneInfo.read_scene(scene)
    interesting_imgs = scene_info.imgs_for_comparing_difficulty(0, ".jpg")
    for img in interesting_imgs:
        print('diff_0_img.add("{}")'.format(img))


if __name__ == "__main__":
    clusters()
