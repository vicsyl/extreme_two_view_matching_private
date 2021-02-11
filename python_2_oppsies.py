from scene_info import SceneInfo

def main():

    scene = "scene1"
    scene_info = SceneInfo.read_scene(scene)
    interesting_imgs = scene_info.imgs_for_comparing_difficulty(0, ".jpg")
    for img in interesting_imgs:
        print('diff_0_img.add("{}")'.format(img))


if __name__ == "__main__":
    main()
