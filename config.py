import cv2 as cv


class Config:

    key_rectify = "rectify"
    key_do_flann = "do_flann"
    key_planes_based_matching_merge_components = "key_planes_based_matching_merge_components"

    show_sky_mask = "show_sky_mask"
    show_normals_in_img = "show_normals_in_img"
    save_normals_in_img = "save_normals_in_img"

    svd_smoothing = False
    svd_smoothing_sigma = 1.33
    svd_weighted = True
    svd_weighted_sigma = 0.8
    rectification_interpolation_key = "rectification_interpolation_key"

    # window size

    # init the map and set the default values
    config_map = {}
    config_map[key_rectify] = True
    config_map[key_do_flann] = True
    config_map[key_planes_based_matching_merge_components] = True
    config_map[show_normals_in_img] = True
    config_map[show_sky_mask] = True
    config_map[save_normals_in_img] = True
    config_map[rectification_interpolation_key] = cv.INTER_LINEAR

    @staticmethod
    def log():
        print("Config:\n  {}".format("\n  ".join("{}: {},".format(k, v) for k, v in Config.config_map.items())))

        attr_list = [attr for attr in dir(Config) if not callable(getattr(Config, attr)) and not attr.startswith("__")]
        for attr_name in attr_list:
            print("  {} = {}".format(attr_name, getattr(Config, attr_name)))
        print()

    @staticmethod
    def do_flann():
        return Config.config_map[Config.key_do_flann]

    @staticmethod
    def rectify():
        return Config.config_map[Config.key_rectify]

    @staticmethod
    def set_rectify(bool_val):
        Config.config_map[Config.key_rectify] = bool_val

    @staticmethod
    def set_save(bool_val):
        Config.config_map[Config.save_normals_in_img] = bool_val
