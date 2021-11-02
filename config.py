import cv2 as cv


class Config:

    key_do_flann = "do_flann"
    key_planes_based_matching_merge_components = "key_planes_based_matching_merge_components"

    # Toft et al. use 80 (but the implementation details actually differ)
    plane_threshold_degrees = 75

    svd_smoothing = False
    svd_smoothing_sigma = 1.33
    svd_weighted = True
    svd_weighted_sigma = 0.8
    rectification_interpolation_key = "rectification_interpolation_key"

    # window size

    # init the map and set the default values
    config_map = {}
    config_map[key_do_flann] = True
    config_map[key_planes_based_matching_merge_components] = True
    config_map[rectification_interpolation_key] = cv.INTER_LINEAR

    @staticmethod
    def log():
        print("Config:")
        print("\t{}".format("\n\t".join("{}\t{}".format(k, v) for k, v in Config.config_map.items())))

        attr_list = [attr for attr in dir(Config) if not callable(getattr(Config, attr)) and not attr.startswith("__")]
        for attr_name in attr_list:
            print("\t{}\t{}".format(attr_name, getattr(Config, attr_name)))
        print()

    @staticmethod
    def do_flann():
        return Config.config_map[Config.key_do_flann]


def get_default_cfg():
    return {
        "fginn": False,
        # TODO should I use it outside of fginn?
        "num_nn": 2,
        "fginn_spatial_th": 100,
        "ratio_th": 0.5
    }


def parse_line(key: str, value: str, cfg_map):

    if key == "fginn":
        cfg_map["fginn"] = value.lower() == "true"
    elif key == "num_nn":
        cfg_map["num_nn"] = int(value)
    elif key == "fginn_spatial_th":
        cfg_map["fginn_spatial_th"] = int(value)
    elif key == "ratio_th":
        cfg_map["ratio_th"] = float(value)
    else:
        print("WARNING - unrecognized param: {}".format(key))
