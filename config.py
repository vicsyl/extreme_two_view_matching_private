
class Config:

    key_rectify = "rectify"
    key_do_flann = "do_flann"

    # init the map and set the default values
    config_map = {}
    config_map[key_rectify] = True
    config_map[key_do_flann] = True

    @staticmethod
    def do_flann():
        return Config.config_map[Config.key_do_flann]

    @staticmethod
    def rectify():
        return Config.config_map[Config.key_rectify]
