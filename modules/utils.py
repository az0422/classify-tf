import yaml
import os

def parse_cfg(cfg):
    cfg_default = yaml.full_load(open(cfg, "r"))
    if cfg_default["user_option"] is None or not os.path.isfile(cfg_default["user_option"]):
        return cfg_default
    
    user_option = yaml.full_load(open(cfg_default["user_option"], "r"))

    if user_option is None:
        return cfg_default
    
    for key in user_option.keys():
        if user_option[key] is None: continue
        cfg_default[key] = user_option[key]
    
    return cfg_default

def apply_local_cfg(cfg, file):
    assert os.path.isfile(file), "%s is not file" % file
    local_cfg = yaml.full_load(open(file, "r"))

    for key in local_cfg.keys():
        if local_cfg[key] is None: continue
        cfg[key] = local_cfg[key]
    
    return cfg