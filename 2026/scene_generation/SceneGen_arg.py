import argparse

parser = argparse.ArgumentParser(description="Sanjabu Scene Generator")
parser.add_argument("--output_root_path",       default="/nas/Dataset/Dataset_2026/test",              help="", type=str)
parser.add_argument("--env_name",       default="Logistic_site",              help="", type=str)
parser.add_argument("--section_name",   default="General_LogisticSite",  help="", type=str)
parser.add_argument("--platform_name",  default="rack_small_A5_01",              help="", type=str)
parser.add_argument("--scene_start",      default=0,                          help="", type=int)
parser.add_argument("--scene_end",        default=10,                         help="", type=int)
parser.add_argument("--object_num",     default=5,                          help="", type=int)

args = parser.parse_args()

import sanjabu_scene_generator as ssg

ssg.main( 
        output_root_path = args.output_root_path,
        env_name      = args.env_name,
        section_name  = args.section_name,
        platform_name = args.platform_name,
        scene_start     = args.scene_start,
        scene_end       = args.scene_end,
        object_num    = args.object_num,
)
