import re

import json
import os
import numpy as np
import sys
import TUI_utils as TU



file_path = os.path.dirname(os.path.abspath(__file__))
# root_path = os.path.join(file_path, "../../Data/dataset_v1")
root_path = os.path.join(file_path, "../Data_samples")

result_path = os.path.join(file_path, "../result/Syntatic_acc")


inst_seg_path = "inst_seg/top_view_camera"
bbox_path = "bbox/top_view_camera"
grasp_path = "output_grasp"
scene_meta_path = "scene_meta"

with open("/home/cubox/ochansol/isaac_code/python/sanjabu/2025/evaluation/Syntatic_acc/syntatic_config.json", "r") as f:
    syntatic_config = json.load(f)


_OBJ_ID_RE = re.compile(r"^obj_\d{3}$")

def _is_obj_id(s: str) -> bool:
    return isinstance(s, str) and bool(_OBJ_ID_RE.match(s))

def _parse_rgba_key(k: str):
    # "(r, g, b, a)" -> [r,g,b,a] (빠른 파서)
    if not (isinstance(k, str) and k.startswith("(") and k.endswith(")")):
        return None
    try:
        parts = [p.strip() for p in k[1:-1].split(",")]
        if len(parts) != 4:
            return None
        vals = [int(p) for p in parts]
        if any(v < 0 or v > 255 for v in vals):
            return None
        return vals
    except Exception:
        return None

def _ensure_keys(d: dict, req: list[str]):
    missing = [k for k in req if k not in d]
    return missing

# -------- 1) inst_seg --------
def validate_inst_seg(data: dict):
    """
    data: { "(r,g,b,a)": {"class": "..."} }
    syntatic_config: {
        "inst_seg_classes": ["obj_006", ...]  # + BACKGROUND, UNLABELLED 자동 허용
    }
    """
    errors = []
    if not isinstance(data, dict):
        return True, ["root: must be object"]

    allowed = set(syntatic_config.get("inst_seg_classes", [])) | {"BACKGROUND", "UNLABELLED"}

    for k, v in data.items():
        if _parse_rgba_key(k) is None:
            errors.append(f"{k}: invalid RGBA key")
            continue
        if not isinstance(v, dict):
            errors.append(f"{k}: value must be object")
            continue
        missing = _ensure_keys(v, ["class"])
        if missing:
            errors.append(f"{k}: missing {missing}")
            continue
        cls = v["class"]
        if not isinstance(cls, str):
            errors.append(f"{k}: class must be string")
            continue
        # 패턴과 enum 동시 충족
        if not (cls in allowed or (_is_obj_id(cls) and cls in allowed)):
            errors.append(f"{k}: class '{cls}' not in allowed list")
    return (len(errors) > 0), errors

# -------- 2) bbox --------
def validate_bbox(data: dict,  max_w=1920, max_h=1080):
    """
    data: { "obj_###": [x1,y1,x2,y2] }
    syntatic_config: { "bbox_classes": ["obj_006", ...] }
    """
    errors = []
    if not isinstance(data, dict):
        return True, ["root: must be object"]

    allowed_keys = set(syntatic_config.get("bbox_classes", []))
    for k, v in data.items():
        if k not in allowed_keys:
            errors.append(f"{k}: key not in enum list")
            continue
        if not (isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(n, int) for n in v)):
            errors.append(f"{k}: value must be 4-int list [x1,y1,x2,y2]")
            continue
        x1, y1, x2, y2 = v
        if not (0 <= x1 <= max_w and 0 <= x2 <= max_w and 0 <= y1 <= max_h and 0 <= y2 <= max_h):
            errors.append(f"{k}: coords out of range (0..{max_w},0..{max_h})")
    return (len(errors) > 0), errors

# -------- 3) grasp_bbox --------
def validate_grasp_bbox(items,  max_w=1920, max_h=1080):
    """
    items: [ {bbox_2d:{bbox:[[[x,y]x4]], center:[cx,cy], width:int, height:int, angle:float}, ...}, ... ]
    syntatic_config: {
        "grasp_classes": ["obj_006", ...],
        "gripper_models": ["UON_Robotics_Jamin_Gripper", ...]
    }
    """
    errors = []
    if not isinstance(items, list):
        return True, ["root: must be array"]
    allowed_objs = set(syntatic_config.get("grasp_classes", []))
    allowed_grippers = set(syntatic_config.get("gripper_models", []))

    req_top = ["bbox_2d","target_points","target_orientation","target_width",
               "target_object","gripper_model","gripper_type","disturbed_object_count"]
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            errors.append(f"[{i}]: must be object"); continue
        miss = _ensure_keys(it, req_top)
        if miss:
            errors.append(f"[{i}]: missing {miss}"); continue

        # bbox_2d
        b2d = it["bbox_2d"]
        if not isinstance(b2d, dict):
            errors.append(f"[{i}].bbox_2d: must be object"); continue
        miss2 = _ensure_keys(b2d, ["bbox","center","width","height","angle"])
        if miss2:
            errors.append(f"[{i}].bbox_2d: missing {miss2}"); continue

        # bbox polygon: [[[ [x,y],...x4 ]]]
        bbox = b2d["bbox"]
        ok_poly = True
        if isinstance(bbox, list) and len(bbox) >= 1 and isinstance(bbox[0], list) and len(bbox[0]) == 4:
            for bbx in bbox:
                for pt in bbx:
                    if pt[0]<0 or pt[0]>=max_w or pt[1]<0 or pt[1]>=max_h :
                      ok_poly = False
                      break
                if ok_poly == False :
                    break
        else:
            ok_poly = False

        if not ok_poly:
            errors.append(f"[{i}].bbox_2d.bbox: invalid out of range")
            continue

        # center
        c = b2d["center"]
        if not (isinstance(c, (list, tuple)) and len(c) == 2 and
                all(isinstance(n, int) for n in c) and
                0 <= c[0] <= max_w and 0 <= c[1] <= max_h):
            errors.append(f"[{i}].bbox_2d.center: invalid or out of range"); continue


        # width/height
        w = b2d["width"]; h = b2d["height"]
        if not (isinstance(w, int) and 1 <= w <= 1080):
            errors.append(f"[{i}].bbox_2d.width: must be int in [1,1080]"); continue

        if not (isinstance(h, int) and 1 <= h <= 1080):
            errors.append(f"[{i}].bbox_2d.height: must be int in [1,1080]"); continue


        # angle (radian, [-pi, pi])
        ang = b2d["angle"]
        if not (isinstance(ang, (int, float)) and -3.141593 <= float(ang) <= 3.141593):
            errors.append(f"[{i}].bbox_2d.angle: must be number in [-pi,pi]"); continue


        # target_points: len 3, number
        tp = it["target_points"]
        if not (isinstance(tp, list) and len(tp) == 3 and all(isinstance(n, (int,float)) for n in tp)):
            errors.append(f"[{i}].target_points: must be 3 numbers"); continue


        # target_orientation: len 3, number in [-180,180]
        to = it["target_orientation"]
        if not (isinstance(to, list) and len(to) == 3 and
                all(isinstance(n, (int, float)) and -180 <= float(n) <= 180 for n in to)):
            errors.append(f"[{i}].target_orientation: must be 3 numbers in [-180,180]"); continue


        # target_width: number in [0,1]
        tw = it["target_width"]
        if not (isinstance(tw, (int,float)) and 0 <= float(tw) <= 1):
            errors.append(f"[{i}].target_width: must be number in [0,1]"); continue


        # target_object: enum & pattern
        tobj = it["target_object"]
        if not (isinstance(tobj, str) and _is_obj_id(tobj) and tobj in allowed_objs):
            errors.append(f"[{i}].target_object: must match obj_### and be in enum list"); continue


        # gripper_model: enum
        gmodel = it["gripper_model"]
        if not (isinstance(gmodel, str) and gmodel in allowed_grippers):
            errors.append(f"[{i}].gripper_model: not in allowed enum"); continue


        # gripper_type: string (제약 없음)
        if not isinstance(it["gripper_type"], str):
            errors.append(f"[{i}].gripper_type: must be string"); continue


        # disturbed_object_count: int >= 0
        doc = it["disturbed_object_count"]
        if not (isinstance(doc, int) and doc >= 0):
            errors.append(f"[{i}].disturbed_object_count: must be int >= 0"); continue


    return (len(errors) > 0), errors

# -------- 4) scene_meta --------
def validate_scene_meta(data: dict):
    """
    {
      "objects": { "obj_###": {...} limited by enum list },
      "Description": str
    }
    """
    errors = []
    if not isinstance(data, dict):
        return True, ["root: must be object"]
    miss = _ensure_keys(data, ["objects","Description"])
    if miss:
        return True, [f"root: missing {miss}"]

    if not isinstance(data["Description"], str):
        errors.append("Description: must be string")

    objs = data["objects"]
    if not isinstance(objs, dict):
        errors.append("objects: must be object")
        return (len(errors) > 0), errors

    allowed_keys = set(syntatic_config.get("scene_meta_classes", []))
    req_fields = ["level_1","level_2","level_3","object_name","color","packaging","features","description"]

    for k, v in objs.items():
        if k not in allowed_keys:
            errors.append(f"objects.{k}: key not in enum list")
            continue
        if not _is_obj_id(k):
            errors.append(f"objects.{k}: must match obj_###")
            continue
        if not isinstance(v, dict):
            errors.append(f"objects.{k}: value must be object"); continue
        miss2 = _ensure_keys(v, req_fields)
        if miss2:
            errors.append(f"objects.{k}: missing {miss2}"); continue

        # level/object_name 최소 길이 1
        for f in ["level_1","level_2","level_3","object_name", "color","features", "description"]:
            if not (isinstance(v[f], str) and len(v[f]) >= 1):
                errors.append(f"objects.{k}.{f}: must be non-empty string")

        # 나머지 문자열 허용 (빈문자열 가능)
        for f in ["packaging"]:
            if not isinstance(v[f], str):
                errors.append(f"objects.{k}.{f}: must be string")
    return (len(errors) > 0), errors



inst_seg_acc_total = []
bbox_acc_total = []
grasp_acc_total = []
scene_meta_acc_total = []



env_list = [i for i in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, i))]
for env in env_list:
    section_list = [i for i in os.listdir(os.path.join(root_path, env)) if os.path.isdir(os.path.join(root_path, env, i))]
    for section in section_list:
        platform_list = [i for i in os.listdir(os.path.join(root_path, env, section)) if os.path.isdir(os.path.join(root_path, env, section, i))]
        for platform in platform_list:

            inst_seg_err_cnt = 0
            bbox_err_cnt = 0
            grasp_err_cnt = 0
            scene_meta_err_cnt = 0

            grasp_total_num = 0

            sample_list = [int(i.strip(".png")) for i in os.listdir(os.path.join(root_path,   env,section, platform, "rgb","top_view_camera")) if i.endswith(".png")]
            data_sample_num = len(sample_list)

            # sample_list = np.random.choice(range(ALL_DATA_NUM),size = data_sample_num, replace=False) #np.random.randint(0,ALL_DATA_NUM,data_sample_num)

            total = len(sample_list)
            bar_w = 32

            for i, scene_num in enumerate(sample_list):
                with open(os.path.join(root_path,   env,section,platform, inst_seg_path, f"semantics_mapping_{scene_num:04d}.json"),'r') as f:
                    inst_seg_label = json.load(f)
                with open(os.path.join(root_path,   env,section,platform, bbox_path, f"{scene_num:04d}.json"),'r') as f:
                    bbox_label = json.load(f)
                with open(os.path.join(root_path,   env,section,platform, grasp_path, f"{scene_num:04d}.json"),'r') as f:
                    grasp_label = json.load(f)
                with open(os.path.join(root_path,   env,section,platform, scene_meta_path, f"{scene_num:04d}.json"),'r') as f:
                    scene_meta_label = json.load(f)
                
                grasp_label_total = len(grasp_label)
                grasp_total_num += grasp_label_total

                inst_seg_err = validate_inst_seg(inst_seg_label)
                bbox_err = validate_bbox(bbox_label)
                grasp_err = validate_grasp_bbox(grasp_label)
                scene_meta_err = validate_scene_meta(scene_meta_label)


                if inst_seg_err[0]:
                    inst_seg_err_cnt += 1
                    print("inst_seg_err : ",inst_seg_err)
                if bbox_err[0]:
                    bbox_err_cnt += 1
                    print("bbox_err",bbox_err)
                if grasp_err[0] :
                    # print("err scene_num : ",scene_num)
                    # print("grasp_err_cnt : " , len(grasp_err[1]))
                    grasp_err_cnt += len(grasp_err[1])
                    # print("grasp_err : ",grasp_err)
                if scene_meta_err[0]:
                    scene_meta_err_cnt += 1
                    print("scene_meta_err : ",scene_meta_err)
                    print("err scene_num : ",scene_num)


                inst_seg_acc = (data_sample_num - inst_seg_err_cnt) / data_sample_num *100
                bbox_acc     = (data_sample_num - bbox_err_cnt) / data_sample_num *100
                grasp_acc    = (grasp_total_num - grasp_err_cnt) / grasp_total_num *100
                scene_meta_acc = (data_sample_num - scene_meta_err_cnt) / data_sample_num *100

                TU.move_up(7)

                print("==============================================                             ")
                print("Processing : ", os.path.join(env, section, platform), "                             ")
                print(TU.render_metric_line("inst_seg acc",  inst_seg_acc,  i+1, total, bar_w)); TU.clr_line()
                print(TU.render_metric_line("bbox acc",      bbox_acc,      i+1, total, bar_w)); TU.clr_line()
                print(TU.render_metric_line("grasp acc",     grasp_acc,     i+1, total, bar_w)); TU.clr_line()
                print(TU.render_metric_line("scene_meta acc", scene_meta_acc,i+1, total, bar_w)); TU.clr_line()
                print("==============================================                              ")
                sys.stdout.flush()


            inst_seg_acc_total.append( inst_seg_acc )
            bbox_acc_total.append(bbox_acc)
            grasp_acc_total.append(grasp_acc)
            scene_meta_acc_total.append(scene_meta_acc)





print(f"{TU.BOLD}{TU.FG['blue']}== Final Summary =={TU.RESET}")
print(TU.render_total_line("inst_seg acc total",   np.mean(inst_seg_acc_total)))
print(TU.render_total_line("bbox acc total",       np.mean(bbox_acc_total)))
print(TU.render_total_line("grasp acc total",      np.mean(grasp_acc_total)))
print(TU.render_total_line("scene_meta acc total", np.mean(scene_meta_acc_total)))

result = {
    "inst_seg_acc_total": np.mean(inst_seg_acc_total),
    "bbox_acc_total": np.mean(bbox_acc_total),
    "grasp_acc_total": np.mean(grasp_acc_total),
    "scene_meta_acc_total": np.mean(scene_meta_acc_total)
}
with open(os.path.join(result_path, "result.json"), 'w') as f:
    json.dump(result, f, indent=4)
# print("==============================================")
# print("inst_seg acc total : ",    np.mean(inst_seg_acc_total), "%")
# print("bbox acc total : ",        np.mean(bbox_acc_total), "%")
# print("grasp acc total : ",       np.mean(grasp_acc_total), "%")
# print("scene_meta acc total : ",  np.mean(scene_meta_acc_total), "%")
 

