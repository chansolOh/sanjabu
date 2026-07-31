import numpy as np
import json 
# import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import os
import tkinter as tk
import ast
import utils
from tqdm import tqdm

file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_path, "/nas/Dataset/Dataset_2025/dataset_v1")
# root_path = os.path.join(file_path, "../Data_samples")
result_path = os.path.join(file_path, "../result/Semantic_acc/grasp_bbox")

# with open("/home/cubox/ochansol/isaac_code/python/sanjabu/2025/evaluation/Semantic_acc/sampling_list_sample.txt", "r") as f:
#     sampling_path_list = f.read().splitlines()
with open("/home/cubox/ochansol/isaac_code/python/sanjabu/2025/evaluation/Semantic_acc/sampling_list.txt", "r") as f:
    sampling_path_list = f.read().splitlines()


###### data path ######
rgb_path = "rgb/top_view_camera"
bbox_path  = "output_grasp"
inst_seg_path = "inst_seg/top_view_camera"


def rot_z(rad):

    return np.array([
        [np.cos(rad), -np.sin(rad) ],
        [np.sin(rad),  np.cos(rad)],
    ])


class bbox_display():
    def __init__(self, root_path, rgb_path, bbox_path, inst_seg_path, sampling_path_list):
        
        self.root_path = root_path
        self.rgb_path = rgb_path
        self.bbox_path = bbox_path
        self.inst_seg_path = inst_seg_path
        self.sampling_path_list = sampling_path_list
        
        self.img_sampling_num = len(sampling_path_list)
        self.grasp_sample_num_list = []

        if os.path.exists(os.path.join(result_path,"acc.json")):
            with open(os.path.join(result_path,"acc.json"), 'r') as f:
                self.acc_dict = json.load(f)
        else:
            self.acc_dict = {}
            for sampling_num in self.sampling_path_list:
                self.acc_dict[sampling_num] = {}



        
        
        
        self.window = tk.Tk()
        self.window.title("Bounding Box")
        self.window.geometry("1000x1300")

        
        
        ####### plt #######
        
        
        
        
        ##### Frame #####
        self.img_frame = tk.Frame(self.window)
        self.img_frame.pack(side=tk.TOP, pady=(0,30) ,fill=tk.X)
        
        self.left_frame = tk.Frame(self.window)
        self.left_frame.pack(side=tk.LEFT,fill=tk.X)
        
        self.right_frame = tk.Frame(self.window)
        self.right_frame.pack(side=tk.RIGHT,fill=tk.X)
        
        self.eval_frame = tk.Frame(self.left_frame)
        self.eval_frame.pack(side=tk.TOP,pady=30,fill=tk.X)
        
        self.eval_top_frame = tk.Frame(self.eval_frame)
        self.eval_top_frame.pack(side=tk.TOP,fill=tk.X)
        self.eval_L_frame = tk.Frame(self.eval_frame)
        self.eval_L_frame.pack(side=tk.LEFT,fill=tk.X)
        self.eval_R_frame = tk.Frame(self.eval_frame)
        self.eval_R_frame.pack(side=tk.RIGHT,fill=tk.X)
        
        self.move_frame = tk.Frame(self.left_frame)
        self.move_frame.pack(side=tk.TOP,fill=tk.X)
        
        self.scene_frame = tk.Frame(self.move_frame)
        self.scene_frame.pack(side=tk.BOTTOM,fill=tk.X)
        
        self.grasp_option_frame = tk.Frame(self.right_frame)
        self.grasp_option_frame.pack(side=tk.TOP,fill=tk.X)
        
        self.grasp_move_frame = tk.Frame(self.right_frame)
        self.grasp_move_frame.pack(side=tk.TOP,fill=tk.X)
        
    
        
        
        ######  layout ######
        
        self.file_name_label = tk.Label(self.img_frame,  font = ("Arial",20), text="file_name")
        self.file_name_label.pack(side=tk.TOP)

        self.img_label = tk.Label(self.img_frame, bg="black")
        self.img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.img_w = 1120
        self.img_size = (self.img_w, int(self.img_w/16*9))  # 16:9 비율
        self._scale = self.img_size[0]/1920  # 1920x1080 → 640x360
        self._photo = None  # 현재 표시중인 PhotoImage
        self._grasp_vis_mode = "all"  # "one" 또는 "all"
        self.plt_bbox = []  # 그랩 별 라인세그먼트 캐시(아래 3-2에서 채움)

        
        self.slider = tk.Scale(self.img_frame, orient=tk.HORIZONTAL, width = 30, length=400, sliderlength=100, from_=0, to=100, command=self.slider_move)
        self.slider.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        self.slider_label = tk.Label(self.img_frame, font = ("Arial",20), text="Line Width")
        self.slider_label.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        
        self.acc_label = tk.Label(self.eval_top_frame, font = ("Arial",20,"bold"),fg="#FF0000", text="Accuracy")
        self.acc_label.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        
        self.correct_label = tk.Label(self.eval_L_frame, font = ("Arial",20), text="Correct")
        self.correct_label.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)
        self.incorrect_label = tk.Label(self.eval_R_frame, font = ("Arial",20), text="Incorrect")
        self.incorrect_label.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)
        
        self.correct_btn = tk.Button(self.eval_L_frame, width = 3, text="O", font = ("Arial",80), command=self.correct, bg="#FFFFFF" , activebackground="#50FF50")
        self.correct_btn.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        self.incorrect_btn = tk.Button(self.eval_R_frame, width = 3, text="X", font = ("Arial",80),command=self.incorrect, bg="#FFFFFF", activebackground="#FF5050")
        self.incorrect_btn.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        

        self.prev_btn = tk.Button(self.move_frame, width =7, font = ("Arial",30), text="Prev", command=self.prev)
        self.prev_btn.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        self.next_btn = tk.Button(self.move_frame, width = 7,font = ("Arial",30), text="Next", command=self.next)
        self.next_btn.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        self.save_btn = tk.Button(self.move_frame, width = 6,font = ("Arial",30), text="Save", command=self.save)
        self.save_btn.pack(side=tk.RIGHT)
        
        self.scene_num_label = tk.Label(self.scene_frame, font = ("Arial",22), text="Scene Num")
        self.scene_num_label.pack(side=tk.LEFT, padx=10,fill=tk.BOTH, expand=True)
        self.scene_entry = tk.Entry(self.scene_frame, width=5, font = ("Arial",20))
        self.scene_entry.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        self.scene_entry.bind("<Return>", self.go_to_scene)
        self.scene_entry.bind("<KP_Enter>", self.go_to_scene)
        self.scene_change_btn = tk.Button(self.scene_frame, width = 3, font = ("Arial",20), text="Go", command= lambda : self.go_to_scene(None))
        self.scene_change_btn.pack(side=tk.RIGHT,fill=tk.BOTH, expand=True)
        
        self.window.bind("<Button-1>", lambda event: event.widget.focus_set())
        self.window.bind("<Left>", lambda x:self.prev() if self.window.focus_get() != self.scene_entry else None)
        self.window.bind("<Right>", lambda x: self.next() if self.window.focus_get() != self.scene_entry else None)
        
        
        self.inst_seg_var = tk.IntVar()
        self.inst_seg_vis_check = tk.Checkbutton(self.grasp_option_frame, width = 15, font = ("Arial",20), text="Viz Segmentation", bg = "white",activebackground="#eeeeee", variable=self.inst_seg_var, command=self.inst_seg_vis)
        self.inst_seg_vis_check.pack(side=tk.TOP, pady=(0,20),fill=tk.BOTH, expand=True)
        self.grasp_auto_eval_btn = tk.Button(self.grasp_option_frame, width = 8, font = ("Arial",25), text="Auto Eval", bg = "#90F060",command=self.grasp_auto_eval)
        self.grasp_auto_eval_btn.pack(side=tk.LEFT, pady=(0,30),fill=tk.BOTH, expand=True)
        self.grasp_auto_eval_all_data_btn = tk.Button(self.grasp_option_frame, width = 5, font = ("Arial",10), text="Auto Eval All", bg = "#90F060",command=self.auto_eval_all)
        self.grasp_auto_eval_all_data_btn.pack(side=tk.RIGHT, pady=(0,30),fill=tk.BOTH, expand=True)
        
        self.grasp_show_all_btn = tk.Button(self.grasp_move_frame, width = 10,font = ("Arial",30), text="Show All", command=self.grasp_show_all)
        self.grasp_show_all_btn.pack(side=tk.BOTTOM,pady=(10,0),fill=tk.BOTH, expand=True)
        self.grasp_slider = tk.Scale(self.grasp_move_frame, orient=tk.HORIZONTAL, width = 30, length=400, sliderlength=100, from_=1, to=100, command=self.grasp_slider_move)
        self.grasp_slider.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)
        self.grasp_num_label = tk.Label(self.grasp_move_frame, font = ("Arial",22), text="Grasp Num")
        self.grasp_num_label.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)
        self.grasp_prev_btn = tk.Button(self.grasp_move_frame, width = 5, font = ("Arial",50), text="\u2190", command=self.grasp_prev)
        self.grasp_prev_btn.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        self.grasp_next_btn = tk.Button(self.grasp_move_frame, width = 5,font = ("Arial",50), text="\u2192", command=self.grasp_next)
        self.grasp_next_btn.pack(side=tk.RIGHT,fill=tk.BOTH, expand=True)





        ###### init ######
        self.scene_num = -1
        self.grasp_num = -1
        self.bbox_line_width = 2

        self.acc_label.config(text=f"Accuracy : {self.evaluate()}")
        
        self.window.mainloop()
        
        
        
        
    def next(self):
        if self.scene_num<self.img_sampling_num:
            self.scene_num += 1
            self.data_load(self.sampling_path_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.img_sampling_num}")
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, f"{self.scene_num+1}")
            
            self.eval_btn_update()

    def prev(self):
        if self.scene_num>0:
            self.scene_num -= 1
            self.data_load(self.sampling_path_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.img_sampling_num}")
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, f"{self.scene_num+1}")
            
            self.eval_btn_update()
    def save(self):
        with open(os.path.join(result_path,"acc.json"), 'w') as f:
            json.dump(self.acc_dict, f, indent=4)

    def correct(self):
        if self.scene_num != -1 and self.plt_bbox:
            self.acc_dict[self.sampling_path_list[self.scene_num]][self.grasp_num] = True
            self.acc_label.config(text=f"Accuracy : {round(self.evaluate(),3)}")
            # 세그먼트 재생성(파랑/빨강)
            grasp = self.bbox[self.grasp_num]
            gtype = grasp["gripper_type"]
            # bbox 원본 다시 계산
            bbox_2d = grasp["bbox_2d"]["bbox"]
            center  = np.array(grasp["bbox_2d"]["center"], dtype=float)
            w = float(grasp["bbox_2d"]["width"]); h = float(grasp["bbox_2d"]["height"]); a = float(grasp["bbox_2d"]["angle"])
            if gtype in ["finger2_parallel","finger2"]:
                rect = np.array([[-w/2,-h/2],[-w/2,+h/2],[+w/2,+h/2],[+w/2,-h/2]], dtype=float)
                bb = [rot_z(a).dot(rect.T).T + center]
            else:
                bb = np.array(bbox_2d, dtype=float)
            self.plt_bbox[self.grasp_num] = self._build_segments(bb, gtype, ok=True)
            self.eval_btn_update()
            self._repaint()

    def incorrect(self):
        if self.scene_num != -1 and self.plt_bbox:
            self.acc_dict[self.sampling_path_list[self.scene_num]][self.grasp_num] = False
            self.acc_label.config(text=f"Accuracy : {round(self.evaluate(),3)}")
            grasp = self.bbox[self.grasp_num]
            gtype = grasp["gripper_type"]
            bbox_2d = grasp["bbox_2d"]["bbox"]
            center  = np.array(grasp["bbox_2d"]["center"], dtype=float)
            w = float(grasp["bbox_2d"]["width"]); h = float(grasp["bbox_2d"]["height"]); a = float(grasp["bbox_2d"]["angle"])
            if gtype in ["finger2_parallel","finger2"]:
                rect = np.array([[-w/2,-h/2],[-w/2,+h/2],[+w/2,+h/2],[+w/2,-h/2]], dtype=float)
                bb = [rot_z(a).dot(rect.T).T + center]
            else:
                bb = np.array(bbox_2d, dtype=float)
            self.plt_bbox[self.grasp_num] = self._build_segments(bb, gtype, ok=False)
            self.eval_btn_update()
            self._repaint()

    
    def eval_btn_update(self):
        self.correct_btn.config(bg="#FFFFFF")
        self.incorrect_btn.config(bg="#FFFFFF")
        try:
            if self.acc_dict[self.sampling_path_list[self.scene_num]][self.grasp_num] == True:
                self.correct_btn.config(bg="#00FF00")
            elif self.acc_dict[self.sampling_path_list[self.scene_num]][self.grasp_num] == False:
                self.incorrect_btn.config(bg="#FF0000")
        except:
            pass



    


    def data_load(self, scene_path):


        scene_num = int(scene_path.split("/")[-1])
        # import pdb; pdb.set_trace()
        middle_path = scene_path.split(f"/{scene_num:04d}")[0]
        self.file_name_label.config(text=f"File Name : {middle_path}/{scene_num:04d}.png")


        with open(os.path.join(self.root_path, middle_path, self.inst_seg_path, f"semantics_mapping_{scene_num:04d}.json"),"r") as f:
            self.inst_seg_label = json.load(f)
        with open(os.path.join(self.root_path, middle_path, self.bbox_path, f"{scene_num:04d}.json"), "r") as f:
            self.bbox = json.load(f)
            
        # --- 이미지 로드 (원본 1920x1080) ---
        self.rgb_img = Image.open(os.path.join(self.root_path, middle_path, self.rgb_path, f"{scene_num:04d}.png")).convert("RGB")
        self.inst_seg_img = Image.open(os.path.join(self.root_path, middle_path, self.inst_seg_path, f"{scene_num:04d}.png")).convert("RGBA")
        self.inst_seg_arr = np.array(self.inst_seg_img)

        # --- seg overlay 생성 & 둘 다 640x360으로 축소 보관 ---
        new_size = self.img_size
        rgb_small = self.rgb_img.resize(new_size, Image.LANCZOS)
        seg_small = self.inst_seg_img.resize(new_size, Image.LANCZOS)
        # self.inst_seg_arr = np.array(seg_small)
        ov = seg_small.copy(); ov.putalpha(128)
        self.small_rgb = rgb_small
        self.small_comp = Image.alpha_composite(rgb_small.convert("RGBA"), ov).convert("RGB")

        # --- bbox 세그먼트 캐시 만들기 (원본 좌표 유지, 색상은 일단 'ok=True'로 초기화) ---
        self.plt_bbox = []   # 각 grasp마다 segments 리스트
        cnt = 0
        for grasp in self.bbox:
            bbox_2d = grasp["bbox_2d"]["bbox"]
            center  = np.array(grasp["bbox_2d"]["center"], dtype=float)
            width   = float(grasp["bbox_2d"]["width"])
            height  = float(grasp["bbox_2d"]["height"])
            angle   = float(grasp["bbox_2d"]["angle"])
            gtype   = grasp["gripper_type"]

            if gtype in ["finger2_parallel","finger2"]:
                rect = np.array([[-width/2, -height/2],
                                [-width/2, +height/2],
                                [+width/2, +height/2],
                                [+width/2, -height/2]], dtype=float)
                bb = [rot_z(angle).dot(rect.T).T + center]  # (4,2) 원본 좌표
            else:
                bb = np.array(bbox_2d, dtype=float)       # (4,2) or (n,4,2)

            # 기존 저장된 정오표시 있으면 그 색으로, 없으면 기본 True색
            try:
                ok = bool(self.acc_dict[self.sampling_path_list[self.scene_num]][cnt])
            except:
                ok = True
            self.plt_bbox.append(self._build_segments(bb, gtype, ok=ok))
            cnt += 1

        # --- 첫 화면 그리기 (현재 모드에 맞춰 렌더) ---
        self.grasp_slider.config(to=len(self.plt_bbox))
        self.grasp_num = 0
        self.grasp_slider.set(self.grasp_num+1)
        self.grasp_num_label.config(text=f"Grasp Num : {self.grasp_num+1}/{len(self.plt_bbox)}")
        self._repaint()   # <<< 새로 추가(아래 4번)

            
            
        
    
    def slider_move(self, val):
        self.bbox_line_width = float(val)/100 * 15
        self._repaint()
    
    def evaluate(self):
        correct = 0
        incorrect = 0
        for key in self.acc_dict.keys():
            for grasp_key in self.acc_dict[key].keys():
                if self.acc_dict[key][grasp_key] == True:
                    correct += 1
                elif self.acc_dict[key][grasp_key] == False:
                    incorrect += 1
        # print(correct, incorrect)
        self.correct_label.config(text=f"Correct : {correct}")
        self.incorrect_label.config(text=f"Incorrect : {incorrect}")
        
        if correct==0:
            return 0
        return correct/(correct+incorrect)
    
    def go_to_scene(self, event):
        if int(self.scene_entry.get()) <= self.img_sampling_num and int(self.scene_entry.get()) > 0:
            self.scene_num = int(self.scene_entry.get())-1
            self.data_load(self.sampling_path_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.img_sampling_num}")
            
            self.eval_btn_update()
            
    def grasp_next(self):
        if self.grasp_num<len(self.plt_bbox):
            self.grasp_num += 1
            self.grasp_slider.set(self.grasp_num+1)
            self.eval_btn_update()
        

    def grasp_prev(self):
        if self.grasp_num>0:
            self.grasp_num -= 1
            self.grasp_slider.set(self.grasp_num+1)
            self.eval_btn_update()
            
    def grasp_slider_move(self, val):
        if self.plt_bbox:
            self._grasp_vis_mode = "one"
            self.grasp_num = int(val) - 1
            self.grasp_num_label.config(text=f"Grasp Num : {self.grasp_num+1}/{len(self.plt_bbox)}")
            self.eval_btn_update()
            self._repaint()

    def grasp_show_all(self):
        if self.plt_bbox:
            self._grasp_vis_mode = "all"
            self._repaint()
            
            
    def grasp_auto_eval(self):
        self.grasp_auto_eval_btn.config(state="disabled")

        points = []
        bbox_list = []
        inst_dict = {}
        inst_seg_sampling_rate = 0.2
        # import time

        # t = time.time()
        for inst_key in self.inst_seg_label.keys():
            if self.inst_seg_label[inst_key]["class"] not in ["BACKGROUND","UNLABELLED"]:
                rgba = ast.literal_eval(inst_key)
                idx = np.where(np.all(self.inst_seg_arr==np.array(rgba) ,axis=2) == True)   # idx = 2 x N
                idx_num = np.random.choice( np.arange(len(idx[0])), size=int(len(idx[0])*inst_seg_sampling_rate), replace=False)
                idx = np.array(idx)[:,idx_num]
                inst_dict[self.inst_seg_label[inst_key]["class"]] = idx
        # print("inst seg time :", time.time()-t)
        # t = time.time()


        for grasp in self.bbox:
            bbox_2d = grasp["bbox_2d"]["bbox"]
            center  = np.array(grasp["bbox_2d"]["center"], dtype=float)
            width   = float(grasp["bbox_2d"]["width"])
            height  = float(grasp["bbox_2d"]["height"])
            angle   = float(grasp["bbox_2d"]["angle"])
            gtype   = grasp["gripper_type"]
            target = grasp["target_object"]

            if gtype in ["finger2_parallel","finger2"]:
                rect = np.array([[-width/2, -height/2],
                                [-width/2, +height/2],
                                [+width/2, +height/2],
                                [+width/2, -height/2]], dtype=float)
                bb = [rot_z(angle).dot(rect.T).T + center]  # (4,2) 원본 좌표
            else:
                bb = np.array(bbox_2d)      # (4,2) or (n,4,2)
            bbox_list.append({target:bb})

        # print("bbox prep time :", time.time()-t)
        # t = time.time()


        for bbox_dict in bbox_list:
            for key, bbox in bbox_dict.items():
                bbox = np.array(bbox).transpose(0,2,1)
                idx = utils.select_points(bboxes=bbox, points = inst_dict[key], early_stop=True)
                points.append(idx)


        for cnt, pt_c in enumerate(points):
            point_flag = False
            for pt_b in pt_c:
                if len(pt_b) != 0:
                    point_flag = True
                    break
            if not point_flag:
                self.plt_bbox[cnt] = self._build_segments(list(bbox_list[cnt].values())[0], gtype, ok=False)
                self.acc_dict[self.sampling_path_list[self.scene_num]][cnt] = False
            else:
                self.acc_dict[self.sampling_path_list[self.scene_num]][cnt] = True

                # self.ax.scatter(pt_b.T[1], pt_b.T[0], c="g", s=1)
        # print("auto eval time :", time.time()-t)


        self._repaint()
        self.acc_label.config(text=f"Accuracy : {round(self.evaluate(),3)}")
        self.eval_btn_update()
        self.grasp_auto_eval_btn.config(state="normal")
        


    def inst_seg_vis(self):
        if self.inst_seg_var.get() == 1:
            self.inst_seg_vis_check.config(bg="yellow")
            self.inst_seg_vis_check.config(font=("Arial",20,"bold"))
        else:
            self.inst_seg_vis_check.config(bg="white")
            self.inst_seg_vis_check.config(font=("Arial",20))
        self._repaint()
            
    def _edge_colors(self, ok=True):
        # finger2: 파랑/빨강, incorrect: 보라/연두
        return (("#0000FF","#FF0000") if ok else ("#FF00FF","#00FF00"))

    def _build_segments(self, bbox, gripper_type, ok=True):
        """bbox(4,2) 또는 (n,4,2)를 받아 그릴 선분 목록으로 변환
        return: [((x1,y1),(x2,y2),color), ...]   # 이미 축소스케일 적용 안 함
        """
        c1, c2 = self._edge_colors(ok)
        segs = []
        def add4(bb):
            # bb: (4,2) [0,1,2,3]
            segs.append((bb[1], bb[2], c1))
            segs.append((bb[3], bb[0], c1))
            segs.append((bb[0], bb[1], c2))
            segs.append((bb[2], bb[3], c2))

        def add3(bb):
            # bb: (4,2) [0,1,2,3]
            segs.append((bb[1], bb[2], c1))
            segs.append((bb[3], bb[0], c1))
            segs.append((bb[0], bb[1], c2))
            # segs.append((bb[2], bb[3], c2))

        if gripper_type in ["finger2","finger2_parallel"]:
            add4(bbox[0])
        else:
            arr = np.array(bbox, dtype=float)
            if arr.ndim == 2 and arr.shape == (4,2):
                add3(arr)
            else:
                for one in arr:
                    add3(one)
        return segs

    def _draw_segments(self, img, segments, width):
        """segments: [((x1,y1),(x2,y2),color), ...]  좌표는 원본(1920x1080) 기준
        img는 640x360이므로 그리기 전에 스케일 적용
        """
        s = self._scale
        draw = ImageDraw.Draw(img)
        for (p1, p2, col) in segments:
            x1,y1 = p1[0]*s, p1[1]*s
            x2,y2 = p2[0]*s, p2[1]*s
            draw.line((x1,y1,x2,y2), fill=col, width=int(max(1, width)))

    def _base_image(self):
        # 세그멘테이션 토글 상태에 따라 배경 선택
        return (self.small_comp.copy() if self.inst_seg_var.get()==1 else self.small_rgb.copy())

    def _repaint(self):
        """현재 상태(self.grasp_num, show all/one, 선두께/색상)에 맞게
        small 이미지 위에 선을 그리고 Label에 표시"""
        img = self._base_image()
        if self.plt_bbox:
            if getattr(self, "_grasp_vis_mode", "one") == "all":
                # 전부 그림
                for segs in self.plt_bbox:
                    self._draw_segments(img, segs, self.bbox_line_width)
            else:
                # 현재 것만
                idx = max(0, min(self.grasp_num, len(self.plt_bbox)-1))
                self._draw_segments(img, self.plt_bbox[idx], self.bbox_line_width)

        self._photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self._photo)
        self.img_label.image = self._photo  # GC 방지



    def auto_eval_all(self):
        self.scene_num = -1
        for num in tqdm(range(self.img_sampling_num)):
            try:
                self.next()
                self.grasp_auto_eval()
            except:
                print(num)
        
            

    



            
if __name__ == "__main__":
    bbox_display(root_path, rgb_path, bbox_path, inst_seg_path, sampling_path_list)