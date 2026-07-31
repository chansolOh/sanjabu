import numpy as np
import json 
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg , NavigationToolbar2Tk
from PIL import Image, ImageTk
import os
import tkinter as tk
import ast

root_path = "data"
# root_path = "/home/nia/ochansol/isaac/sanjabu/data/"
sampling_num = 10
all_data_num = 10



###### data path ######
dir_path = "scene/Home" 
rgb_path = os.path.join(root_path, dir_path, "rgb/cam_00")
inst_seg_path  = os.path.join(root_path, dir_path, "inst_seg/cam_00")



class inst_seg_display():
    def __init__(self, rgb_path, inst_seg_path, sampling_num, all_data_num):
        
        self.rgb_path = rgb_path
        self.inst_seg_path = inst_seg_path
        self.sampling_num = sampling_num
        self.all_data_num = all_data_num
        
        self.sample_num_list = np.random.choice(range(all_data_num),size = sampling_num, replace=False)
        self.acc_dict = {}
        for sample_num in self.sample_num_list:
            self.acc_dict[sample_num] = None


        
        
        
        self.window = tk.Tk()
        self.window.title("Instance Segmentation")
        self.window.geometry("1000x1300")

        
        
        ####### plt #######
        self.fig, self.ax = plt.subplots(figsize = (14,7))
        self.fig.tight_layout()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        
        
        
        ##### Frame #####
        self.img_frame = tk.Frame(self.window)
        self.img_frame.pack(side=tk.TOP, pady=(0,30))
        
        self.eval_frame = tk.Frame(self.window)
        self.eval_frame.pack(side=tk.TOP,pady=30)
        
        self.eval_top_frame = tk.Frame(self.eval_frame)
        self.eval_top_frame.pack(side=tk.TOP)
        self.eval_L_frame = tk.Frame(self.eval_frame)
        self.eval_L_frame.pack(side=tk.LEFT)
        self.eval_R_frame = tk.Frame(self.eval_frame)
        self.eval_R_frame.pack(side=tk.RIGHT)
        
        self.move_frame = tk.Frame(self.window)
        self.move_frame.pack(side=tk.TOP)
        
        self.scene_frame = tk.Frame(self.move_frame)
        self.scene_frame.pack(side=tk.BOTTOM)
    
        
        
        ######  layout ######
        
        self.file_name_label = tk.Label(self.img_frame,  font = ("Arial",20), text="file_name")
        self.file_name_label.pack(side=tk.TOP)
        
        self.can_plt = FigureCanvasTkAgg(self.fig,master = self.img_frame)
        self.can_plt.draw()
        self.can_plt.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.plt_scatter = []
        
        self.toolbar = NavigationToolbar2Tk(self.can_plt, self.img_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.slider = tk.Scale(self.img_frame, orient=tk.HORIZONTAL, width = 30, length=400, sliderlength=100, from_=0, to=100, command=self.slider_move)
        self.slider.pack(side=tk.TOP)
        self.slider_label = tk.Label(self.img_frame, font = ("Arial",20), text="Alpha")
        self.slider_label.pack(side=tk.TOP)
        
        self.class_name_var = tk.IntVar()
        self.class_name_vis_check = tk.Checkbutton(self.img_frame, width = 20,text="Viz Class Name", font = ("Arial",20), variable=self.class_name_var ,command=self.class_name_vis)
        self.class_name_vis_check.pack(side=tk.TOP)
        
        self.acc_label = tk.Label(self.eval_top_frame, font = ("Arial",20,"bold"),fg="#FF0000", text="Accuracy")
        self.acc_label.pack(side=tk.TOP)
        
        self.correct_label = tk.Label(self.eval_L_frame, font = ("Arial",20), text="Correct")
        self.correct_label.pack(side=tk.BOTTOM)
        self.incorrect_label = tk.Label(self.eval_R_frame, font = ("Arial",20), text="Incorrect")
        self.incorrect_label.pack(side=tk.BOTTOM)
        
        self.correct_btn = tk.Button(self.eval_L_frame, width = 3, text="O", font = ("Arial",80), command=self.correct, bg="#FFFFFF" , activebackground="#50FF50")
        self.correct_btn.pack(side=tk.TOP)
        self.incorrect_btn = tk.Button(self.eval_R_frame, width = 3, text="X", font = ("Arial",80),command=self.incorrect, bg="#FFFFFF", activebackground="#FF5050")
        self.incorrect_btn.pack(side=tk.TOP)
        

        self.prev_btn = tk.Button(self.move_frame, width =10, font = ("Arial",30), text="Prev", command=self.prev)
        self.prev_btn.pack(side=tk.LEFT)
        self.next_btn = tk.Button(self.move_frame, width = 10,font = ("Arial",30), text="Next", command=self.next)
        self.next_btn.pack(side=tk.RIGHT)
        
        self.scene_num_label = tk.Label(self.scene_frame, font = ("Arial",22), text="Scene Num")
        self.scene_num_label.pack(side=tk.LEFT, padx=10)
        self.scene_entry = tk.Entry(self.scene_frame, width=5, font = ("Arial",20))
        self.scene_entry.pack(side=tk.LEFT)
        self.scene_entry.bind("<Return>", self.go_to_scene)
        self.scene_entry.bind("<KP_Enter>", self.go_to_scene)
        self.scene_change_btn = tk.Button(self.scene_frame, width = 3, font = ("Arial",20), text="Go", command= lambda : self.go_to_scene(None))
        self.scene_change_btn.pack(side=tk.RIGHT)
        
        self.window.bind("<Button-1>", lambda event: event.widget.focus_set())
        self.window.bind("<Left>", lambda x:self.prev() if self.window.focus_get() != self.scene_entry else None)
        self.window.bind("<Right>", lambda x: self.next() if self.window.focus_get() != self.scene_entry else None)
        self.window.bind("<space>", lambda x: self.class_name_vis_space() )


        ###### init ######
        self.scene_num = -1
        self.inst_seg_alpha = 0.3
        self.acc_label.config(text=f"Accuracy : {self.evaluate()}")
        
        self.window.mainloop()
        
        
        
        
    def next(self):
        if self.scene_num<self.sampling_num:
            self.scene_num += 1
            self.data_load(self.sample_num_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.sampling_num}")
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, f"{self.scene_num+1}")
            self.can_plt.draw()
            self.eval_btn_update()

    def prev(self):
        if self.scene_num>0:
            self.scene_num -= 1
            self.data_load(self.sample_num_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.sampling_num}")
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, f"{self.scene_num+1}")
            self.can_plt.draw()
            self.eval_btn_update()

    def correct(self):
        if self.scene_num != -1:
            self.acc_dict[self.sample_num_list[self.scene_num]] = True
            self.acc_label.config(text=f"Accuracy : {round(self.evaluate(),3)}")
            self.eval_btn_update()


    def incorrect(self):
        if self.scene_num != -1:
            self.acc_dict[self.sample_num_list[self.scene_num]] = False
            self.acc_label.config(text=f"Accuracy : {round(self.evaluate(),3)}")
            self.eval_btn_update()
    
    def eval_btn_update(self):
        self.correct_btn.config(bg="#FFFFFF")
        self.incorrect_btn.config(bg="#FFFFFF")
        if self.acc_dict[self.sample_num_list[self.scene_num]] == True:
            self.correct_btn.config(bg="#00FF00")
        elif self.acc_dict[self.sample_num_list[self.scene_num]] == False:
            self.incorrect_btn.config(bg="#FF0000")


    
    def data_load(self, scene_num):
        self.file_name_label.config(text=f"File Name : {scene_num:04d}.png")
        self.rgb_img = Image.open(os.path.join(self.rgb_path, f"{scene_num:04d}.png"))
        self.inst_seg_img = Image.open(os.path.join(self.inst_seg_path, f"{scene_num:04d}.png"))
        with open(os.path.join(self.inst_seg_path, f"semantics_mapping_{scene_num:04d}.json")) as f:
            self.inst_seg_label = json.load(f)
            
        self.ax.cla()
        self.ax.imshow(self.rgb_img)
        self.plt_scatter = []
        self.plt_class = []
        self.plt_legend = []
        self.inst_seg_arr = np.array(self.inst_seg_img)
        for key in self.inst_seg_label.keys():
            if self.inst_seg_label[key]["class"] not in ["BACKGROUND", "UNLABELLED"]:
                rgba = ast.literal_eval(key)
                idx = np.where(np.all(self.inst_seg_arr==np.array(rgba) ,axis=2) == True)
                self.plt_scatter += [self.ax.scatter(idx[1], idx[0], color='#{:02X}{:02X}{:02X}'.format(*(rgba[:3])), s=0.01, alpha=self.inst_seg_alpha, label=self.inst_seg_label[key]["class"] )]
                xmin,ymin = min(idx[1]), min(idx[0])
                tmp = self.ax.text(xmin+15, ymin+30, self.inst_seg_label[key]["class"], fontsize=15, color='#{:02X}{:02X}{:02X}'.format(*(rgba[:3])) )
                tmp.set_path_effects([path_effects.withStroke(linewidth=3, foreground="black" )])
                self.plt_class+=[tmp]
                self.plt_legend+=[self.ax.legend(loc="upper right", scatterpoints=100)]
        for plt_leg in self.plt_legend:
            plt_leg.set_alpha(1)
        
    
    def slider_move(self, val):
        if self.plt_scatter is not []:
            self.inst_seg_alpha = float(val)/100
            for scatter in self.plt_scatter:
                scatter.set_alpha(self.inst_seg_alpha)
            self.can_plt.draw()
    
    def evaluate(self):
        correct = 0
        incorrect = 0
        for key in self.acc_dict.keys():
            if self.acc_dict[key] == True:
                correct += 1
            elif self.acc_dict[key] == False:
                incorrect += 1
        print(correct, incorrect)
        self.correct_label.config(text=f"Correct : {correct}")
        self.incorrect_label.config(text=f"Incorrect : {incorrect}")
        
        if correct==0:
            return 0
        return correct/(correct+incorrect)
    
    def go_to_scene(self, event):
        if int(self.scene_entry.get()) <= self.sampling_num and int(self.scene_entry.get()) > 0:
            self.scene_num = int(self.scene_entry.get())-1
            self.data_load(self.sample_num_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.sampling_num}")
            self.can_plt.draw()
            self.eval_btn_update()
            
    def class_name_vis(self):
        if self.class_name_var.get() == 1:
            self.class_name_vis_check.config(bg="yellow")
            self.class_name_vis_check.config(font = ("Arial",20,"bold"))
            for plt_cls in self.plt_class:
                plt_cls.set_visible(True)
            for plt_leg in self.plt_legend:
                plt_leg.set_visible(True)
            self.can_plt.draw()
            self.class_name_var.set(1)
        else:
            self.class_name_vis_check.config(bg="white")
            self.class_name_vis_check.config(font = ("Arial",20))
            for plt_cls in self.plt_class:
                plt_cls.set_visible(False)
            for plt_leg in self.plt_legend:
                plt_leg.set_visible(False)
            self.can_plt.draw()
            self.class_name_var.set(0)
    def class_name_vis_space(self):
        self.class_name_var.set(1) if self.class_name_var.get() == 0 else self.class_name_var.set(0)
        self.class_name_vis()

inst_seg_display(rgb_path, inst_seg_path, sampling_num, all_data_num)