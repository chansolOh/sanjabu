from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage
import omni.usd
import omni.kit.commands

import tkinter as tk
from tkinter import ttk, messagebox
import json
import csv
import os




class JSONClassEditor:
    def __init__(self, root, json_file_path):
        self.root = root
        self.root.title("JSON Class Editor")
        self.root.geometry("500x600")
        
        # ⭐ JSON 파일 경로를 여기에 직접 설정
        self.json_file_path = json_file_path
        self.json_data = None
        self.editing_item = None

        
        self.setup_ui()
        self.load_json_file()
        
    def setup_ui(self):
        # 파일 정보 표시
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(fill=tk.X)
        
        ttk.Label(info_frame, text="JSON 파일:").pack(side=tk.LEFT, padx=(0, 5))
        self.file_label = ttk.Label(info_frame, text="", relief=tk.SUNKEN)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 새로고침 버튼
        ttk.Button(info_frame, text="새로고침", command=self.load_json_file).pack(side=tk.LEFT)
        
        # 리스트박스 프레임
        list_frame = ttk.Frame(self.root, padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(list_frame, text="Class 목록 (더블클릭으로 편집):").pack(anchor=tk.W)
        
        # 스크롤바가 있는 리스트박스
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        # 더블클릭 이벤트 바인딩
        self.listbox.bind("<Double-Button-1>", self.on_double_click)
        
        # 편집 프레임 (초기에는 숨김)
        self.edit_frame = ttk.Frame(self.root, padding="10", relief=tk.RAISED, borderwidth=2)
        
        ttk.Label(self.edit_frame, text="새 이름:").grid(row=0, column=0, padx=5, pady=5)
        self.edit_entry = ttk.Entry(self.edit_frame, width=30)
        self.edit_entry.grid(row=0, column=1, padx=5, pady=5)
        
        button_frame = ttk.Frame(self.edit_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="저장", command=self.save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="취소", command=self.cancel_edit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="자동", command=self.auto_change).pack(side=tk.LEFT, padx=5)
        
        
        # Enter 키로도 저장 가능
        self.edit_entry.bind("<Return>", lambda e: self.save_changes())
        
        # 상태 표시줄
        self.status_bar = ttk.Label(self.root, text="준비", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_json_file(self):
        """JSON 파일을 불러오는 함수"""
        if not os.path.exists(self.json_file_path):
            messagebox.showerror("오류", f"파일을 찾을 수 없습니다:\n{self.json_file_path}")
            self.status_bar.config(text="파일 로드 실패")
            return
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            
            self.file_label.config(text=os.path.basename(self.json_file_path))
            self.populate_listbox()
            self.status_bar.config(text=f"파일 로드 완료: {os.path.basename(self.json_file_path)}")
            self.json_dict = {}
            for data in self.json_data:
                self.json_dict[data["name"]] = data
            
        except Exception as e:
            messagebox.showerror("오류", f"파일을 불러올 수 없습니다:\n{str(e)}")
            self.status_bar.config(text="파일 로드 실패")
    
    def populate_listbox(self):
        """JSON 리스트에서 각 항목의 class 값들을 리스트박스에 표시"""
        self.listbox.delete(0, tk.END)
        
        if self.json_data and isinstance(self.json_data, list):
            class_count = 0
            for item in self.json_data:
                if isinstance(item, dict) and "name" in item:
                    class_value = item["name"]
                    self.listbox.insert(tk.END, class_value)
                    class_count += 1
            
            if class_count == 0:
                messagebox.showwarning("경고", "리스트 항목들에 'class' 키가 없습니다.")
            else:
                self.status_bar.config(text=f"{class_count}개의 클래스 로드됨")
        else:
            messagebox.showwarning("경고", "JSON 파일이 리스트 형태가 아닙니다.")
    
    def on_double_click(self, event):
        """리스트 아이템 더블클릭 시 편집 모드 활성화"""
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            current_value = self.listbox.get(index)
            
            self.editing_item = index
            self.edit_entry.delete(0, tk.END)
            self.edit_entry.insert(0, current_value)
            
            # 편집 프레임 표시
            self.edit_frame.pack(fill=tk.X, before=self.status_bar)
            self.edit_entry.focus_select()
            
            self.status_bar.config(text=f"편집 중: {current_value}")
    
    def save_changes(self):
        """변경사항 저장"""
        if self.editing_item is not None:
            new_value = self.edit_entry.get().strip()
            
            if not new_value:
                messagebox.showwarning("경고", "빈 값은 입력할 수 없습니다.")
                return
            
            old_value = self.listbox.get(self.editing_item)
            changed = self.on_name_changed(old_value, new_value, self.editing_item)

            if changed:
                # 리스트박스 업데이트
                self.listbox.delete(self.editing_item)
                self.listbox.insert(self.editing_item, new_value)
                
                # JSON 데이터 업데이트
                if self.json_data and isinstance(self.json_data, list):
                    # 실제 JSON 리스트에서 해당 인덱스의 항목 찾기
                    list_index = 0
                    for i, item in enumerate(self.json_data):
                        if isinstance(item, dict) and "name" in item:
                            if list_index == self.editing_item:
                                # 해당 항목의 name 값 업데이트
                                self.json_data[i]["name"] = new_value
                                self.json_data[i]["path"] = item["path"].replace(old_value, new_value)
                                break
                            list_index += 1
                self.json_dict = {}
                for data in self.json_data:
                    self.json_dict[data["name"]] = data

                self.save_to_file()
            else:
                return
            
            # 편집 모드 종료
            self.cancel_edit()
            self.status_bar.config(text=f"변경 완료: {old_value} → {new_value}")

    def auto_change(self):
        self.json_file_path_dir = os.path.dirname(self.json_file_path)
        csv_file_path = [i for i in os.listdir(self.json_file_path_dir) if "objects_cat_attr" in i]
        csv_file_path = os.path.join(self.json_file_path_dir, csv_file_path[0])
        with open(csv_file_path, 'r') as f:
            csv_file = csv.DictReader(f)
            csv_file = list(csv_file)
        count = 0

        for item in csv_file:
            old_name = item["Class_name"]
            new_name = item["Object_name"]
            if old_name == new_name:
                continue
            if old_name not in self.json_dict:
                continue
            count+=1
            if count > 10:
                break

            self.on_name_changed(old_name, new_name, 0)
            for i, data in enumerate(self.json_data):
                if data["category"]["object_name"] == new_name:
                    self.json_data[i]["name"] = new_name
                    self.json_data[i]["path"] = data["path"].replace(old_name, new_name)
 

            self.json_dict = {}
            for data in self.json_data:
                self.json_dict[data["name"]] = data

            self.save_to_file()




    def cancel_edit(self):
        """편집 취소"""
        self.editing_item = None
        self.edit_frame.pack_forget()
        self.status_bar.config(text="준비")
    
    
    def save_to_file(self):
        """JSON 파일에 저장"""
        if self.json_file_path and self.json_data:
            try:
                with open(self.json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.json_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 실패:\n{str(e)}")
    
    def on_name_changed(self, old_name, new_name, index):

        usd_path = self.json_dict[old_name]["path"]
        usd_dir_path = usd_path.split("/edited")[0]
        



        # #### usd change
        if not change_usd(usd_path, old_name, new_name):
            return False
        usd_rigid_path = usd_path.split(".")[0] + "_rigid.usd"
        if not change_usd(usd_rigid_path, old_name, new_name):
            return False
        
        #### usd file change
        os.rename(usd_path, os.path.dirname(usd_path) + f"/{new_name}.usd")
        os.rename(usd_rigid_path, os.path.dirname(usd_rigid_path) + f"/{new_name}_rigid.usd")
        #### change 3d model
        change_3d_model(usd_dir_path, old_name, new_name)

        #### directory change
        os.rename(usd_dir_path, os.path.dirname(usd_dir_path) + f"/{new_name}")        
        #### data change




        change_data(old_name, new_name)

        for data in self.json_data:
                self.json_dict[data["name"]] = data

        print(f"[변경됨] 인덱스 {index}: '{old_name}' → '{new_name}'")
        return True


def modify_map_kd(mtl_file, old_name, new_name):
    with open(mtl_file, 'r') as file:
        lines = file.readlines()  # 모든 줄 읽기
    
    # 수정된 내용 저장
    with open(mtl_file, 'w') as file:
        for line in lines:
            # map_Kd가 포함된 줄을 찾음
            if "map_Kd" in line:
                file.write(line.replace(old_name,new_name))  # 새로운 텍스처 경로로 수정
            else:
                file.write(line)  # 다른 줄은 그대로 유지

def change_data(old_name, new_name):
    data_root_path ="/nas/Dataset/Dataset_2025/dataset_v1"

    
    for env_name in os.listdir(data_root_path):
        env_path = os.path.join(data_root_path, env_name)
        if not os.path.isdir(env_path):
            continue
        for section_name in os.listdir(env_path):
            section_path = os.path.join(env_path, section_name)
            if not os.path.isdir(section_path):
                continue
            for platform_name in os.listdir(section_path):
                platform_path = os.path.join(section_path, platform_name)
                if not os.path.isdir(platform_path):
                    continue
                target_dir_path = os.path.join(platform_path,"conf")
                for file_name in os.listdir(target_dir_path):
                    if file_name.endswith(".json"):
                        with open(os.path.join(target_dir_path, file_name), 'r') as f:
                            data = json.load(f)
                        exist_flag = False
                        for obj in data["objects"]:
                            if obj["class"] == old_name:
                                exist_flag = True
                        if not exist_flag:
                            continue

                        for idx, obj in enumerate(data["objects"]):
                            if obj["class"] == old_name:
                                data["objects"][idx]["class"] = new_name
                                data["objects"][idx]["usd_path"] = obj["usd_path"].replace(old_name, new_name)
                        with open(os.path.join(target_dir_path, file_name), 'w') as f:
                            json.dump(data, f, indent=4)


                        ##### bbox top
                        target_dir_path_sub = os.path.join(platform_path,"bbox","top_view_camera")
                        with open(os.path.join(target_dir_path_sub, file_name), 'r') as f:
                            data = json.load(f)
                        if old_name in data:
                            data[new_name] = data.pop(old_name)
                            with open(os.path.join(target_dir_path_sub, file_name), 'w') as f:
                                json.dump(data, f, indent=4)
                        
                        #### bbox side
                        target_dir_path_sub = os.path.join(platform_path,"bbox","side_view_camera")
                        with open(os.path.join(target_dir_path_sub, file_name), 'r') as f:
                            data = json.load(f)
                        if old_name in data:
                            data[new_name] = data.pop(old_name)
                            with open(os.path.join(target_dir_path_sub, file_name), 'w') as f:
                                json.dump(data, f, indent=4)
                        # print("bbox change complete : ", file_name)
                        
                        #### inst_seg top
                        target_dir_path_sub = os.path.join(platform_path,"inst_seg","top_view_camera")
                        with open(os.path.join(target_dir_path_sub, "semantics_mapping_"+file_name), 'r') as f:
                            data = json.load(f)
                        for key in data.keys():
                            if data[key]["class"] == old_name:
                                data[key]["class"] = new_name
                        with open(os.path.join(target_dir_path_sub, "semantics_mapping_"+file_name), 'w') as f:
                            json.dump(data, f, indent=4)

                        #### inst_seg side
                        target_dir_path_sub = os.path.join(platform_path,"inst_seg","side_view_camera")
                        with open(os.path.join(target_dir_path_sub, "semantics_mapping_"+file_name), 'r') as f:
                            data = json.load(f)
                        for key in data.keys():
                            if data[key]["class"] == old_name:
                                data[key]["class"] = new_name
                        with open(os.path.join(target_dir_path_sub, "semantics_mapping_"+file_name), 'w') as f:
                            json.dump(data, f, indent=4)
                        # print("inst_seg change complete : ", file_name)

                        #### pregrasp
                        target_dir_path_sub = os.path.join(platform_path,"pre_grasp")
                        if os.path.exists(os.path.join(target_dir_path_sub, file_name)):
                            with open(os.path.join(target_dir_path_sub, file_name), 'r') as f:
                                data = json.load(f)
                            
                            for idx,dd in enumerate(data):
                                for i,d in enumerate(dd["data"]):
                                    if d["target_object"] == old_name:
                                        data[idx]["data"][i]["target_object"] = new_name
                            with open(os.path.join(target_dir_path_sub, file_name), 'w') as f:
                                json.dump(data, f, indent=4)
                            
                        
                        #### grasp
                        target_dir_path_sub = os.path.join(platform_path,"output_grasp")
                        if os.path.exists(os.path.join(target_dir_path_sub, file_name)):
                            with open(os.path.join(target_dir_path_sub, file_name), 'r') as f:
                                data = json.load(f)
                            for idx,dd in enumerate(data):
                                if dd["target_object"] == old_name:
                                    data[idx]["target_object"] = new_name
                            with open(os.path.join(target_dir_path_sub, file_name), 'w') as f:
                                json.dump(data, f, indent=4)
                        print(f"{env_name}/{section_name}/{[platform_name]} : ", file_name)
                        
                        

                                    

def change_3d_model(usd_dir_path, old_name, new_name):
    target_path = usd_dir_path
    target = [i for i in os.listdir(target_path) if i.split(".")[-1] in ["bmp","mtl","obj"]]
    for t in target:
        if t.endswith(".mtl"):
            mtl_file_path = os.path.join(target_path, t)
            modify_map_kd(mtl_file_path, old_name, new_name)
        elif t.endswith(".obj"):
            obj_file_path = os.path.join(target_path, t)

            with open(obj_file_path, 'r') as file:
                lines = file.readlines()  # 모든 줄 읽기

            for i, line in enumerate(lines):
                if line.startswith('g '):
                    lines[i] = line.replace(old_name, new_name)  # 그룹 이름 수정
                elif line.startswith('mtllib'):
                    lines[i] = line.replace(old_name, new_name)  # mtllib 줄 수정
                elif line.startswith('o '):
                    lines[i] = line.replace(old_name, new_name)

            # 3. 수정된 전체 리스트를 파일로 다시 쓰기
            with open(obj_file_path, 'w') as f:
                f.writelines(lines)

        os.rename(os.path.join(target_path, t),os.path.join(target_path, t.replace(old_name, new_name)))

    

    target_path = os.path.join(usd_dir_path, "edited") 
    target = [i for i in os.listdir(target_path) if i.split(".")[-1] in ["bmp","mtl","obj"]]
    for t in target:
        if t.endswith(".mtl"):
            mtl_file_path = os.path.join(target_path, t)
            modify_map_kd(mtl_file_path, old_name, new_name)
        elif t.endswith(".obj"):
            obj_file_path = os.path.join(target_path, t)

            with open(obj_file_path, 'r') as file:
                lines = file.readlines()  # 모든 줄 읽기

            for i, line in enumerate(lines):
                if line.startswith('g '):
                    lines[i] = line.replace(old_name, new_name)  # 그룹 이름 수정
                elif line.startswith('mtllib'):
                    lines[i] = line.replace(old_name, new_name)  # mtllib 줄 수정

            # 3. 수정된 전체 리스트를 파일로 다시 쓰기
            with open(obj_file_path, 'w') as f:
                f.writelines(lines)

        os.rename(os.path.join(target_path, t),os.path.join(target_path, t.replace(old_name, new_name)))


def change_usd(usd_path, old_name, new_name):
    if old_name == new_name:
        return False
    try:
        open_stage(usd_path)
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath(f"/World")
        shad = find_targets(world_prim,["Shader"])
        for sh in shad:
            if sh.HasAttribute("inputs:diffuse_texture"):
                attr = sh.GetAttribute("inputs:diffuse_texture")
                if old_name in attr.Get().path: 
                    attr.Set(attr.Get().path.replace(f"{old_name}", f"{new_name}"))
 
        while True:
            all_prims = find_targets(world_prim,["Xform"])
            all_prims = [ i for i in all_prims if old_name == i.GetName()]
            if len(all_prims) == 0:
                break
            root_dir = os.path.dirname(str(all_prims[0].GetPath()))

            omni.kit.commands.execute('MovePrims',
                paths_to_move={str(all_prims[0].GetPath()): os.path.join(root_dir,new_name) },
                keep_world_transform=False,
                destructive=False)
            
        stage.GetRootLayer().Save()
        # stage.Flatten().Save()


        return True
    except:
        return False


def find_targets(prims, target_list):
    ls = [prims]
    result_ls = []
    cnt = 0
    while len(ls)>cnt:
        prim = ls[cnt]
        if prim.GetTypeName() in target_list:
            result_ls.append(prim)

        child = prim.GetAllChildren()
        if len(child)==0:
            cnt+=1
            continue
        [ls.append(ch) for ch in child]
        cnt+=1
    return result_ls


def main():
    """메인 함수"""
    # ========================================
    # ⭐ 여기에 JSON 파일 경로를 직접 입력하세요!
    # ========================================
    JSON_FILE_PATH = "/nas/ochansol/3d_model/peel3_scan_data_2025/objects_conf.json"  # <- 여기를 수정하세요!
    
    # 테스트용 샘플 파일이 필요한 경우 아래 주석을 해제하세요
    # create_sample_file(JSON_FILE_PATH)
    
    # import pdb;pdb.set_trace()
    root = tk.Tk()
    app = JSONClassEditor(root, JSON_FILE_PATH)
    root.mainloop()





if __name__ == "__main__":
    main()

