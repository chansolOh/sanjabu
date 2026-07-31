import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.figure

class DataMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Structure Monitor")
        self.root.geometry("1000x700")
        
        # 모니터링 경로와 업데이트 스레드
        self.monitor_path = ""
        self.monitoring = False
        self.update_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 경로 선택 프레임
        path_frame = ttk.Frame(main_frame)
        path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(path_frame, text="Monitor Path:").grid(row=0, column=0, sticky=tk.W)
        self.path_var = tk.StringVar(value="/nas/Dataset/Dataset_2025/dataset_v1")  # 기본 경로 설정
        self.path_entry = ttk.Entry(path_frame, textvariable=self.path_var, width=50)
        self.path_entry.grid(row=0, column=1, padx=(10, 10), sticky=(tk.W, tk.E))
        
        ttk.Button(path_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2)
        self.start_btn = ttk.Button(path_frame, text="Start Monitor", command=self.toggle_monitoring)
        self.start_btn.grid(row=0, column=3, padx=(10, 0))
        
        # 상태 표시
        self.status_label = ttk.Label(main_frame, text="Status: Not monitoring", foreground="red")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # 트리뷰 프레임
        tree_frame = ttk.Frame(main_frame)
        tree_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 트리뷰 생성
        self.tree = ttk.Treeview(tree_frame, columns=('Type', 'Range', 'Missing'), show='tree headings')
        self.tree.heading('#0', text='Structure')
        self.tree.heading('Type', text='Type')
        self.tree.heading('Range', text='Data Range')
        self.tree.heading('Missing', text='Missing Files')
        
        # 컬럼 너비 설정
        self.tree.column('#0', width=300)
        self.tree.column('Type', width=100)
        self.tree.column('Range', width=150)
        self.tree.column('Missing', width=200)
        
        # 스크롤바
        scrollbar_v = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_h = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # 그리드 배치
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_v.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_h.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 상세 정보 프레임
        detail_frame = ttk.LabelFrame(main_frame, text="Details", padding="10")
        detail_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.detail_text = tk.Text(detail_frame, height=8, wrap=tk.WORD)
        detail_scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scrollbar.set)
        
        self.detail_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        detail_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 그리드 가중치 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        path_frame.columnconfigure(1, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        
        # 트리 선택 이벤트
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        self.tree.bind('<Double-1>', self.on_tree_double_click)
        
    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_var.set(folder)
            
    def toggle_monitoring(self):
        if not self.monitoring:
            path = self.path_var.get().strip()
            if not path or not os.path.exists(path):
                messagebox.showerror("Error", "Please select a valid path")
                return
            
            self.monitor_path = path
            self.monitoring = True
            self.start_btn.config(text="Stop Monitor")
            self.status_label.config(text="Status: Monitoring...", foreground="green")
            
            # 모니터링 스레드 시작
            self.update_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.update_thread.start()
            
        else:
            self.monitoring = False
            self.start_btn.config(text="Start Monitor")
            self.status_label.config(text="Status: Not monitoring", foreground="red")
            
    def monitor_loop(self):
        while self.monitoring:
            try:
                self.update_tree()
                time.sleep(2)  # 2초마다 업데이트
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)
                
    def update_tree(self):
        self.root.after(0, self._update_tree_ui)
        
    def _update_tree_ui(self):
        # 현재 펼침/접힘 상태 저장
        expanded_items = self.get_expanded_state()
        
        # 트리 초기화
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if not os.path.exists(self.monitor_path):
            return
            
        # 데이터 구조 스캔
        data_structure = self.scan_structure(self.monitor_path)
        
        # 트리에 데이터 추가
        self.populate_tree(data_structure)
        
        # 이전 상태 복원
        self.restore_expanded_state(expanded_items)
        
    def scan_structure(self, base_path):
        structure = {}
        # 제외할 platform 목록
        excluded_platforms = {'rgb', 'depth', 'inst_seg', 'normals', 'bbox', 'pointcloud'}
        
        try:
            for env in os.listdir(base_path):
                env_path = os.path.join(base_path, env)
                if not os.path.isdir(env_path):
                    continue
                    
                structure[env] = {}
                
                for section in os.listdir(env_path):
                    section_path = os.path.join(env_path, section)
                    if not os.path.isdir(section_path):
                        continue
                        
                    structure[env][section] = {}
                    
                    for platform in os.listdir(section_path):
                        # 제외할 platform은 건너뛰기
                        if platform in excluded_platforms:
                            continue
                            
                        platform_path = os.path.join(section_path, platform)
                        if not os.path.isdir(platform_path):
                            continue
                            
                        conf_path = os.path.join(platform_path, 'conf')
                        if os.path.exists(conf_path) and os.path.isdir(conf_path):
                            json_files = self.scan_json_files(conf_path)
                            structure[env][section][platform] = json_files
                        else:
                            structure[env][section][platform] = {'files': [], 'range': 'No conf folder', 'missing': []}
                            
        except Exception as e:
            print(f"Scan error: {e}")
            
        return structure
        
    def scan_json_files(self, conf_path):
        files = []
        try:
            for file in os.listdir(conf_path):
                if file.endswith('.json'):
                    try:
                        # 파일명에서 번호 추출 (예: 0000.json -> 0)
                        num = int(file.split('.')[0])
                        files.append(num)
                    except ValueError:
                        continue
        except Exception as e:
            print(f"JSON scan error: {e}")
            
        files.sort()
        
        if not files:
            return {'files': [], 'range': 'No JSON files', 'missing': []}
            
        # 범위와 누락 파일 계산
        min_num = min(files)
        max_num = max(files)
        expected = set(range(min_num, max_num + 1))
        actual = set(files)
        missing = sorted(list(expected - actual))
        
        range_str = f"{min_num:04d} - {max_num:04d}" if min_num != max_num else f"{min_num:04d}"
        
        return {
            'files': files,
            'range': range_str,
            'missing': missing
        }
        
    def populate_tree(self, structure):
        for env, sections in structure.items():
            env_item = self.tree.insert('', 'end', text=env, values=('Environment', '', ''))
            
            for section, platforms in sections.items():
                section_item = self.tree.insert(env_item, 'end', text=section, values=('Section', '', ''))
                
                for platform, data in platforms.items():
                    missing_str = ', '.join([f"{m:04d}" for m in data['missing']]) if data['missing'] else 'None'
                    platform_item = self.tree.insert(section_item, 'end', text=platform, 
                                                   values=('Platform', data['range'], missing_str))
        
        # 첫 실행시에만 모든 노드 확장
        if not hasattr(self, '_first_run_done'):
            self.expand_all(self.tree.get_children())
            self._first_run_done = True
                    
    def get_expanded_state(self):
        """현재 트리의 펼침 상태를 저장"""
        expanded = {}
        
        def collect_state(items, path=""):
            for item in items:
                item_text = self.tree.item(item, 'text')
                current_path = f"{path}/{item_text}" if path else item_text
                expanded[current_path] = self.tree.item(item, 'open')
                collect_state(self.tree.get_children(item), current_path)
        
        collect_state(self.tree.get_children())
        return expanded
    
    def restore_expanded_state(self, expanded_state):
        """저장된 펼침 상태를 복원"""
        def restore_state(items, path=""):
            for item in items:
                item_text = self.tree.item(item, 'text')
                current_path = f"{path}/{item_text}" if path else item_text
                
                if current_path in expanded_state:
                    self.tree.item(item, open=expanded_state[current_path])
                
                restore_state(self.tree.get_children(item), current_path)
        
        restore_state(self.tree.get_children())
        
    def expand_all(self, items):
        for item in items:
            self.tree.item(item, open=True)
            self.expand_all(self.tree.get_children(item))
            
    def on_tree_double_click(self, event):
        """트리 아이템 더블클릭 이벤트"""
        selection = self.tree.selection()
        if not selection:
            return
            
        item = selection[0]
        values = self.tree.item(item, 'values')
        
        # Platform 타입인 경우에만 이미지 뷰어 열기
        if len(values) > 0 and values[0] == 'Platform':
            self.open_image_viewer(item)
    
    def open_image_viewer(self, tree_item):
        """이미지 뷰어 창 열기"""
        # 경로 구성
        path_parts = []
        current = tree_item
        while current:
            path_parts.append(self.tree.item(current, 'text'))
            current = self.tree.parent(current)
            
        path_parts.reverse()
        
        if len(path_parts) >= 3:  # env/section/platform
            env, section, platform = path_parts[:3]
            platform_path = os.path.join(self.monitor_path, env, section, platform)
            
            # 이미지 뷰어 창 생성
            ImageViewerWindow(self.root, platform_path)
    
    def on_tree_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
            
        item = selection[0]
        values = self.tree.item(item, 'values')
        text = self.tree.item(item, 'text')
        
        # 선택된 항목의 상세 정보 표시
        detail_info = f"Selected: {text}\n"
        detail_info += f"Type: {values[0]}\n"
        
        if len(values) > 1 and values[1]:
            detail_info += f"Data Range: {values[1]}\n"
            
        if len(values) > 2 and values[2] and values[2] != 'None':
            detail_info += f"Missing Files: {values[2]}\n"
            
        # 전체 경로 표시
        path_parts = []
        current = item
        while current:
            path_parts.append(self.tree.item(current, 'text'))
            current = self.tree.parent(current)
            
        path_parts.reverse()
        if len(path_parts) > 1:
            full_path = '/'.join(path_parts)
            detail_info += f"Path: {full_path}\n"
            
        # Platform인 경우 더블클릭 안내 추가
        if len(values) > 0 and values[0] == 'Platform':
            detail_info += "\nDouble-click to view images\n"
            
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, detail_info)


class ImageViewerWindow:
    def __init__(self, parent, platform_path):
        self.parent = parent
        self.platform_path = platform_path
        
        # 새 창 생성
        self.window = tk.Toplevel(parent)
        self.window.title(f"Image Viewer - {os.path.basename(platform_path)}")
        self.window.geometry("800x650")
        
        # 이미지 타입 옵션
        self.image_types = ['rgb', 'depth', 'inst_seg']
        
        self.current_image_index = 0
        self.available_indices = []
        
        self.setup_ui()
        self.load_available_indices()
        
        # 키보드 이벤트 바인딩
        self.window.bind('<Key>', self.on_key_press)
        self.window.focus_set()  # 포커스 설정
        
    def setup_ui(self):
        # 컨트롤 프레임
        control_frame = ttk.Frame(self.window, padding="10")
        control_frame.pack(fill=tk.X)
        
        # 이미지 타입 선택
        ttk.Label(control_frame, text="Image Type:").grid(row=0, column=0, sticky=tk.W)
        self.image_type_var = tk.StringVar(value='rgb')
        image_type_combo = ttk.Combobox(control_frame, textvariable=self.image_type_var, 
                                       values=self.image_types, state='readonly')
        image_type_combo.grid(row=0, column=1, padx=(5, 20))
        image_type_combo.bind('<<ComboboxSelected>>', self.on_type_change)
        
        # 이미지 인덱스 네비게이션
        ttk.Label(control_frame, text="Index:").grid(row=0, column=2, sticky=tk.W)
        self.prev_btn = ttk.Button(control_frame, text="◀", command=self.prev_image)
        self.prev_btn.grid(row=0, column=3, padx=(5, 2))
        
        # 인덱스 직접 입력
        self.index_entry_var = tk.StringVar()
        self.index_entry = ttk.Entry(control_frame, textvariable=self.index_entry_var, width=8)
        self.index_entry.grid(row=0, column=4, padx=2)
        self.index_entry.bind('<Return>', self.on_index_entry)
        self.index_entry.bind('<KeyRelease>', self.on_index_key)
        
        ttk.Label(control_frame, text="/").grid(row=0, column=5, padx=2)
        
        self.total_count_var = tk.StringVar()
        self.total_count_label = ttk.Label(control_frame, textvariable=self.total_count_var)
        self.total_count_label.grid(row=0, column=6, padx=2)
        
        self.next_btn = ttk.Button(control_frame, text="▶", command=self.next_image)
        self.next_btn.grid(row=0, column=7, padx=(2, 20))
        
        # Go 버튼
        self.go_btn = ttk.Button(control_frame, text="Go", command=self.go_to_index)
        self.go_btn.grid(row=0, column=8, padx=(2, 20))
        
        # 새로고침 버튼
        ttk.Button(control_frame, text="Refresh", command=self.refresh_data).grid(row=0, column=9)
        
        # 이미지 표시 프레임 (1행 2열)
        self.image_frame = ttk.Frame(self.window)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Side view 프레임
        self.side_frame = ttk.LabelFrame(self.image_frame, text="Side View Camera", padding="5")
        self.side_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Top view 프레임
        self.top_frame = ttk.LabelFrame(self.image_frame, text="Top View Camera", padding="5")
        self.top_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # 물체 정보 프레임
        self.objects_frame = ttk.LabelFrame(self.window, text="Objects Information", padding="5")
        self.objects_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 그리드 가중치 설정
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        # Side view용 matplotlib figure와 canvas 생성
        self.side_fig = matplotlib.figure.Figure(figsize=(4, 3), dpi=80, tight_layout=True)
        self.side_ax = self.side_fig.add_subplot(111)
        self.side_canvas = FigureCanvasTkAgg(self.side_fig, self.side_frame)
        self.side_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Side view 툴바 (좌표 표시 비활성화)
        self.side_toolbar = NavigationToolbar2Tk(self.side_canvas, self.side_frame)
        self.side_toolbar.update()
        # 좌표 표시 기능 비활성화
        self.side_canvas.mpl_connect('motion_notify_event', lambda event: None)
        
        # Top view용 matplotlib figure와 canvas 생성
        self.top_fig = matplotlib.figure.Figure(figsize=(4, 3), dpi=80, tight_layout=True)
        self.top_ax = self.top_fig.add_subplot(111)
        self.top_canvas = FigureCanvasTkAgg(self.top_fig, self.top_frame)
        self.top_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Top view 툴바 (좌표 표시 비활성화)
        self.top_toolbar = NavigationToolbar2Tk(self.top_canvas, self.top_frame)
        self.top_toolbar.update()
        # 좌표 표시 기능 비활성화
        self.top_canvas.mpl_connect('motion_notify_event', lambda event: None)
        
        # 물체 정보 표시 라벨
        self.objects_label = ttk.Label(self.objects_frame, text="No data loaded", foreground="gray")
        self.objects_label.pack(anchor=tk.W)
        
    def load_available_indices(self):
        """사용 가능한 이미지 인덱스 로드"""
        try:
            conf_path = os.path.join(self.platform_path, 'conf')
            
            indices = set()
            if os.path.exists(conf_path):
                for file in os.listdir(conf_path):
                    if file.endswith('.json'):
                        try:
                            index = int(file.split('.')[0])
                            indices.add(index)
                        except ValueError:
                            continue
            
            self.available_indices = sorted(list(indices))
            
            if self.available_indices:
                # 현재 인덱스가 유효한지 확인
                if not hasattr(self, 'current_image_index') or self.current_image_index >= len(self.available_indices):
                    self.current_image_index = 0
                self.update_display()
            else:
                self.show_no_data()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load indices: {e}")
            
    def refresh_data(self):
        """현재 인덱스를 유지하면서 데이터 새로고침"""
        current_actual_index = None
        if self.available_indices and hasattr(self, 'current_image_index') and self.current_image_index < len(self.available_indices):
            current_actual_index = self.available_indices[self.current_image_index]
        
        # 데이터 다시 로드
        self.load_available_indices()
        
        # 이전 인덱스 복원 시도
        if current_actual_index is not None and current_actual_index in self.available_indices:
            self.current_image_index = self.available_indices.index(current_actual_index)
            self.update_display()
        
    def on_key_press(self, event):
        """키보드 이벤트 처리"""
        if event.keysym == 'Left':
            self.prev_image()
        elif event.keysym == 'Right':
            self.next_image()
        elif event.keysym == 'space':
            self.next_image()
        elif event.keysym == 'BackSpace':
            self.prev_image()
    
    def on_index_entry(self, event):
        """엔터키로 인덱스 이동"""
        self.go_to_index()
        
    def on_index_key(self, event):
        """입력 중 실시간 유효성 검사 - 수정된 버전"""
        try:
            value = self.index_entry_var.get().strip()
            if not value:
                self.index_entry.config(foreground='black')
                return
                
            # 숫자만 입력되었는지 확인
            if not value.isdigit():
                self.index_entry.config(foreground='red')
                return
                
            # 4자리 이하인 경우 (입력 중)에는 색상 변경하지 않음
            if len(value) < 4:
                self.index_entry.config(foreground='black')
                return
                
            # 4자리 이상인 경우에만 유효성 검사
            index_num = int(value)
            if index_num in self.available_indices:
                self.index_entry.config(foreground='green')
            else:
                self.index_entry.config(foreground='orange')  # 빨간색 대신 주황색으로 경고
                
        except ValueError:
            self.index_entry.config(foreground='red')
            
    def go_to_index(self):
        """입력된 인덱스로 이동"""
        try:
            target_index = int(self.index_entry_var.get())
            if target_index in self.available_indices:
                self.current_image_index = self.available_indices.index(target_index)
                self.update_display()
                self.index_entry.config(foreground='green')
            else:
                self.index_entry.config(foreground='red')
                # 사용 가능한 인덱스 범위 정보 제공
                if self.available_indices:
                    min_idx = min(self.available_indices)
                    max_idx = max(self.available_indices)
                    messagebox.showwarning("Invalid Index", 
                        f"Index {target_index:04d} not found.\n"
                        f"Available range: {min_idx:04d} ~ {max_idx:04d}\n"
                        f"Total {len(self.available_indices)} files available.")
                else:
                    messagebox.showwarning("Invalid Index", "No data available")
        except ValueError:
            self.index_entry.config(foreground='red')
            messagebox.showwarning("Invalid Input", "Please enter a valid number")
            
    def on_type_change(self, event=None):
        """이미지/뷰 타입 변경시 호출"""
        self.update_display()
        
    def prev_image(self):
        """이전 이미지"""
        if self.available_indices and self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_display()
            
    def next_image(self):
        """다음 이미지"""
        if self.available_indices and self.current_image_index < len(self.available_indices) - 1:
            self.current_image_index += 1
            self.update_display()
            
    def update_display(self):
        """이미지 표시 업데이트"""
        if not self.available_indices:
            self.show_no_data()
            return
            
        current_index = self.available_indices[self.current_image_index]
        
        # 인덱스 정보 업데이트
        self.index_entry_var.set(f"{current_index:04d}")
        self.total_count_var.set(f"{len(self.available_indices)}")
        self.index_entry.config(foreground='green')
        
        # 버튼 상태 업데이트
        self.prev_btn.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_image_index < len(self.available_indices) - 1 else tk.DISABLED)
        
        # 이미지 로드 및 표시
        self.load_and_display_image(current_index)
        
        # 물체 정보 로드 및 표시
        self.load_and_display_objects(current_index)
        
    def load_and_display_image(self, index):
        """side와 top 이미지를 동시에 로드 및 표시"""
        try:
            image_type = self.image_type_var.get()
            
            # Side view 이미지 로드
            side_path = os.path.join(self.platform_path, image_type, 'side_view_camera', f"{index:04d}.png" if image_type != 'depth' else f"{index:04d}.npy")
            self.load_image_to_axes(side_path, self.side_ax, self.side_canvas, image_type, "Side View")
            
            # Top view 이미지 로드  
            top_path = os.path.join(self.platform_path, image_type, 'top_view_camera', f"{index:04d}.png" if image_type != 'depth' else f"{index:04d}.npy")
            self.load_image_to_axes(top_path, self.top_ax, self.top_canvas, image_type, "Top View")
                
        except Exception as e:
            self.show_error(f"Failed to load images: {e}")
            
    def load_image_to_axes(self, image_path, ax, canvas, image_type, view_name):
        """matplotlib axes에 이미지 로드"""
        # 디버깅: 경로 출력
        print(f"Trying to load: {image_path}")
        print(f"File exists: {os.path.exists(image_path)}")
        
        # axes 초기화
        ax.clear()
        ax.axis('off')  # 축 완전히 제거
        
        # 좌표 정보 표시 비활성화
        ax.format_coord = lambda x, y: ''
        
        if not os.path.exists(image_path):
            # 대안 확장자 시도 (inst_seg의 경우)
            if image_type == 'inst_seg':
                alt_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
                base_path = os.path.splitext(image_path)[0]
                
                for ext in alt_extensions:
                    alt_path = base_path + ext
                    print(f"Trying alternative: {alt_path}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        print(f"Found alternative: {alt_path}")
                        break
                else:
                    # 폴더 내용 확인
                    folder_path = os.path.dirname(image_path)
                    if os.path.exists(folder_path):
                        files = os.listdir(folder_path)
                        print(f"Files in {folder_path}: {files}")
                    ax.text(0.5, 0.5, f"File not found", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
                    canvas.draw()
                    return
            else:
                ax.text(0.5, 0.5, f"File not found", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
                canvas.draw()
                return
            
        try:
            if image_type == 'depth':
                # NPY 파일 로드
                depth_data = np.load(image_path)
                im = ax.imshow(depth_data, cmap='viridis', aspect='equal')
                # 컬러바 제거 (depth 수치 표시 안함)
                if hasattr(ax, '_colorbar'):
                    ax._colorbar.remove()
            else:
                # PNG 파일 로드
                image = Image.open(image_path)
                image_array = np.array(image)
                ax.imshow(image_array, aspect='equal')
            
            # 여백 최소화하면서 비율 유지
            ax.margins(0)
            
            # figure 여백 최소화
            ax.figure.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            canvas.draw()
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            ax.text(0.5, 0.5, f"Load Error", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
            canvas.draw()

    def load_and_display_objects(self, index):
        """JSON 파일에서 물체 정보 로드 및 표시"""
        try:
            json_path = os.path.join(self.platform_path, 'conf', f"{index:04d}.json")
            
            if not os.path.exists(json_path):
                self.objects_label.config(text="No object data available", foreground="gray")
                return
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'objects' not in data:
                self.objects_label.config(text="No objects found in data", foreground="gray")
                return
            
            objects = data['objects']
            if not objects:
                self.objects_label.config(text="No objects in scene", foreground="blue")
                return
            
            # 물체 클래스 이름들 추출
            object_classes = []
            for obj in objects:
                if isinstance(obj, dict) and 'class' in obj:
                    object_classes.append(obj['class'])
            
            if object_classes:
                # 중복 제거 및 정렬
                unique_classes = sorted(list(set(object_classes)))
                objects_text = f"Objects ({len(objects)} total): {', '.join(unique_classes)}"
                self.objects_label.config(text=objects_text, foreground="black")
            else:
                self.objects_label.config(text="No class information found", foreground="orange")
                
        except json.JSONDecodeError:
            self.objects_label.config(text="Invalid JSON format", foreground="red")
        except Exception as e:
            self.objects_label.config(text=f"Error loading objects: {str(e)}", foreground="red")
            print(f"Error loading objects from {json_path}: {e}")
        
    def show_error(self, message):
        """양쪽 axes에 에러 메시지 표시"""
        self.side_ax.clear()
        self.side_ax.axis('off')
        self.side_ax.text(0.5, 0.5, message, ha='center', va='center', 
                         transform=self.side_ax.transAxes, fontsize=12, color='red')
        self.side_canvas.draw()
        
        self.top_ax.clear()
        self.top_ax.axis('off')
        self.top_ax.text(0.5, 0.5, message, ha='center', va='center', 
                        transform=self.top_ax.transAxes, fontsize=12, color='red')
        self.top_canvas.draw()
        
        # 물체 정보도 에러로 표시
        self.objects_label.config(text="Error loading data", foreground="red")
        
    def show_no_data(self):
        """데이터 없음 메시지 표시"""
        self.side_ax.clear()
        self.side_ax.axis('off')
        self.side_ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                         transform=self.side_ax.transAxes, fontsize=12, color='gray')
        self.side_canvas.draw()
        
        self.top_ax.clear()
        self.top_ax.axis('off')
        self.top_ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                        transform=self.top_ax.transAxes, fontsize=12, color='gray')
        self.top_canvas.draw()
        
        self.index_entry_var.set("No data")
        self.total_count_var.set("0")
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
        self.go_btn.config(state=tk.DISABLED)
        
        # 물체 정보도 초기화
        self.objects_label.config(text="No data loaded", foreground="gray")

def main():
    root = tk.Tk()
    app = DataMonitorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()