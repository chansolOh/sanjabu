import tkinter as tk
from tkinter import ttk, messagebox
import json
import socket
import threading
import os


class MultiPCController:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-PC 관제 시스템")

        # Socket 서버 설정
        self.clients = {}  # client_id: conn 저장
        self.server_socket = None
        self.server_thread = None
        
        self.output_root_path = "/nas/Dataset/Dataset_2025/test"
        root_path = "/nas/ochansol/isaac/sanjabu/envs"

        # dict 구조 예시
        self.env_dict = {
            "Home" : {
                "LivingRoom_Kitchen" : {},
            },
            "Logistic_site" : {
                "FOODnamoo" : {},
            },
            "Manufactory" : {
                "FOODnamoo_poultry_plant" : {},
            }
        }

        for env_name in self.env_dict.keys():
            for section_name in self.env_dict[env_name].keys():
                platform_dir_path = f"{root_path}/{env_name}/platform_usd/{section_name}"
                platform_list = [i for i in os.listdir(platform_dir_path) if i.endswith(".usd") or i.endswith(".usda") or i.endswith(".usdc")]
                for platform_name in platform_list:
                    self.env_dict[env_name][section_name][platform_name.split(".")[0]] = {
                        "usd_path": f"{platform_dir_path}/{platform_name}",
                        "total_data_num" : 0,
                    }
                # 각 env와 section에 대해 platforms 키 설정 (루프 안으로 이동)
                self.env_dict[env_name][section_name]["platforms"] = platform_list

        self.pc_list = []
        self.status_vars = {}
        self.setting_widgets = {}  # 각 PC의 설정 위젯들을 저장

        # 서버 상태 변수
        self.server_status_var = tk.StringVar(value="서버 중지")
        
        self.setup_ui()
        self.start_server()

    def setup_ui(self):
        # 서버 상태 표시
        self.frame_server = ttk.LabelFrame(self.root, text="서버 상태")
        self.frame_server.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(self.frame_server, text="서버 상태:").grid(row=0, column=0, padx=5)
        ttk.Label(self.frame_server, textvariable=self.server_status_var).grid(row=0, column=1, padx=5)
        ttk.Button(self.frame_server, text="서버 시작", command=self.start_server).grid(row=0, column=2, padx=5)
        ttk.Button(self.frame_server, text="서버 중지", command=self.stop_server).grid(row=0, column=3, padx=5)

        # PC 수동 등록 (테스트용)
        self.frame_manual = ttk.LabelFrame(self.root, text="수동 PC 등록 (테스트용)")
        self.frame_manual.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.name_var = tk.StringVar()
        ttk.Label(self.frame_manual, text="PC 이름").grid(row=0, column=0)
        ttk.Entry(self.frame_manual, textvariable=self.name_var, width=15).grid(row=0, column=1)
        ttk.Button(self.frame_manual, text="수동 등록", command=self.manual_register_pc).grid(row=0, column=2, padx=5)

        self.frame_status = ttk.LabelFrame(self.root, text="실행 상태")
        self.frame_status.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        headers = ["PC 이름", "연결상태", "실행상태", "현재 Scene", "Env", "Section", "Platform", "Scene Start", "Scene End", "", "", "", ""]
        for col, title in enumerate(headers):
            tk.Label(self.frame_status, text=title, font=("Helvetica", 9, "bold"), bg="#ddd").grid(row=0, column=col, sticky="nsew", padx=1, pady=1)

    def start_server(self):
        """Socket 서버 시작"""
        if self.server_socket is not None:
            return
            
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("192.168.0.137", 1823))
            self.server_socket.listen()
            
            self.server_status_var.set("서버 실행중 (192.168.0.137:1823)")
            
            # 클라이언트 접속 대기 스레드
            self.server_thread = threading.Thread(target=self.accept_clients, daemon=True)
            self.server_thread.start()
            
            print("[SERVER STARTED] Waiting for clients...")
            
        except Exception as e:
            messagebox.showerror("서버 오류", f"서버 시작 실패: {e}")
            self.server_status_var.set("서버 시작 실패")

    def stop_server(self):
        """Socket 서버 중지"""
        try:
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            
            # 모든 클라이언트 연결 해제
            for client_id, conn in list(self.clients.items()):
                try:
                    conn.close()
                except:
                    pass
            self.clients.clear()
            
            self.server_status_var.set("서버 중지")
            
            # 연결된 PC들의 상태를 연결 해제로 변경
            for pc in self.pc_list:
                if pc['name'] in self.status_vars:
                    self.status_vars[pc['name']]['connection_color'].config(bg="red")
            
            print("[SERVER STOPPED]")
            
        except Exception as e:
            print(f"[ERROR] 서버 중지 중 오류: {e}")

    def accept_clients(self):
        """클라이언트 접속 대기"""
        while self.server_socket:
            try:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
            except:
                break

    def handle_client(self, conn, addr):
        """클라이언트 처리"""
        client_id = None
        try:
            # 클라이언트로부터 ID 받기
            client_id = conn.recv(1024).decode().strip()
            
            if not client_id:
                return
                
            self.clients[client_id] = conn
            print(f"[CONNECTED] {client_id} from {addr}")
            
            # GUI에서 PC 자동 등록
            self.root.after(0, self.auto_register_pc, client_id)
            
            # 클라이언트 메시지 수신 대기
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                message = data.decode()
                print(f"[{client_id}] {message}")
                
                # 클라이언트 상태 업데이트 처리
                self.root.after(0, self.handle_client_message, client_id, message)

        except Exception as e:
            print(f"[ERROR] {addr} - {e}")
        finally:
            print(f"[DISCONNECTED] {client_id} from {addr}")
            if client_id:
                self.clients.pop(client_id, None)
                # GUI에서 연결 상태 업데이트
                self.root.after(0, self.update_connection_status, client_id, False)
            conn.close()

    def handle_client_message(self, client_id, message):
        """클라이언트 메시지 처리"""
        try:
            # JSON 메시지인 경우 파싱해서 상태 업데이트
            data = json.loads(message)
            
            if data.get("type") == "status":
                status = data.get("status", "Unknown")
                if client_id in self.status_vars:
                    self.status_vars[client_id]['status'].set(status)
                    
            elif data.get("cmd") == "scene_num_check":
                # 클라이언트가 현재 진행 중인 scene 번호를 알려줌
                client_name = data.get("name", client_id)
                scene_num = data.get("data", "")
                print(f"[SCENE UPDATE] {client_name} - Scene: {scene_num}")
                
                # GUI에서 현재 Scene 번호 업데이트
                if client_name in self.status_vars:
                    self.status_vars[client_name]['current_scene'].set(f"Scene {scene_num}")
                    
            elif data.get("cmd") == "stop":
                # 클라이언트가 작업 완료/중지를 알려줌
                client_name = data.get("name", client_id)
                print(f"[WORK COMPLETED] {client_name} - Stopped")
                
                # GUI에서 상태를 Stopped로 변경
                if client_name in self.status_vars:
                    self.status_vars[client_name]['status'].set("Stopped")
                    self.status_vars[client_name]['color'].config(bg="red")
                    self.status_vars[client_name]['current_scene'].set("-")
                    
        except json.JSONDecodeError:
            # 일반 텍스트 메시지는 그냥 로그만
            pass

    def auto_register_pc(self, client_id):
        """클라이언트 접속 시 자동 PC 등록"""
        # 이미 등록된 PC인지 확인
        if any(pc['name'] == client_id for pc in self.pc_list):
            # 기존 PC의 연결 상태만 업데이트
            self.update_connection_status(client_id, True)
            return

        info = {
            "name": client_id,
            "scene_start": 0,
            "scene_end": 0,
            "env": "",
            "section": "",
            "platform": ""
        }
        self.pc_list.append(info)

        status_var = tk.StringVar(value="Idle")
        current_scene_var = tk.StringVar(value="-")  # 현재 Scene 번호 표시
        connection_color = tk.Label(self.frame_status, width=2, height=1, bg="green")  # 연결됨
        status_color = tk.Label(self.frame_status, width=2, height=1, bg="gray")
        self.status_vars[info['name']] = {
            "status": status_var, 
            "color": status_color,
            "connection_color": connection_color,
            "current_scene": current_scene_var
        }

        self.create_pc_row(info)

    def manual_register_pc(self):
        """수동 PC 등록 (테스트용)"""
        name = self.name_var.get().strip()
        
        if not name:
            messagebox.showerror("입력 오류", "PC 이름을 입력하세요.")
            return
            
        if name in [pc['name'] for pc in self.pc_list]:
            messagebox.showerror("중복 오류", f"이미 등록된 PC 이름입니다: {name}")
            return

        info = {
            "name": name,
            "scene_start": 0,
            "scene_end": 0,
            "env": "",
            "section": "",
            "platform": ""
        }
        self.pc_list.append(info)

        status_var = tk.StringVar(value="Idle")
        current_scene_var = tk.StringVar(value="-")  # 현재 Scene 번호 표시
        connection_color = tk.Label(self.frame_status, width=2, height=1, bg="red")  # 연결안됨
        status_color = tk.Label(self.frame_status, width=2, height=1, bg="gray")
        self.status_vars[info['name']] = {
            "status": status_var, 
            "color": status_color,
            "connection_color": connection_color,
            "current_scene": current_scene_var
        }

        self.create_pc_row(info)
        self.name_var.set("")

    def create_pc_row(self, info):
        """PC 행 생성"""
        row = len(self.pc_list)
        
        # PC 이름
        tk.Label(self.frame_status, text=info['name'], fg="navy", font=("Helvetica", 10, "bold")).grid(row=row, column=0, padx=5, pady=2)
        
        # 연결 상태 표시등
        self.status_vars[info['name']]['connection_color'].grid(row=row, column=1, padx=5)
        
        # 실행 상태 표시등
        self.status_vars[info['name']]['color'].grid(row=row, column=2, padx=5)
        
        # 현재 Scene 번호 표시
        ttk.Label(self.frame_status, textvariable=self.status_vars[info['name']]['current_scene']).grid(row=row, column=3, padx=5)
        
        # Env 콤보박스
        env_combo = ttk.Combobox(self.frame_status, values=list(self.env_dict.keys()), state="readonly", width=12)
        env_combo.grid(row=row, column=4, padx=2)
        
        # Section 콤보박스
        section_combo = ttk.Combobox(self.frame_status, state="readonly", width=15)
        section_combo.grid(row=row, column=5, padx=2)
        
        # Platform 콤보박스
        platform_combo = ttk.Combobox(self.frame_status, state="readonly", width=15)
        platform_combo.grid(row=row, column=6, padx=2)
        
        # Scene Start 입력
        scene_start_var = tk.StringVar(value="0")
        scene_start_entry = ttk.Entry(self.frame_status, textvariable=scene_start_var, width=8)
        scene_start_entry.grid(row=row, column=7, padx=2)
        
        # Scene End 입력
        scene_end_var = tk.StringVar(value="0")
        scene_end_entry = ttk.Entry(self.frame_status, textvariable=scene_end_var, width=8)
        scene_end_entry.grid(row=row, column=8, padx=2)
        
        # 설정 변경 시 PC 정보 업데이트
        def update_pc_info():
            info['env'] = env_combo.get()
            info['section'] = section_combo.get()
            info['platform'] = platform_combo.get()
            try:
                info['scene_start'] = int(scene_start_var.get()) if scene_start_var.get() else 0
                info['scene_end'] = int(scene_end_var.get()) if scene_end_var.get() else 0
            except ValueError:
                pass
        
        # 콤보박스 이벤트 바인딩
        def on_env_change(event):
            self.update_section_for_pc(event, section_combo, platform_combo)
            update_pc_info()
            
        def on_section_change(event):
            self.update_platform_for_pc(event, env_combo, platform_combo)
            update_pc_info()
        
        env_combo.bind("<<ComboboxSelected>>", on_env_change)
        section_combo.bind("<<ComboboxSelected>>", on_section_change)
        platform_combo.bind("<<ComboboxSelected>>", lambda e: update_pc_info())
        scene_start_entry.bind("<KeyRelease>", lambda e: update_pc_info())
        scene_end_entry.bind("<KeyRelease>", lambda e: update_pc_info())
        
        # 위젯들을 저장
        self.setting_widgets[info['name']] = {
            "env_combo": env_combo,
            "section_combo": section_combo,
            "platform_combo": platform_combo,
            "scene_start_var": scene_start_var,
            "scene_end_var": scene_end_var,
            "row": row
        }
        
        # 제어 버튼들
        ttk.Button(self.frame_status, text="Play", command=lambda i=info: self.play(i)).grid(row=row, column=9)
        ttk.Button(self.frame_status, text="Pause", command=lambda i=info: self.pause(i)).grid(row=row, column=10)
        ttk.Button(self.frame_status, text="Stop", command=lambda i=info: self.stop(i)).grid(row=row, column=11)
        ttk.Button(self.frame_status, text="삭제", command=lambda i=info: self.delete_pc(i)).grid(row=row, column=12)

    def update_connection_status(self, client_id, connected):
        """클라이언트 연결 상태 업데이트"""
        if client_id in self.status_vars:
            color = "green" if connected else "red"
            self.status_vars[client_id]['connection_color'].config(bg=color)

    def update_section_for_pc(self, event, section_combo, platform_combo):
        env = event.widget.get()
        if env and env in self.env_dict:
            section_combo['values'] = list(self.env_dict[env].keys())
            section_combo.set('')
            platform_combo.set('')
            platform_combo['values'] = []

    def update_platform_for_pc(self, event, env_combo, platform_combo):
        env = env_combo.get()
        section = event.widget.get()
        if env and section and env in self.env_dict and section in self.env_dict[env]:
            platform_keys = [k for k in self.env_dict[env][section].keys() if k != "platforms"]
            platform_combo['values'] = platform_keys
            platform_combo.set('')

    def delete_pc(self, pc_info):
        result = messagebox.askyesno("삭제 확인", f"PC '{pc_info['name']}'을(를) 삭제하시겠습니까?")
        if result:
            pc_name = pc_info['name']
            
            # 클라이언트 연결 해제
            if pc_name in self.clients:
                try:
                    self.clients[pc_name].close()
                except:
                    pass
                del self.clients[pc_name]
            
            # PC 리스트에서 제거
            self.pc_list = [pc for pc in self.pc_list if pc['name'] != pc_name]
            
            # 상태 변수 제거
            if pc_name in self.status_vars:
                del self.status_vars[pc_name]
            
            # 설정 위젯 제거
            if pc_name in self.setting_widgets:
                del self.setting_widgets[pc_name]
            
            # UI 다시 그리기
            self.refresh_status_frame()

    def refresh_status_frame(self):
        # 기존 상태 프레임 내용 삭제 (헤더 제외)
        for widget in self.frame_status.winfo_children():
            info = widget.grid_info()
            if info and info.get('row', 0) > 0:
                widget.destroy()
        
        # 상태 변수와 설정 위젯 초기화
        self.status_vars.clear()
        self.setting_widgets.clear()
        
        # PC 리스트 다시 그리기
        temp_pc_list = self.pc_list.copy()
        self.pc_list.clear()
        
        for pc_info in temp_pc_list:
            self.recreate_pc_row(pc_info)

    def recreate_pc_row(self, pc_info):
        self.pc_list.append(pc_info)
        
        status_var = tk.StringVar(value="Idle")
        current_scene_var = tk.StringVar(value="-")  # 현재 Scene 번호 표시
        # 연결 상태 확인
        is_connected = pc_info['name'] in self.clients
        connection_color = tk.Label(self.frame_status, width=2, height=1, bg="green" if is_connected else "red")
        status_color = tk.Label(self.frame_status, width=2, height=1, bg="gray")
        self.status_vars[pc_info['name']] = {
            "status": status_var, 
            "color": status_color,
            "connection_color": connection_color,
            "current_scene": current_scene_var
        }

        row = len(self.pc_list)
        
        # PC 이름
        tk.Label(self.frame_status, text=pc_info['name'], fg="navy", font=("Helvetica", 10, "bold")).grid(row=row, column=0, padx=5, pady=2)
        
        # 연결 상태 표시등
        connection_color.grid(row=row, column=1, padx=5)
        
        # 실행 상태 표시등  
        status_color.grid(row=row, column=2, padx=5)
        
        # 현재 Scene 번호 표시
        ttk.Label(self.frame_status, textvariable=current_scene_var).grid(row=row, column=3, padx=5)
        
        # Env 콤보박스
        env_combo = ttk.Combobox(self.frame_status, values=list(self.env_dict.keys()), state="readonly", width=12)
        env_combo.set(pc_info['env'])
        env_combo.grid(row=row, column=4, padx=2)
        
        # Section 콤보박스
        section_combo = ttk.Combobox(self.frame_status, state="readonly", width=15)
        if pc_info['env'] and pc_info['env'] in self.env_dict:
            section_combo['values'] = list(self.env_dict[pc_info['env']].keys())
            section_combo.set(pc_info['section'])
        section_combo.grid(row=row, column=5, padx=2)
        
        # Platform 콤보박스
        platform_combo = ttk.Combobox(self.frame_status, state="readonly", width=15)
        if pc_info['env'] and pc_info['section'] and pc_info['env'] in self.env_dict and pc_info['section'] in self.env_dict[pc_info['env']]:
            platform_keys = [k for k in self.env_dict[pc_info['env']][pc_info['section']].keys() if k != "platforms"]
            platform_combo['values'] = platform_keys
            platform_combo.set(pc_info['platform'])
        platform_combo.grid(row=row, column=6, padx=2)
        
        # Scene Start 입력
        scene_start_var = tk.StringVar(value=str(pc_info['scene_start']))
        scene_start_entry = ttk.Entry(self.frame_status, textvariable=scene_start_var, width=8)
        scene_start_entry.grid(row=row, column=7, padx=2)
        
        # Scene End 입력
        scene_end_var = tk.StringVar(value=str(pc_info['scene_end']))
        scene_end_entry = ttk.Entry(self.frame_status, textvariable=scene_end_var, width=8)
        scene_end_entry.grid(row=row, column=8, padx=2)
        
        # 설정 변경 시 PC 정보 업데이트
        def update_pc_info():
            pc_info['env'] = env_combo.get()
            pc_info['section'] = section_combo.get()
            pc_info['platform'] = platform_combo.get()
            try:
                pc_info['scene_start'] = int(scene_start_var.get()) if scene_start_var.get() else 0
                pc_info['scene_end'] = int(scene_end_var.get()) if scene_end_var.get() else 0
            except ValueError:
                pass
        
        # 콤보박스 이벤트 바인딩
        def on_env_change(event):
            self.update_section_for_pc(event, section_combo, platform_combo)
            update_pc_info()
            
        def on_section_change(event):
            self.update_platform_for_pc(event, env_combo, platform_combo)
            update_pc_info()
        
        env_combo.bind("<<ComboboxSelected>>", on_env_change)
        section_combo.bind("<<ComboboxSelected>>", on_section_change)
        platform_combo.bind("<<ComboboxSelected>>", lambda e: update_pc_info())
        scene_start_entry.bind("<KeyRelease>", lambda e: update_pc_info())
        scene_end_entry.bind("<KeyRelease>", lambda e: update_pc_info())
        
        # 위젯들을 저장
        self.setting_widgets[pc_info['name']] = {
            "env_combo": env_combo,
            "section_combo": section_combo,
            "platform_combo": platform_combo,
            "scene_start_var": scene_start_var,
            "scene_end_var": scene_end_var,
            "row": row
        }
        
        # 제어 버튼들
        ttk.Button(self.frame_status, text="Play", command=lambda i=pc_info: self.play(i)).grid(row=row, column=9)
        ttk.Button(self.frame_status, text="Pause", command=lambda i=pc_info: self.pause(i)).grid(row=row, column=10)
        ttk.Button(self.frame_status, text="Stop", command=lambda i=pc_info: self.stop(i)).grid(row=row, column=11)
        ttk.Button(self.frame_status, text="삭제", command=lambda i=pc_info: self.delete_pc(i)).grid(row=row, column=12)

    def check_scene_conflict(self, target_pc):
        """
        target_pc와 같은 env, section, platform을 사용하는 실행중인 PC들과 
        scene 범위가 겹치지 않는지 확인
        """
        target_env = target_pc.get('env', '')
        target_section = target_pc.get('section', '')
        target_platform = target_pc.get('platform', '')
        target_start = target_pc.get('scene_start', 0)
        target_end = target_pc.get('scene_end', 0)
        
        # 필수 정보가 없으면 경고하고 차단
        if not all([target_env, target_section, target_platform]):
            messagebox.showerror("설정 오류", f"PC '{target_pc['name']}'의 Env, Section, Platform을 모두 설정해주세요.")
            return False
            
        # scene_start와 scene_end가 올바르게 설정되었는지 확인
        if target_start < 0 or target_end < 0:
            messagebox.showerror("설정 오류", f"PC '{target_pc['name']}'의 Scene Start와 Scene End는 0 이상이어야 합니다.")
            return False
            
        if target_start > target_end:
            messagebox.showerror("설정 오류", f"PC '{target_pc['name']}'의 Scene Start({target_start})가 Scene End({target_end})보다 클 수 없습니다.")
            return False
        
        # 같은 env, section, platform을 사용하는 실행중인 PC들 찾기
        for pc in self.pc_list:
            if pc['name'] == target_pc['name']:
                continue  # 자기 자신은 제외
                
            # 현재 실행중인 PC인지 확인
            if pc['name'] in self.status_vars:
                current_status = self.status_vars[pc['name']]['status'].get()
                if current_status != "Running":
                    continue  # 실행중이 아니면 패스
            else:
                continue
                
            # 같은 환경 설정인지 확인
            if (pc.get('env', '') == target_env and 
                pc.get('section', '') == target_section and 
                pc.get('platform', '') == target_platform):
                
                pc_start = pc.get('scene_start', 0)
                pc_end = pc.get('scene_end', 0)
                
                # scene 범위 겹침 검사 (range overlap)
                # 겹치지 않는 조건: target_end < pc_start or target_start > pc_end
                # 겹치는 조건: not (겹치지 않는 조건)
                if not (target_end < pc_start or target_start > pc_end):
                    messagebox.showerror(
                        "Scene 범위 충돌", 
                        f"PC '{target_pc['name']}'의 Scene 범위({target_start}~{target_end})가 "
                        f"실행중인 PC '{pc['name']}'의 범위({pc_start}~{pc_end})와 겹칩니다.\n"
                        f"환경: {target_env}/{target_section}/{target_platform}"
                    )
                    return False
        
        return True

    def send_command_to_client(self, pc_info, command_type="start"):
        """클라이언트에게 명령 전송"""
        client_id = pc_info['name']
        
        if client_id not in self.clients:
            messagebox.showerror("연결 오류", f"PC '{client_id}'가 연결되어 있지 않습니다.")
            return False
        
        # 원본 socket 코드의 메시지 형태에 맞춤
        msg_dict = {
            "cmd": command_type,
            "output_root_path" : self.output_root_path,
            "env_name": pc_info.get('env', ''),
            "section_name": pc_info.get('section', ''),
            "platform_name": pc_info.get('platform', ''),
            "scene_start": pc_info.get('scene_start', 0),
            "scene_end": pc_info.get('scene_end', 0),
            "object_num": 5  # 기본값
        }
        
        try:
            send_msg = json.dumps(msg_dict).encode()
            self.clients[client_id].sendall(send_msg)
            print(f"[COMMAND SENT] {client_id}: {msg_dict}")
            return True
        except Exception as e:
            print(f"[ERROR] 명령 전송 실패 {client_id}: {e}")
            messagebox.showerror("전송 오류", f"PC '{client_id}'에 명령 전송 실패: {e}")
            return False

    def play(self, pc_info):
        # scene 범위 겹침 검사
        if not self.check_scene_conflict(pc_info):
            return
            
        # 클라이언트에게 start 명령 전송
        if self.send_command_to_client(pc_info, "start"):
            self.status_vars[pc_info['name']]['status'].set("Running")
            self.status_vars[pc_info['name']]['color'].config(bg="green")
            # Play 시작 시 현재 Scene을 "Starting..."으로 표시
            self.status_vars[pc_info['name']]['current_scene'].set("Starting...")

    def pause(self, pc_info):
        # 클라이언트에게 pause 명령 전송
        if self.send_command_to_client(pc_info, "pause"):
            self.status_vars[pc_info['name']]['status'].set("Paused")
            self.status_vars[pc_info['name']]['color'].config(bg="orange")

    def stop(self, pc_info):
        # 클라이언트에게 stop 명령 전송
        if self.send_command_to_client(pc_info, "stop"):
            self.status_vars[pc_info['name']]['status'].set("Stopped")
            self.status_vars[pc_info['name']]['color'].config(bg="red")
            self.status_vars[pc_info['name']]['current_scene'].set("-")

    def __del__(self):
        """소멸자 - 서버 정리"""
        self.stop_server()

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiPCController(root)
    
    # 창 닫기 시 서버 정리
    def on_closing():
        app.stop_server()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()