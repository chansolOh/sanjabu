import json
import socket
import threading
import os
from typing import Dict, List, Any

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, DataTable, Button, Static, Input, Select, Label, OptionList
)
from textual.widgets.option_list import Option
from textual.screen import ModalScreen
from textual.binding import Binding
from textual import events


class NumberInputModal(ModalScreen[str]):
    """숫자 입력 모달"""
    
    DEFAULT_CSS = """
    NumberInputModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    
    #input_dialog {
        width: 50%;
        max-width: 60;
        min-width: 30;
        height: 12;
        border: thick $primary;
        background: $surface;
        padding: 2;
        margin: 4;
        layer: above;
        offset: 0 -2;
    }
    
    #input_field {
        margin: 1 0;
        height: 3;
    }
    
    #title {
        text-align: center;
        text-style: bold;
        height: 2;
        margin-bottom: 1;
    }
    
    Horizontal {
        height: 3;
        margin-top: 1;
        align: center middle;
    }
    
    Button {
        margin: 0 1;
        min-width: 10;
    }
    """
    
    def __init__(self, title: str, current_value: str, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.current_value = current_value
        
    def compose(self) -> ComposeResult:
        yield Container(
            Static(f"Enter {self.title}", id="title"),
            Input(
                value=self.current_value,
                placeholder="Enter number (0-9999)",
                type="integer",
                id="input_field"
            ),
            Horizontal(
                Button("Save [Enter]", variant="primary", id="save"),
                Button("Cancel [Esc]", variant="default", id="cancel"),
            ),
            id="input_dialog"
        )
    
    def on_mount(self) -> None:
        """포커스를 입력 필드에 설정"""
        self.set_timer(0.1, self.focus_input)
    
    def focus_input(self) -> None:
        """입력 필드에 포커스 설정"""
        try:
            input_field = self.query_one("#input_field", Input)
            input_field.focus()
        except:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            input_field = self.query_one("#input_field", Input)
            self.dismiss(input_field.value)
        else:
            self.dismiss("")
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter 키로 저장"""
        self.dismiss(event.value)
    
    def on_key(self, event: events.Key) -> None:
        """키 이벤트 처리"""
        if event.key == "escape":
            self.dismiss("")
            event.stop()


class DropdownOverlay(Container):
    """드롭다운 오버레이"""
    
    DEFAULT_CSS = """
    DropdownOverlay {
        width: 30%;
        max-width: 40;
        min-width: 20;
        height: auto;
        max-height: 10;
        border: thick $primary;
        background: $surface;
        layer: overlay;
    }
    
    DropdownOverlay > OptionList {
        height: auto;
        width: 1fr;
    }
    """
    
    def __init__(self, options, current_value, callback, **kwargs):
        super().__init__(**kwargs)
        self.options = options
        self.current_value = current_value
        self.callback = callback
        self.option_list = None
        
    def compose(self) -> ComposeResult:
        option_items = []
        highlighted_index = 0
        
        for i, option in enumerate(self.options):
            if option == self.current_value:
                highlighted_index = i
                option_items.append(Option(f"● {option}", id=str(i)))
            else:
                option_items.append(Option(f"  {option}", id=str(i)))
        
        self.option_list = OptionList(*option_items, id="dropdown_options")
        self.option_list.highlighted = highlighted_index
        yield self.option_list
    
    def on_mount(self) -> None:
        """마운트 시 포커스 설정"""
        if self.option_list:
            self.option_list.focus()
    
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """옵션 선택 처리 (Enter 키 또는 클릭)"""
        selected_index = int(event.option.id)
        selected_value = self.options[selected_index]
        self.callback(selected_value)
        self.remove()
        event.stop()
    
    def on_click(self, event: events.Click) -> None:
        """마우스 클릭 처리"""
        if self.option_list:
            relative_y = event.y - self.option_list.region.y
            if 0 <= relative_y < len(self.options):
                selected_index = relative_y
                if 0 <= selected_index < len(self.options):
                    selected_value = self.options[selected_index]
                    self.callback(selected_value)
                    self.remove()
                    event.stop()
    
    def on_key(self, event: events.Key) -> None:
        """키 이벤트 처리"""
        if event.key == "escape":
            self.callback(None)
            self.remove()
            event.stop()
        elif event.key == "enter":
            if self.option_list:
                highlighted = self.option_list.highlighted
                if 0 <= highlighted < len(self.options):
                    selected_value = self.options[highlighted]
                    self.callback(selected_value)
                    self.remove()
            event.stop()
        elif event.key in ["up", "down"]:
            if self.option_list:
                if event.key == "up":
                    new_highlighted = max(0, self.option_list.highlighted - 1)
                elif event.key == "down":
                    new_highlighted = min(len(self.options) - 1, self.option_list.highlighted + 1)
                self.option_list.highlighted = new_highlighted
            event.stop()
        else:
            event.stop()


class EditableDataTable(DataTable):
    """편집 가능한 DataTable"""
    
    def __init__(self, controller, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.edit_mode = False
        self.dropdown_overlay = None
        self.saved_cursor_position = None
        self.preserve_cursor_on_refresh = False  # 새로운 플래그
        
    def on_key(self, event: events.Key) -> None:
        """키 이벤트 처리"""
        if self.edit_mode and self.dropdown_overlay:
            event.stop()
            return
            
        if event.key == "enter" and not self.edit_mode:
            self.start_edit()
            event.stop()
        elif event.key == "escape" and self.edit_mode:
            self.cancel_edit()
            event.stop()
        
        if event.key == "enter" and self.edit_mode:
            event.stop()
    
    def on_click(self, event: events.Click) -> None:
        """마우스 클릭 처리"""
        if event.button == 1 and not self.edit_mode:
            self.set_timer(0.1, self.start_edit)
    
    def save_cursor_position(self) -> None:
        """현재 커서 위치 저장"""
        try:
            self.saved_cursor_position = (self.cursor_row, self.cursor_column)
            self.preserve_cursor_on_refresh = True
        except:
            self.saved_cursor_position = (0, 0)
            self.preserve_cursor_on_refresh = True
    
    def restore_cursor_position(self) -> None:
        """커서 위치 복원"""
        try:
            if self.saved_cursor_position and self.preserve_cursor_on_refresh:
                row, col = self.saved_cursor_position
                # 테이블 범위 체크
                max_row = len(self.controller.pc_list) - 1
                max_col = len(self.columns) - 1
                
                if max_row >= 0 and max_col >= 0:
                    safe_row = min(max(0, row), max_row)
                    safe_col = min(max(0, col), max_col)
                    
                    # 포커스와 커서 이동을 동시에 처리
                    self.can_focus = True
                    # 커서 이동을 먼저 하고 포커스를 나중에
                    self.move_cursor(row=safe_row, column=safe_col)
                    self.focus()
                
                self.preserve_cursor_on_refresh = False
            else:
                # 저장된 위치가 없으면 기본 동작
                self.can_focus = True
                self.focus()
        except Exception as e:
            # 실패 시 기본 위치로
            self.can_focus = True
            self.move_cursor(row=0, column=0)
            self.focus()
    
    def start_edit(self) -> None:
        """편집 모드 시작"""
        if self.cursor_row >= len(self.controller.pc_list):
            return
            
        self.save_cursor_position()
            
        pc = self.controller.pc_list[self.cursor_row]
        col = self.cursor_column
        
        # Command 열 (4번째) 추가, 기존 열들은 한 칸씩 뒤로 이동
        if col == 4:  # Command 열
            self.edit_command_cell(pc)
        elif col in [5, 6, 7, 8, 9]:  # Env, Section, Platform, Start, End
            if col == 5:
                self.edit_env_cell(pc)
            elif col == 6:
                self.edit_section_cell(pc)
            elif col == 7:
                self.edit_platform_cell(pc)
            elif col in [8, 9]:
                self.edit_number_cell(pc, col)
    
    def edit_command_cell(self, pc):
        """Command 편집"""
        options = ["SceneGen", "PreGrasp", "Grasp"]
        current = pc.get('command', 'SceneGen')
        self.show_dropdown(options, current, lambda val: self.save_edit(pc, 'command', val))
    
    def edit_env_cell(self, pc):
        """Environment 편집"""
        options = list(self.controller.env_dict.keys())
        if options:
            current = pc.get('env', '')
            self.show_dropdown(options, current, lambda val: self.save_edit(pc, 'env', val))
    
    def edit_section_cell(self, pc):
        """Section 편집"""
        env = pc.get('env', '')
        if env and env in self.controller.env_dict:
            options = list(self.controller.env_dict[env].keys())
            current = pc.get('section', '')
            self.show_dropdown(options, current, lambda val: self.save_edit(pc, 'section', val))
        else:
            self.controller.notify("Please select Environment first", severity="warning")
            self.edit_mode = False
    
    def edit_platform_cell(self, pc):
        """Platform 편집"""
        env = pc.get('env', '')
        section = pc.get('section', '')
        if env and section and env in self.controller.env_dict and section in self.controller.env_dict[env]:
            options = [k for k in self.controller.env_dict[env][section].keys() if k != "platforms"]
            current = pc.get('platform', '')
            self.show_dropdown(options, current, lambda val: self.save_edit(pc, 'platform', val))
        else:
            self.controller.notify("Please select Environment and Section first", severity="warning")
            self.edit_mode = False
    
    def edit_number_cell(self, pc, col):
        """숫자 입력 편집"""
        field = 'scene_start' if col == 8 else 'scene_end'  # 열 번호 조정
        current = str(pc.get(field, 0))
        title = f"Scene {'Start' if col == 8 else 'End'}"
        
        self.show_number_input(title, current, 
                              lambda val: self.save_number_edit(pc, field, val))
    
    def show_dropdown(self, options, current, callback):
        """드롭다운 표시"""
        if not options:
            self.edit_mode = False
            return
        
        if self.dropdown_overlay:
            self.dropdown_overlay.remove()
        
        def wrapped_callback(value):
            if value is not None:
                callback(value)
            
            if self.dropdown_overlay:
                try:
                    self.dropdown_overlay.remove()
                except:
                    pass
                self.dropdown_overlay = None
            
            self.edit_mode = False
            self.restore_cursor_position()
        
        try:
            self.dropdown_overlay = DropdownOverlay(
                options, 
                current, 
                wrapped_callback,
                id="dropdown_overlay"
            )
            
            app = self.app
            app.mount(self.dropdown_overlay)
            
            self.can_focus = False
            self.dropdown_overlay.focus()
            
        except Exception as e:
            self.controller.notify(f"Dropdown error: {e}", severity="error")
            self.edit_mode = False
    
    def show_number_input(self, title, current, callback):
        """숫자 입력 모달 표시"""
        def input_callback(result: str) -> None:
            if result:
                callback(result)
            self.edit_mode = False
            self.restore_cursor_position()
        
        modal = NumberInputModal(title, current)
        self.app.push_screen(modal, input_callback)
    
    def save_edit(self, pc, field, value):
        """편집 저장"""
        pc[field] = value
        
        if field == 'env':
            pc['section'] = ''
            pc['platform'] = ''
        elif field == 'section':
            pc['platform'] = ''
        
        # 커서 위치 저장 후 테이블 새로고침
        self.save_cursor_position()
        self.controller.refresh_table()
        self.controller.notify(f"Updated {field} to: {value}")
        
        self.edit_mode = False
    
    def save_number_edit(self, pc, field, value):
        """숫자 편집 저장"""
        try:
            pc[field] = int(value) if value else 0
            # 커서 위치 저장 후 테이블 새로고침
            self.save_cursor_position()
            self.controller.refresh_table()
            self.controller.notify(f"Updated {field} to: {value}")
        except ValueError:
            self.controller.notify("Invalid number", severity="error")
    
    def cancel_edit(self):
        """편집 취소"""
        if self.dropdown_overlay:
            try:
                self.dropdown_overlay.remove()
            except:
                pass
            self.dropdown_overlay = None
        
        self.edit_mode = False
        self.restore_cursor_position()
        self.controller.notify("Edit cancelled")


class MultiPCController(App):
    """Multi-PC Controller TUI Application"""
    
    CSS = """
    /* 반응형 레이아웃 개선 */
    Screen {
        layout: vertical;
    }
    
    #server_status {
        height: auto;
        min-height: 3;
        max-height: 5;
        margin: 1;
        border: solid green;
        padding: 1;
    }
    
    #pc_table {
        height: 1fr;
        min-height: 10;
        margin: 1;
        border: solid white;
    }
    
    #control_buttons {
        height: auto;
        min-height: 3;
        max-height: 5;
        margin: 1;
    }
    
    #info_panel {
        height: auto;
        min-height: 3;
        max-height: 7;
        margin: 1;
        border: solid blue;
        padding: 1;
        text-wrap: wrap;
    }
    
    /* 테이블 커서 가시성 개선 */
    DataTable > .datatable--cursor {
        background: $accent 70%;
        color: $text;
    }
    
    /* 버튼 반응형 크기 */
    Button {
        min-width: 8;
        margin: 0 1;
    }
    
    /* 컨테이너 반응형 */
    Container {
        width: 100%;
        height: auto;
    }
    
    Horizontal {
        width: 100%;
        height: auto;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("e,enter", "edit", "Edit"),
        Binding("p", "play", "Play"),
        Binding("pause", "pause", "Pause"),
        Binding("ctrl+s", "stop", "Stop"),
        Binding("delete", "delete", "Delete"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Socket 서버 설정
        self.clients: Dict[str, socket.socket] = {}
        self.server_socket = None
        self.server_thread = None
        
        self.output_root_path = "/nas/Dataset/Dataset_2025/dataset_v1"
        root_path = "/nas/ochansol/isaac/sanjabu/envs"
        
        # Environment 딕셔너리
        self.env_dict = {
            "Home": {"LivingRoom_Kitchen": {}},
            "Logistic_site": {"FOODnamoo": {}},
            "Manufactory": {"FOODnamoo_poultry_plant": {}},
        }
        
        # 실제 환경에서는 파일시스템 읽기
        try:
            for env_name in self.env_dict.keys():
                for section_name in self.env_dict[env_name].keys():
                    platform_dir_path = f"{root_path}/{env_name}/platform_usd/{section_name}"
                    if os.path.exists(platform_dir_path):
                        platform_list = [
                            i for i in os.listdir(platform_dir_path) 
                            if i.endswith((".usd", ".usda", ".usdc"))
                        ]
                        for platform_name in platform_list:
                            self.env_dict[env_name][section_name][platform_name.split(".")[0]] = {
                                "usd_path": f"{platform_dir_path}/{platform_name}",
                                "total_data_num": 0,
                            }
                        self.env_dict[env_name][section_name]["platforms"] = platform_list
        except Exception as e:
            # 테스트용 더미 데이터
            for env_name in self.env_dict.keys():
                for section_name in self.env_dict[env_name].keys():
                    self.env_dict[env_name][section_name].update({
                        "platform1": {"usd_path": "test1.usd", "total_data_num": 0},
                        "platform2": {"usd_path": "test2.usd", "total_data_num": 0},
                        "platforms": ["platform1.usd", "platform2.usd"]
                    })
        
        self.pc_list: List[Dict[str, Any]] = []
        self.pc_status: Dict[str, Dict[str, str]] = {}
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        yield Container(
            Static("Server: Starting...", id="server_status"),
            
            Static(
                "Navigation: ↑↓ Move | Enter/Click: Edit | P: Play | Pause: Pause | Ctrl+S: Stop | Del: Delete | Q: Quit",
                id="info_panel"
            ),
            
            EditableDataTable(
                self,
                id="pc_table",
                cursor_type="cell",
                zebra_stripes=True
            ),
            
            Horizontal(
                Button("Play", variant="success", id="play_btn"),
                Button("Pause", variant="warning", id="pause_btn"),
                Button("Stop", variant="error", id="stop_btn"),
                Button("Delete", variant="default", id="delete_btn"),
                id="control_buttons"
            )
        )
        
        yield Footer()
        
    def on_mount(self) -> None:
        # 테이블 설정 - Command 열 추가
        table = self.query_one(EditableDataTable)
        table.add_columns(
            "PC Name", "Conn", "Status", "Scene", "Command", "Env", "Section", "Platform", 
            "Start", "End"
        )
        
        # 서버 시작
        self.start_server()
        
    def start_server(self) -> None:
        """Socket 서버 시작"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("192.168.0.137", 1823))
            self.server_socket.listen()
            
            self.query_one("#server_status", Static).update(
                "Server: Running (192.168.0.137:1823)"
            )
            
            self.server_thread = threading.Thread(target=self.accept_clients, daemon=True)
            self.server_thread.start()
            
        except Exception as e:
            self.query_one("#server_status", Static).update(f"Server: Failed - {e}")
    
    def accept_clients(self) -> None:
        """클라이언트 접속 대기"""
        while self.server_socket:
            try:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
            except:
                break
    
    def handle_client(self, conn: socket.socket, addr) -> None:
        """클라이언트 처리"""
        client_id = None
        try:
            client_id = conn.recv(1024).decode().strip()
            
            if not client_id:
                return
                
            self.clients[client_id] = conn
            self.call_from_thread(self.auto_register_pc, client_id)
            
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                message = data.decode()
                self.call_from_thread(self.handle_client_message, client_id, message)

        except Exception as e:
            pass
        finally:
            if client_id:
                self.clients.pop(client_id, None)
                self.call_from_thread(self.update_connection_status, client_id, False)
            conn.close()
    
    def handle_client_message(self, client_id: str, message: str) -> None:
        """클라이언트 메시지 처리"""
        try:
            data = json.loads(message)
            
            if data.get("cmd") == "scene_num_check":
                client_name = data.get("name", client_id)
                scene_num = data.get("data", "")
                
                if client_name in self.pc_status:
                    if isinstance(scene_num, int):
                        self.pc_status[client_name]['current_scene'] = f"{scene_num:04d}"
                    else:
                        self.pc_status[client_name]['current_scene'] = str(scene_num)
                    self.refresh_table_silent()  # 조용한 새로고침 사용
                    
            elif data.get("cmd") == "stop":
                client_name = data.get("name", client_id)
                if client_name in self.pc_status:
                    self.pc_status[client_name]['status'] = "Stopped"
                    self.pc_status[client_name]['current_scene'] = "-"
                    self.refresh_table_silent()  # 조용한 새로고침 사용
                    
            elif data.get("cmd") == "complete":
                client_name = data.get("name", client_id)
                if client_name in self.pc_status:
                    self.pc_status[client_name]['status'] = "Completed"
                    self.pc_status[client_name]['current_scene'] = "Completed"
                    self.refresh_table_silent()  # 조용한 새로고침 사용
                    
        except json.JSONDecodeError:
            pass
    
    def auto_register_pc(self, client_id: str) -> None:
        """클라이언트 접속 시 자동 PC 등록"""
        existing_pc = next((pc for pc in self.pc_list if pc['name'] == client_id), None)
        
        if existing_pc:
            self.pc_status[client_id] = {
                'connection': 'Connected',
                'status': 'Idle',
                'current_scene': '-'
            }
            self.refresh_table_silent()  # 조용한 새로고침 사용
            return
        
        info = {
            "name": client_id,
            "command": "SceneGen",  # 기본값 추가
            "scene_start": 0,
            "scene_end": 0,
            "env": "",
            "section": "",
            "platform": ""
        }
        self.pc_list.append(info)
        
        self.pc_status[client_id] = {
            'connection': 'Connected',
            'status': 'Idle',
            'current_scene': '-'
        }
        
        self.refresh_table_silent()  # 조용한 새로고침 사용
    
    def update_connection_status(self, client_id: str, connected: bool) -> None:
        """연결 상태 업데이트"""
        if client_id in self.pc_status:
            self.pc_status[client_id]['connection'] = 'Connected' if connected else 'Disconnected'
            if not connected:
                self.pc_status[client_id]['status'] = 'Idle'
                self.pc_status[client_id]['current_scene'] = '-'
            self.refresh_table_silent()  # 조용한 새로고침 사용
    
    def get_command_display(self, command: str) -> str:
        """Command를 색상과 함께 표시"""
        command_colors = {
            "SceneGen": "#01f8df",    # rgb(1,248,223)
            "PreGrasp": "#0901ee",    # rgb(9,1,238)
            "Grasp": "#fcbc2f"        # rgb(252,188,47)
        }
        
        color = command_colors.get(command, "#ffffff")
        return f"[bold {color}]{command}[/bold {color}]"
    
    def refresh_table(self) -> None:
        """사용자 액션으로 인한 테이블 새로고침 (커서 복원: 이름 기준)"""
        table = self.query_one(EditableDataTable)

        # 1️⃣ 현재 선택된 PC 이름과 컬럼 저장
        sel_name = None
        sel_col  = table.cursor_column
        if 0 <= table.cursor_row < len(self.pc_list):
            sel_name = self.pc_list[table.cursor_row]['name']

        # 2️⃣ 기존 데이터 지우기
        table.clear()

        # 3️⃣ 새 데이터 채우기
        for pc in self.pc_list:
            name = pc['name']
            status_info = self.pc_status.get(name, {})
            conn_icon = "🟢" if status_info.get('connection') == 'Connected' else "🔴"

            status = status_info.get('status', 'Idle')
            if status == 'Running':
                status_display = f"[green]{status}[/green]"
            elif status == 'Stopped':
                status_display = f"[red]{status}[/red]"
            elif status == 'Completed':
                status_display = f"[blue]{status}[/blue]"
            elif status == 'Paused':
                status_display = f"[yellow]{status}[/yellow]"
            else:
                status_display = status

            command_display = self.get_command_display(pc.get('command', 'SceneGen'))

            table.add_row(
                name,
                conn_icon,
                status_display,
                status_info.get('current_scene', '-') ,
                command_display,
                pc.get('env',''),
                pc.get('section',''),
                pc.get('platform',''),
                str(pc.get('scene_start',0)),
                str(pc.get('scene_end',0)),
            )

        # 4️⃣ 커서 복원 (이름 매칭)
        if sel_name:
            for idx, pc in enumerate(self.pc_list):
                if pc['name'] == sel_name:
                    table.move_cursor(row=idx, column=sel_col)
                    break
        table.focus()
    
    def refresh_table_silent(self) -> None:
        """소켓 메시지로 인한 조용한 새로고침 (커서 복원: 이름 기준)"""
        table = self.query_one(EditableDataTable)

        # 커서 있었는지 & 이름 / 컬럼 기억
        had_focus = table.has_focus
        sel_name = None
        sel_col  = table.cursor_column
        if 0 <= table.cursor_row < len(self.pc_list):
            sel_name = self.pc_list[table.cursor_row]['name']

        # 포커스 잠시 제거 (깜빡임 방지)
        if had_focus:
            table.can_focus = False

        # 테이블 재구성 ---------------------------------------------------
        table.clear()
        for pc in self.pc_list:
            name = pc['name']
            status_info = self.pc_status.get(name, {})
            conn_icon = "🟢" if status_info.get('connection') == 'Connected' else "🔴"

            status = status_info.get('status', 'Idle')
            if status == 'Running':
                status_display = f"[green]{status}[/green]"
            elif status == 'Stopped':
                status_display = f"[red]{status}[/red]"
            elif status == 'Completed':
                status_display = f"[blue]{status}[/blue]"
            elif status == 'Paused':
                status_display = f"[yellow]{status}[/yellow]"
            else:
                status_display = status

            command_display = self.get_command_display(pc.get('command', 'SceneGen'))

            table.add_row(
                name,
                conn_icon,
                status_display,
                status_info.get('current_scene','-'),
                command_display,
                pc.get('env',''),
                pc.get('section',''),
                pc.get('platform',''),
                str(pc.get('scene_start',0)),
                str(pc.get('scene_end',0)),
            )

        # 커서 & 포커스 복원 ---------------------------------------------
        if sel_name:
            for idx, pc in enumerate(self.pc_list):
                if pc['name'] == sel_name:
                    table.move_cursor(row=idx, column=sel_col)
                    break
        if had_focus:
            table.focus()
            table.can_focus = True
    def get_selected_pc(self) -> Dict[str, Any] | None:
        """선택된 PC 정보 반환 - 새로고침 중일 때는 저장된 커서 위치 사용"""
        table = self.query_one(EditableDataTable)
        
        # 테이블이 새로고침 중이면 저장된 커서 위치 사용
        if table.preserve_cursor_on_refresh and table.saved_cursor_position:
            cursor_row = table.saved_cursor_position[0]
        else:
            cursor_row = table.cursor_row
            
        if 0 <= cursor_row < len(self.pc_list):
            return self.pc_list[cursor_row]
        return None
    
    def send_command_to_client(self, pc_info: Dict, command_type: str = "start") -> bool:
        """클라이언트에게 명령 전송"""
        client_id = pc_info['name']
        
        if client_id not in self.clients:
            self.notify(f"PC '{client_id}' is not connected", severity="error")
            return False
        
        msg_dict = {
            "cmd": command_type,
            "command": pc_info.get('command', 'SceneGen'),  # Command 항목 추가
            "output_root_path": self.output_root_path,
            "env_name": pc_info.get('env', ''),
            "section_name": pc_info.get('section', ''),
            "platform_name": pc_info.get('platform', ''),
            "scene_start": pc_info.get('scene_start', 0),
            "scene_end": pc_info.get('scene_end', 0),
            "object_num": 5
        }
        
        try:
            send_msg = json.dumps(msg_dict).encode()
            self.clients[client_id].sendall(send_msg)
            return True
        except Exception as e:
            self.notify(f"Failed to send command to {client_id}: {e}", severity="error")
            return False
    
    def check_scene_conflict(self, target_pc: Dict) -> bool:
        """Scene 범위 충돌 검사"""
        target_env = target_pc.get('env', '')
        target_section = target_pc.get('section', '')
        target_platform = target_pc.get('platform', '')
        target_command = target_pc.get('command', 'SceneGen')
        target_start = target_pc.get('scene_start', 0)
        target_end = target_pc.get('scene_end', 0)
        
        if not all([target_env, target_section, target_platform]):
            self.notify("Please set Env, Section, Platform", severity="error")
            return False
            
        if target_start < 0 or target_end < 0:
            self.notify("Scene Start/End must be >= 0", severity="error")
            return False
            
        if target_start > target_end:
            self.notify("Scene Start must be <= Scene End", severity="error")
            return False
        
        for pc in self.pc_list:
            if pc['name'] == target_pc['name']:
                continue
                
            status_info = self.pc_status.get(pc['name'], {})
            if status_info.get('status') != 'Running':
                continue
                
            # command가 같을 때만 충돌 검사 진행
            if (pc.get('env') == target_env and 
                pc.get('section') == target_section and 
                pc.get('platform') == target_platform and
                pc.get('command') == target_command):
                
                pc_start = pc.get('scene_start', 0)
                pc_end = pc.get('scene_end', 0)
                
                if not (target_end < pc_start or target_start > pc_end):
                    self.notify(
                        f"Scene range conflict with {pc['name']} "
                        f"({pc_start}~{pc_end}) - Same command: {target_command}", 
                        severity="error"
                    )
                    return False
        
        return True
    
    # 버튼 이벤트 처리
    def on_button_pressed(self, event: Button.Pressed) -> None:
        pc = self.get_selected_pc()
        if not pc:
            self.notify("No PC selected", severity="warning")
            return
        
        # 새로고침 중일 때 추가 확인
        table = self.query_one(EditableDataTable)
        if table.preserve_cursor_on_refresh:
            self.notify(f"Command sent to: {pc['name']} (saved position)", severity="info")
            
        if event.button.id == "play_btn":
            self.action_play()
        elif event.button.id == "pause_btn":
            self.action_pause()
        elif event.button.id == "stop_btn":
            self.action_stop()
        elif event.button.id == "delete_btn":
            self.action_delete()
    
    # 키보드 액션
    def action_play(self) -> None:
        pc = self.get_selected_pc()
        if not pc:
            return
        
        # 새로고침 중일 때 추가 확인
        table = self.query_one(EditableDataTable)
        if table.preserve_cursor_on_refresh:
            self.notify(f"Play command sent to: {pc['name']} (using saved position)", severity="info")
        
        # 이미 실행 중인 PC인지 확인
        status_info = self.pc_status.get(pc['name'], {})
        current_status = status_info.get('status', 'Idle')
        
        if current_status == 'Running':
            self.notify(f"PC '{pc['name']}' is already running", severity="warning")
            return
            
        if not self.check_scene_conflict(pc):
            return
            
        if self.send_command_to_client(pc, "start"):
            self.pc_status[pc['name']]['status'] = 'Running'
            self.pc_status[pc['name']]['current_scene'] = 'Starting...'
            self.refresh_table()
            self.notify(f"Started {pc['name']}")
    
    def action_pause(self) -> None:
        pc = self.get_selected_pc()
        if not pc:
            return
        
        # 새로고침 중일 때 추가 확인
        table = self.query_one(EditableDataTable)
        if table.preserve_cursor_on_refresh:
            self.notify(f"Pause command sent to: {pc['name']} (using saved position)", severity="info")
            
        if self.send_command_to_client(pc, "pause"):
            self.pc_status[pc['name']]['status'] = 'Paused'
            self.refresh_table()
            self.notify(f"Paused {pc['name']}")
    
    def action_stop(self) -> None:
        pc = self.get_selected_pc()
        if not pc:
            return
        
        # 새로고침 중일 때 추가 확인
        table = self.query_one(EditableDataTable)
        if table.preserve_cursor_on_refresh:
            self.notify(f"Stop command sent to: {pc['name']} (using saved position)", severity="info")
            
        self.send_command_to_client(pc, "stop")
        self.notify(f"Stop command sent to {pc['name']}")
    
    def action_edit(self) -> None:
        """편집 모드 시작"""
        table = self.query_one(EditableDataTable)
        table.start_edit()
    
    def action_delete(self) -> None:
        pc = self.get_selected_pc()
        if not pc:
            return
        
        # 새로고침 중일 때 추가 확인
        table = self.query_one(EditableDataTable)
        if table.preserve_cursor_on_refresh:
            self.notify(f"Delete command for: {pc['name']} (using saved position)", severity="info")
            
        client_id = pc['name']
        if client_id in self.clients:
            try:
                self.clients[client_id].close()
            except:
                pass
            del self.clients[client_id]
        
        self.pc_list.remove(pc)
        if client_id in self.pc_status:
            del self.pc_status[client_id]
        
        self.refresh_table()
        self.notify(f"Deleted {client_id}")
    
    def action_refresh(self) -> None:
        self.refresh_table()
        self.notify("Refreshed")
    
    def action_quit(self) -> None:
        if self.server_socket:
            self.server_socket.close()
        for conn in self.clients.values():
            try:
                conn.close()
            except:
                pass
        self.exit()


if __name__ == "__main__":
    app = MultiPCController()
    app.run()