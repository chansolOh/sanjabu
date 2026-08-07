#!/usr/bin/env python3
"""
Data Structure Monitor - TUI Version
데이터셋 구조를 실시간으로 모니터링하는 터미널 기반 인터페이스
"""

import os
import json
import time
import threading
import sys
from pathlib import Path
from datetime import datetime
import argparse

try:
    import curses
    from curses import panel
except ImportError:
    print("curses module not available. Install windows-curses on Windows: pip install windows-curses")
    sys.exit(1)

class DataMonitorTUI:
    def __init__(self, stdscr, initial_path=None):
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        
        # 초기화
        curses.curs_set(0)  # 커서 숨기기
        curses.use_default_colors()
        
        # 색상 초기화
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, -1)    # 정상 상태
        curses.init_pair(2, curses.COLOR_RED, -1)      # 에러/누락
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # 경고
        curses.init_pair(4, curses.COLOR_CYAN, -1)     # 정보
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # 하이라이트
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)  # 선택됨
        
        # 상태 변수
        self.monitor_path = initial_path or "/nas/Dataset/Dataset_2026/dataset_v2"
        self.monitoring = False
        self.monitor_thread = None
        self.data_structure = {}
        self.tree_items = []
        self.selected_index = 0
        self.scroll_offset = 0
        self.expanded_items = set()
        self.last_update = None
        
        # 총 데이터 개수를 폴더별로 관리
        self.total_data_count = {
            'conf': 0,
            'pre_grasp': 0,
            'output_grasp': 0,
            'total': 0
        }
        
        # 화면 영역 정의
        self.header_height = 5  # 헤더 높이 증가 (폴더별 데이터 개수 표시 위해)
        self.footer_height = 3
        self.tree_height = self.height - self.header_height - self.footer_height - 2
        self.detail_start_row = self.height - self.footer_height - 1
        
        # 제외할 platform 목록
        self.excluded_platforms = {'rgb', 'depth', 'inst_seg', 'normals', 'bbox', 'pointcloud'}
        
    def run(self):
        """메인 실행 루프"""
        try:
            while True:
                self.draw_screen()
                key = self.stdscr.getch()
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    self.toggle_monitoring()
                elif key == ord('r') or key == ord('R'):
                    self.refresh_data()
                elif key == ord('p') or key == ord('P'):
                    self.change_path()
                elif key == ord('e') or key == ord('E'):
                    self.export_structure()
                elif key == ord(' '):
                    self.toggle_expand()
                elif key == curses.KEY_UP:
                    self.move_selection(-1)
                elif key == curses.KEY_DOWN:
                    self.move_selection(1)
                elif key == curses.KEY_LEFT:
                    self.collapse_item()
                elif key == curses.KEY_RIGHT:
                    self.expand_item()
                elif key == curses.KEY_ENTER or key == 10:
                    self.toggle_expand()
                elif key == curses.KEY_RESIZE:
                    self.handle_resize()
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_monitoring()
            
    def draw_screen(self):
        """화면 그리기"""
        self.stdscr.clear()
        
        # 헤더 그리기
        self.draw_header()
        
        # 트리 그리기
        self.draw_tree()
        
        # 상세 정보 그리기
        self.draw_details()
        
        # 푸터 그리기
        self.draw_footer()
        
        self.stdscr.refresh()
        
    def draw_header(self):
        """헤더 영역 그리기"""
        title = "Data Structure Monitor - TUI"
        self.stdscr.addstr(0, (self.width - len(title)) // 2, title, 
                          curses.color_pair(4) | curses.A_BOLD)
        
        # 경로와 상태 정보
        path_str = f"Path: {self.monitor_path}"
        if len(path_str) > self.width - 2:
            path_str = "..." + path_str[-(self.width-5):]
        self.stdscr.addstr(1, 1, path_str)
        
        status_color = curses.color_pair(1) if self.monitoring else curses.color_pair(2)
        status_text = "MONITORING" if self.monitoring else "STOPPED"
        status_full = f"Status: {status_text}"
        
        if self.last_update:
            status_full += f" | Last Update: {self.last_update.strftime('%H:%M:%S')}"
            
        self.stdscr.addstr(2, 1, status_full, status_color)
        
        # 폴더별 총 데이터 개수 표시
        conf_count = self.total_data_count['conf']
        pre_grasp_count = self.total_data_count['pre_grasp']
        output_grasp_count = self.total_data_count['output_grasp']
        total_count = self.total_data_count['total']
        
        # 각 폴더별로 다른 색상 적용
        self.stdscr.addstr(3, 1, "Total Files - ", curses.color_pair(4))
        
        # Conf 폴더 (파란색)
        self.stdscr.addstr(3, 15, f"Conf: {conf_count:,}", curses.color_pair(4) | curses.A_BOLD)
        self.stdscr.addstr(3, 15 + len(f"Conf: {conf_count:,}"), " | ", curses.color_pair(4))
        
        # PreGrasp 폴더 (자홍색)
        pre_start = 15 + len(f"Conf: {conf_count:,}") + 3
        self.stdscr.addstr(3, pre_start, f"PreGrasp: {pre_grasp_count:,}", curses.color_pair(5) | curses.A_BOLD)
        self.stdscr.addstr(3, pre_start + len(f"PreGrasp: {pre_grasp_count:,}"), " | ", curses.color_pair(4))
        
        # OutputGrasp 폴더 (초록색)
        out_start = pre_start + len(f"PreGrasp: {pre_grasp_count:,}") + 3
        self.stdscr.addstr(3, out_start, f"OutputGrasp: {output_grasp_count:,}", curses.color_pair(1) | curses.A_BOLD)
        self.stdscr.addstr(3, out_start + len(f"OutputGrasp: {output_grasp_count:,}"), " | ", curses.color_pair(4))
        
        # Total (노란색)
        total_start = out_start + len(f"OutputGrasp: {output_grasp_count:,}") + 3
        self.stdscr.addstr(3, total_start, f"Total: {total_count:,}", curses.color_pair(3) | curses.A_BOLD)
        
        # 구분선
        self.stdscr.addstr(self.header_height, 0, "─" * self.width)
        
    def draw_tree(self):
        """트리 구조 그리기"""
        if not self.tree_items:
            msg = "No data loaded. Press 'S' to start monitoring or 'R' to refresh."
            start_row = self.header_height + (self.tree_height // 2)
            self.stdscr.addstr(start_row, (self.width - len(msg)) // 2, msg, 
                              curses.color_pair(3))
            return
            
        # 표시할 항목들 계산
        visible_items = self.tree_items[self.scroll_offset:self.scroll_offset + self.tree_height]
        
        for i, item in enumerate(visible_items):
            row = self.header_height + 1 + i
            if row >= self.detail_start_row - 1:
                break
                
            item_index = self.scroll_offset + i
            is_selected = item_index == self.selected_index
            
            # 선택된 항목 하이라이트
            if is_selected:
                self.stdscr.addstr(row, 0, " " * self.width, curses.color_pair(6))
                
            # 들여쓰기와 확장/축소 표시
            indent = "  " * item['level']
            if item['type'] in ['environment', 'section', 'platform']:
                expand_char = "▼" if item['path'] in self.expanded_items else "▶"
                prefix = f"{indent}{expand_char} "
            else:  # missing_info, complete_info, no_folder_info, empty_folder_info
                prefix = f"{indent}  "
                
            # 항목 텍스트
            if item['type'] in ['missing_info', 'complete_info', 'no_folder_info', 'empty_folder_info']:
                # 폴더 상태별 상세 정보 표시
                if item['type'] == 'missing_info':
                    missing_files = item['missing_files']
                    if missing_files:
                        # 처음 10개만 표시
                        file_list = ', '.join([f"{f:04d}" for f in missing_files[:10]])
                        if len(missing_files) > 10:
                            file_list += f" ... (+{len(missing_files)-10} more)"
                        text = f"{prefix}{file_list}"
                    else:
                        text = f"{prefix}No missing files"
                elif item['type'] == 'complete_info':
                    text = f"{prefix}All files present"
                elif item['type'] == 'no_folder_info':
                    text = f"{prefix}Folder does not exist"
                elif item['type'] == 'empty_folder_info':
                    text = f"{prefix}Folder exists but no JSON files"
            else:
                text = f"{prefix}{item['name']}"
            
            # 타입별 색상
            if item['type'] == 'environment':
                color = curses.color_pair(4) | curses.A_BOLD
            elif item['type'] == 'section':
                color = curses.color_pair(5)
            elif item['type'] == 'platform':
                # platform은 항상 기본 색상 (missing 정보는 하위 항목에서 처리)
                color = curses.color_pair(1)
            elif item['type'] == 'missing_info':
                # missing 정보는 빨간색으로 표시
                color = curses.color_pair(2)
            elif item['type'] == 'complete_info':
                # complete 정보는 초록색으로 표시
                color = curses.color_pair(1)
            elif item['type'] == 'no_folder_info':
                # 폴더 없음은 회색/어두운 색으로 표시
                color = curses.color_pair(3)
            elif item['type'] == 'empty_folder_info':
                # 빈 폴더는 노란색으로 표시
                color = curses.color_pair(3)
            else:
                color = curses.color_pair(1)
                
            if is_selected:
                color |= curses.A_REVERSE
                
            # 텍스트 길이 제한 (정보 표시 공간 확보)
            max_text_width = min(40, self.width - 80)  # 공간 조정
            if len(text) > max_text_width:
                text = text[:max_text_width-3] + "..."
                
            self.stdscr.addstr(row, 0, text.ljust(max_text_width), color)
            
            # 추가 정보 (정렬된 형태로 표시)
            info_x = max_text_width + 2
            remaining_width = self.width - info_x - 2
            
            if item['type'] == 'platform':
                # Platform 정보: 3개 폴더별 파일 개수 표시 (색상 분리를 위해 개별 처리)
                conf_count = item['folder_counts']['conf']
                pre_grasp_count = item['folder_counts']['pre_grasp']
                output_grasp_count = item['folder_counts']['output_grasp']
                
                # 색상을 개별적으로 적용하기 위해 info_text는 None으로 설정
                info_text = None
                
                # 개별 색상으로 표시
                if info_x < self.width - 50:  # 충분한 공간이 있을 때만
                    # 파일 개수 상태에 따른 상태 표시등 결정
                    if conf_count == pre_grasp_count == output_grasp_count:
                        # 모든 폴더 개수가 동일 → 초록 불
                        status_indicator = "●"
                        status_color = curses.color_pair(1)  # 초록색
                    elif conf_count == pre_grasp_count:
                        # conf와 pre_grasp만 동일 → 주황 불
                        status_indicator = "●"
                        status_color = curses.color_pair(3)  # 주황색(노란색)
                    else:
                        # 모두 다름 → 빨간 불
                        status_indicator = "●"
                        status_color = curses.color_pair(2)  # 빨간색
                    
                    # 상태 표시등 표시
                    if not is_selected:
                        self.stdscr.addstr(row, info_x, status_indicator + " ", status_color)
                    else:
                        self.stdscr.addstr(row, info_x, status_indicator + " ", status_color | curses.A_REVERSE)
                    
                    # 상태 표시등 다음 위치부터 파일 정보 표시
                    info_start_x = info_x + 2
                    
                    # Conf (청록색)
                    conf_text = f"Conf: {conf_count:4d}"
                    conf_color = curses.color_pair(4) if not is_selected else curses.color_pair(4) | curses.A_REVERSE
                    self.stdscr.addstr(row, info_start_x, conf_text, conf_color)
                    
                    # 구분자
                    sep_x = info_start_x + len(conf_text)
                    sep_color = curses.color_pair(1) if not is_selected else curses.color_pair(1) | curses.A_REVERSE
                    self.stdscr.addstr(row, sep_x, " | ", sep_color)
                    
                    # PreG (자홍색)
                    preg_x = sep_x + 3
                    preg_text = f"PreG: {pre_grasp_count:4d}"
                    preg_color = curses.color_pair(5) if not is_selected else curses.color_pair(5) | curses.A_REVERSE
                    self.stdscr.addstr(row, preg_x, preg_text, preg_color)
                    
                    # 구분자
                    sep2_x = preg_x + len(preg_text)
                    self.stdscr.addstr(row, sep2_x, " | ", sep_color)
                    
                    # OutG (초록색)
                    outg_x = sep2_x + 3
                    outg_text = f"OutG: {output_grasp_count:4d}"
                    outg_color = curses.color_pair(1) if not is_selected else curses.color_pair(1) | curses.A_REVERSE
                    self.stdscr.addstr(row, outg_x, outg_text, outg_color)
                
            elif item['type'] in ['missing_info', 'complete_info', 'no_folder_info', 'empty_folder_info']:
                # 폴더 상태별 정보 표시
                folder_name = item['folder_name']
                if item['type'] == 'missing_info':
                    missing_count = len(item['missing_files'])
                    info_text = f"{folder_name} Missing: {missing_count:3d} files"
                elif item['type'] == 'complete_info':
                    file_count = item.get('file_count', 0)
                    info_text = f"{folder_name} Complete: {file_count:3d} files"
                elif item['type'] == 'no_folder_info':
                    info_text = f"{folder_name}: Folder not exists"
                elif item['type'] == 'empty_folder_info':
                    info_text = f"{folder_name}: Empty folder"
                
            elif item['type'] == 'section':
                # Section 정보: 폴더별 파일 개수 (색상 분리를 위해 개별 처리)
                conf_count = item['folder_counts']['conf']
                pre_grasp_count = item['folder_counts']['pre_grasp']
                output_grasp_count = item['folder_counts']['output_grasp']
                total_files = item['folder_counts']['total']
                
                # 색상을 개별적으로 적용하기 위해 info_text는 None으로 설정
                info_text = None
                
                # 개별 색상으로 표시
                if info_x < self.width - 70:  # 충분한 공간이 있을 때만
                    # Conf (청록색)
                    conf_text = f"Conf: {conf_count:5d}"
                    conf_color = curses.color_pair(4) if not is_selected else curses.color_pair(4) | curses.A_REVERSE
                    self.stdscr.addstr(row, info_x, conf_text, conf_color)
                    
                    # 구분자
                    sep_x = info_x + len(conf_text)
                    sep_color = curses.color_pair(5) if not is_selected else curses.color_pair(5) | curses.A_REVERSE
                    self.stdscr.addstr(row, sep_x, " | ", sep_color)
                    
                    # PreG (자홍색)
                    preg_x = sep_x + 3
                    preg_text = f"PreG: {pre_grasp_count:5d}"
                    preg_color = curses.color_pair(5) if not is_selected else curses.color_pair(5) | curses.A_REVERSE
                    self.stdscr.addstr(row, preg_x, preg_text, preg_color)
                    
                    # 구분자
                    sep2_x = preg_x + len(preg_text)
                    self.stdscr.addstr(row, sep2_x, " | ", sep_color)
                    
                    # OutG (초록색)
                    outg_x = sep2_x + 3
                    outg_text = f"OutG: {output_grasp_count:5d}"
                    outg_color = curses.color_pair(1) if not is_selected else curses.color_pair(1) | curses.A_REVERSE
                    self.stdscr.addstr(row, outg_x, outg_text, outg_color)
                    
                    # 구분자
                    sep3_x = outg_x + len(outg_text)
                    self.stdscr.addstr(row, sep3_x, " | ", sep_color)
                    
                    # Total (노란색)
                    total_x = sep3_x + 3
                    total_text = f"Total: {total_files:6d}"
                    total_color = curses.color_pair(3) if not is_selected else curses.color_pair(3) | curses.A_REVERSE
                    self.stdscr.addstr(row, total_x, total_text, total_color)
                
            elif item['type'] == 'environment':
                # Environment 정보: 폴더별 파일 개수 (색상 분리를 위해 개별 처리)
                conf_count = item['folder_counts']['conf']
                pre_grasp_count = item['folder_counts']['pre_grasp']
                output_grasp_count = item['folder_counts']['output_grasp']
                total_files = item['folder_counts']['total']
                
                # 색상을 개별적으로 적용하기 위해 info_text는 None으로 설정
                info_text = None
                
                # 개별 색상으로 표시
                if info_x < self.width - 70:  # 충분한 공간이 있을 때만
                    # Conf (청록색)
                    conf_text = f"Conf: {conf_count:5d}"
                    conf_color = curses.color_pair(4) if not is_selected else curses.color_pair(4) | curses.A_REVERSE
                    self.stdscr.addstr(row, info_x, conf_text, conf_color)
                    
                    # 구분자
                    sep_x = info_x + len(conf_text)
                    sep_color = curses.color_pair(4) if not is_selected else curses.color_pair(4) | curses.A_REVERSE
                    self.stdscr.addstr(row, sep_x, " | ", sep_color)
                    
                    # PreG (자홍색)
                    preg_x = sep_x + 3
                    preg_text = f"PreG: {pre_grasp_count:5d}"
                    preg_color = curses.color_pair(5) if not is_selected else curses.color_pair(5) | curses.A_REVERSE
                    self.stdscr.addstr(row, preg_x, preg_text, preg_color)
                    
                    # 구분자
                    sep2_x = preg_x + len(preg_text)
                    self.stdscr.addstr(row, sep2_x, " | ", sep_color)
                    
                    # OutG (초록색)
                    outg_x = sep2_x + 3
                    outg_text = f"OutG: {output_grasp_count:5d}"
                    outg_color = curses.color_pair(1) if not is_selected else curses.color_pair(1) | curses.A_REVERSE
                    self.stdscr.addstr(row, outg_x, outg_text, outg_color)
                    
                    # 구분자
                    sep3_x = outg_x + len(outg_text)
                    self.stdscr.addstr(row, sep3_x, " | ", sep_color)
                    
                    # Total (노란색)
                    total_x = sep3_x + 3
                    total_text = f"Total: {total_files:6d}"
                    total_color = curses.color_pair(3) if not is_selected else curses.color_pair(3) | curses.A_REVERSE
                    self.stdscr.addstr(row, total_x, total_text, total_color)
            
            # 정보 텍스트 표시 (개별 색상 처리되지 않은 항목들만)
            if info_text and info_x + len(info_text) < self.width:
                if item['type'] == 'missing_info':
                    info_color = curses.color_pair(2)  # 빨간색
                elif item['type'] == 'complete_info':
                    info_color = curses.color_pair(1)  # 초록색
                elif item['type'] in ['no_folder_info', 'empty_folder_info']:
                    info_color = curses.color_pair(3)  # 노란색
                else:
                    info_color = curses.color_pair(3)  # 기본 정보 색상
                    
                if is_selected:
                    info_color |= curses.A_REVERSE
                self.stdscr.addstr(row, info_x, info_text[:remaining_width], info_color)
                        
    def draw_details(self):
        """상세 정보 영역 그리기"""
        # 구분선
        self.stdscr.addstr(self.detail_start_row - 1, 0, "─" * self.width)
        
        if not self.tree_items or self.selected_index >= len(self.tree_items):
            return
            
        item = self.tree_items[self.selected_index]
        
        details = []
        details.append(f"Selected: {item['name']} ({item['type']})")
        details.append(f"Path: {item['path']}")
        
        if item['type'] == 'platform':
            # 3개 폴더별 상세 정보
            conf_data = item['conf_data']
            pre_grasp_data = item['pre_grasp_data'] 
            output_grasp_data = item['output_grasp_data']
            
            # 파일 개수 상태 확인
            conf_count = conf_data.get('file_count', 0)
            pre_grasp_count = pre_grasp_data.get('file_count', 0)
            output_grasp_count = output_grasp_data.get('file_count', 0)
            
            # 상태 메시지 결정
            if conf_count == pre_grasp_count == output_grasp_count:
                status_msg = "● All folders have equal file counts"
                status_type = 'complete'
            elif conf_count == pre_grasp_count:
                status_msg = "● Conf and PreGrasp have equal file counts"
                status_type = 'partial'
            else:
                status_msg = "● File counts differ across folders"
                status_type = 'different'
            
            # 폴더별 파일 개수를 색상과 함께 표시하기 위해 특별 처리
            details.append("Platform Files:")
            details.append(f"Conf: {conf_count} | PreGrasp: {pre_grasp_count} | OutputGrasp: {output_grasp_count}")
            details.append(status_msg)  # 상태 메시지 추가
            
            # 각 폴더별 missing 파일 요약
            conf_missing = len(conf_data.get('missing', []))
            pre_missing = len(pre_grasp_data.get('missing', []))
            out_missing = len(output_grasp_data.get('missing', []))
            
            details.append(f"Missing - Conf: {conf_missing} | PreGrasp: {pre_missing} | OutputGrasp: {out_missing}")
            
        elif item['type'] in ['missing_info', 'complete_info', 'no_folder_info', 'empty_folder_info']:
            # 폴더 상태별 상세 정보 표시
            folder_name = item['folder_name']
            missing_files = item.get('missing_files', [])
            file_count = item.get('file_count', 0)
            folder_range = item.get('folder_range', '')
            
            if item['type'] == 'missing_info' and missing_files:
                details.append(f"{folder_name} Missing Files: {len(missing_files)} total")
                
                # 모든 missing 파일 번호 표시 (20개씩 줄바꿈)
                file_chunks = [missing_files[i:i+20] for i in range(0, len(missing_files), 20)]
                for chunk in file_chunks[:3]:  # 최대 3줄까지만 표시
                    file_list = ', '.join([f"{f:04d}" for f in chunk])
                    details.append(f"  {file_list}")
                if len(file_chunks) > 3:
                    details.append(f"  ... and {len(missing_files) - 60} more files")
            elif item['type'] == 'complete_info':
                details.append(f"{folder_name}: All {file_count} files are present")
            elif item['type'] == 'no_folder_info':
                details.append(f"{folder_name}: Folder does not exist in the platform directory")
            elif item['type'] == 'empty_folder_info':
                details.append(f"{folder_name}: Folder exists but contains no JSON files")
                    
        elif item['type'] in ['section', 'environment']:
            folder_counts = item['folder_counts']
            # 폴더별 색상을 적용한 상세 정보 표시
            detail_text = "Files - "
            details.append(detail_text + f"Conf: {folder_counts['conf']:,} | PreGrasp: {folder_counts['pre_grasp']:,} | OutputGrasp: {folder_counts['output_grasp']:,} | Total: {folder_counts['total']:,}")
                
        # 상세 정보 표시
        for i, detail in enumerate(details):
            if i < self.footer_height - 1:
                row = self.detail_start_row + i
                if i == len(details) - 1 and item['type'] in ['section', 'environment']:
                    # 마지막 줄이고 section/environment인 경우 폴더별 색상 적용
                    base_text = "Files - "
                    self.stdscr.addstr(row, 1, base_text)
                    
                    x_pos = 1 + len(base_text)
                    folder_counts = item['folder_counts']
                    
                    # Conf (청록색)
                    conf_text = f"Conf: {folder_counts['conf']:,}"
                    self.stdscr.addstr(row, x_pos, conf_text, curses.color_pair(4))
                    x_pos += len(conf_text)
                    
                    # 구분자
                    self.stdscr.addstr(row, x_pos, " | ")
                    x_pos += 3
                    
                    # PreGrasp (자홍색)
                    pre_text = f"PreGrasp: {folder_counts['pre_grasp']:,}"
                    self.stdscr.addstr(row, x_pos, pre_text, curses.color_pair(5))
                    x_pos += len(pre_text)
                    
                    # 구분자
                    self.stdscr.addstr(row, x_pos, " | ")
                    x_pos += 3
                    
                    # OutputGrasp (초록색)
                    out_text = f"OutputGrasp: {folder_counts['output_grasp']:,}"
                    self.stdscr.addstr(row, x_pos, out_text, curses.color_pair(1))
                    x_pos += len(out_text)
                    
                    # 구분자
                    self.stdscr.addstr(row, x_pos, " | ")
                    x_pos += 3
                    
                    # Total (노란색)
                    total_text = f"Total: {folder_counts['total']:,}"
                    self.stdscr.addstr(row, x_pos, total_text, curses.color_pair(3))
                else:
                    self.stdscr.addstr(row, 1, detail[:self.width-2])
                
    def draw_footer(self):
        """푸터 영역 그리기"""
        footer_row = self.height - 1
        
        # 키 바인딩 도움말
        help_text = "Q:Quit S:Start/Stop R:Refresh P:Path E:Export Space:Expand ↑↓:Navigate"
        if len(help_text) > self.width:
            help_text = "Q:Quit S:Monitor R:Refresh E:Export ↑↓:Navigate"
            
        self.stdscr.addstr(footer_row, 0, help_text[:self.width], curses.color_pair(4))
        
    def toggle_monitoring(self):
        """모니터링 시작/중지"""
        if not self.monitoring:
            if not os.path.exists(self.monitor_path):
                self.show_message("Error: Invalid path!", curses.color_pair(2))
                return
                
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
        else:
            self.monitoring = False
            
    def stop_monitoring(self):
        """모니터링 완전 중지"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
            
    def monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                self.scan_and_update()
                time.sleep(2)  # 2초마다 업데이트
            except Exception as e:
                # 에러는 조용히 넘어감 (실제 구현에서는 로깅 추가)
                time.sleep(5)
                
    def scan_and_update(self):
        """데이터 스캔 및 업데이트"""
        if not os.path.exists(self.monitor_path):
            return
            
        # 현재 선택 상태 저장
        current_selection_path = None
        if self.tree_items and self.selected_index < len(self.tree_items):
            current_selection_path = self.tree_items[self.selected_index]['path']
            
        # 데이터 스캔
        self.data_structure = self.scan_structure(self.monitor_path)
        self.build_tree_items()
        
        # 선택 상태 복원
        if current_selection_path:
            for i, item in enumerate(self.tree_items):
                if item['path'] == current_selection_path:
                    self.selected_index = i
                    break
                    
        self.last_update = datetime.now()
        
    def scan_structure(self, base_path):
        """데이터 구조 스캔"""
        structure = {}
        
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
                        if platform in self.excluded_platforms:
                            continue
                            
                        platform_path = os.path.join(section_path, platform)
                        if not os.path.isdir(platform_path):
                            continue
                        
                        # 각 폴더별 JSON 파일 스캔
                        platform_data = {}
                        folders_to_scan = ['conf', 'pre_grasp', 'output_grasp']
                        
                        for folder_name in folders_to_scan:
                            folder_path = os.path.join(platform_path, folder_name)
                            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                                json_files = self.scan_json_files_in_folder(folder_path)
                                platform_data[folder_name] = json_files
                            else:
                                platform_data[folder_name] = {
                                    'files': [], 'range': 'No folder', 'missing': [], 'file_count': 0
                                }
                        
                        structure[env][section][platform] = platform_data
                            
        except Exception as e:
            pass  # 에러 무시
            
        return structure
        
    def scan_json_files_in_folder(self, folder_path):
        """JSON 파일 스캔"""
        files = []
        try:
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    try:
                        num = int(file.split('.')[0])
                        files.append(num)
                    except ValueError:
                        continue
        except Exception:
            pass
            
        files.sort()
        
        if not files:
            return {'files': [], 'range': 'No JSON files', 'missing': [], 'file_count': 0}
            
        min_num = min(files)
        max_num = max(files)
        expected = set(range(min_num, max_num + 1))
        actual = set(files)
        missing = sorted(list(expected - actual))
        
        range_str = f"{min_num:04d}-{max_num:04d}" if min_num != max_num else f"{min_num:04d}"
        
        return {
            'files': files,
            'range': range_str,
            'missing': missing,
            'file_count': len(files)
        }
        
    def build_tree_items(self):
        """트리 아이템 목록 구성"""
        self.tree_items = []
        
        # 전체 데이터 개수 초기화
        self.total_data_count = {
            'conf': 0,
            'pre_grasp': 0,
            'output_grasp': 0,
            'total': 0
        }
        
        for env, sections in self.data_structure.items():
            env_path = env
            
            # Environment 레벨의 폴더별 파일 수 계산
            env_folder_counts = {
                'conf': 0,
                'pre_grasp': 0,
                'output_grasp': 0,
                'total': 0
            }
            
            for section, platforms in sections.items():
                for platform, folder_data in platforms.items():
                    # 3개 폴더의 파일 수 각각 계산
                    for folder_name in ['conf', 'pre_grasp', 'output_grasp']:
                        if folder_name in folder_data:
                            count = folder_data[folder_name].get('file_count', 0)
                            env_folder_counts[folder_name] += count
                            env_folder_counts['total'] += count
            
            # 전체 데이터 카운트에 추가
            for folder_name in ['conf', 'pre_grasp', 'output_grasp', 'total']:
                self.total_data_count[folder_name] += env_folder_counts[folder_name]
            
            self.tree_items.append({
                'name': env,
                'type': 'environment',
                'level': 0,
                'path': env_path,
                'folder_counts': env_folder_counts
            })
            
            if env_path in self.expanded_items:
                for section, platforms in sections.items():
                    section_path = f"{env}/{section}"
                    
                    # Section 레벨의 폴더별 파일 수 계산
                    section_folder_counts = {
                        'conf': 0,
                        'pre_grasp': 0,
                        'output_grasp': 0,
                        'total': 0
                    }
                    
                    for platform, folder_data in platforms.items():
                        for folder_name in ['conf', 'pre_grasp', 'output_grasp']:
                            if folder_name in folder_data:
                                count = folder_data[folder_name].get('file_count', 0)
                                section_folder_counts[folder_name] += count
                                section_folder_counts['total'] += count
                    
                    self.tree_items.append({
                        'name': section,
                        'type': 'section', 
                        'level': 1,
                        'path': section_path,
                        'folder_counts': section_folder_counts
                    })
                    
                    if section_path in self.expanded_items:
                        for platform, folder_data in platforms.items():
                            platform_path = f"{env}/{section}/{platform}"
                            
                            # 3개 폴더의 데이터 추출
                            conf_data = folder_data.get('conf', {})
                            pre_grasp_data = folder_data.get('pre_grasp', {})
                            output_grasp_data = folder_data.get('output_grasp', {})
                            
                            # 플랫폼 레벨의 폴더별 파일 수 계산
                            platform_folder_counts = {
                                'conf': conf_data.get('file_count', 0),
                                'pre_grasp': pre_grasp_data.get('file_count', 0),
                                'output_grasp': output_grasp_data.get('file_count', 0),
                                'total': (conf_data.get('file_count', 0) + 
                                         pre_grasp_data.get('file_count', 0) + 
                                         output_grasp_data.get('file_count', 0))
                            }
                            
                            self.tree_items.append({
                                'name': platform,
                                'type': 'platform',
                                'level': 2,
                                'path': platform_path,
                                'conf_data': conf_data,
                                'pre_grasp_data': pre_grasp_data,
                                'output_grasp_data': output_grasp_data,
                                'folder_counts': platform_folder_counts
                            })
                            
                            # Platform이 확장된 경우 missing 정보 표시
                            if platform_path in self.expanded_items:
                                folders = [
                                    ('Conf', conf_data),
                                    ('PreGrasp', pre_grasp_data), 
                                    ('OutputGrasp', output_grasp_data)
                                ]
                                
                                for folder_name, folder_info in folders:
                                    missing_files = folder_info.get('missing', [])
                                    file_count = folder_info.get('file_count', 0)
                                    folder_range = folder_info.get('range', '')
                                    
                                    missing_path = f"{platform_path}/{folder_name}_missing"
                                    
                                    # 폴더 상태 판단
                                    if folder_range == 'No folder':
                                        display_name = f"{folder_name} (No folder)"
                                        color_type = 'no_folder_info'
                                    elif file_count == 0:
                                        display_name = f"{folder_name} (Empty - 0 files)"
                                        color_type = 'empty_folder_info'
                                    elif missing_files:
                                        display_name = f"{folder_name} Missing ({len(missing_files)} files)"
                                        color_type = 'missing_info'
                                    else:
                                        display_name = f"{folder_name} Complete ({file_count} files)"
                                        color_type = 'complete_info'
                                    
                                    self.tree_items.append({
                                        'name': display_name,
                                        'type': color_type,
                                        'level': 3,
                                        'path': missing_path,
                                        'folder_name': folder_name,
                                        'missing_files': missing_files,
                                        'file_count': file_count,
                                        'folder_range': folder_range
                                    })
                            
    def move_selection(self, delta):
        """선택 이동"""
        if not self.tree_items:
            return
            
        self.selected_index = max(0, min(len(self.tree_items) - 1, 
                                        self.selected_index + delta))
        
        # 스크롤 조정
        if self.selected_index < self.scroll_offset:
            self.scroll_offset = self.selected_index
        elif self.selected_index >= self.scroll_offset + self.tree_height:
            self.scroll_offset = self.selected_index - self.tree_height + 1
            
    def toggle_expand(self):
        """현재 항목 확장/축소"""
        if not self.tree_items or self.selected_index >= len(self.tree_items):
            return
            
        item = self.tree_items[self.selected_index]
        if item['type'] in ['missing_info', 'complete_info', 'no_folder_info', 'empty_folder_info']:
            return
            
        path = item['path']
        if path in self.expanded_items:
            self.expanded_items.remove(path)
        else:
            self.expanded_items.add(path)
            
        self.build_tree_items()
        
    def expand_item(self):
        """현재 항목 확장"""
        if not self.tree_items or self.selected_index >= len(self.tree_items):
            return
            
        item = self.tree_items[self.selected_index]
        if item['type'] not in ['missing_info', 'complete_info', 'no_folder_info', 'empty_folder_info']:
            self.expanded_items.add(item['path'])
            self.build_tree_items()
            
    def collapse_item(self):
        """현재 항목 축소"""
        if not self.tree_items or self.selected_index >= len(self.tree_items):
            return
            
        item = self.tree_items[self.selected_index]
        if item['type'] not in ['missing_info', 'complete_info', 'no_folder_info', 'empty_folder_info']:
            self.expanded_items.discard(item['path'])
            self.build_tree_items()
            
    def refresh_data(self):
        """데이터 새로고침"""
        self.scan_and_update()
        
    def change_path(self):
        """모니터링 경로 변경"""
        self.stdscr.clear()
        self.stdscr.addstr(self.height//2, 2, "Enter new path: ")
        self.stdscr.refresh()
        
        curses.curs_set(1)  # 커서 표시
        curses.echo()       # 입력 에코
        
        try:
            new_path = self.stdscr.getstr(self.height//2, 18, 80).decode('utf-8').strip()
            if new_path and os.path.exists(new_path):
                self.monitor_path = new_path
                self.refresh_data()
            elif new_path:
                self.show_message("Invalid path!", curses.color_pair(2))
        except:
            pass
        finally:
            curses.noecho()
            curses.curs_set(0)
            
    def export_structure(self):
        """구조 정보 내보내기"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'monitor_path': self.monitor_path,
                'total_data_count': self.total_data_count,
                'structure': self.data_structure
            }
            
            filename = f"data_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.show_message(f"Exported to {filename}", curses.color_pair(1))
            
        except Exception as e:
            self.show_message(f"Export failed: {str(e)}", curses.color_pair(2))
            
    def show_message(self, message, color_pair):
        """메시지 표시"""
        # 메시지를 푸터 위에 잠깐 표시
        msg_row = self.height - 2
        self.stdscr.addstr(msg_row, 1, " " * (self.width - 2))  # 줄 지우기
        self.stdscr.addstr(msg_row, 1, message[:self.width-2], color_pair)
        self.stdscr.refresh()
        
        # 별도 스레드에서 메시지 지우기
        def clear_message():
            time.sleep(2)
            try:
                self.stdscr.addstr(msg_row, 1, " " * (self.width - 2))
                self.stdscr.refresh()
            except:
                pass
                
        threading.Thread(target=clear_message, daemon=True).start()
        
    def handle_resize(self):
        """화면 크기 변경 처리"""
        self.height, self.width = self.stdscr.getmaxyx()
        self.tree_height = self.height - self.header_height - self.footer_height - 2
        self.detail_start_row = self.height - self.footer_height - 1
        
        # 스크롤 오프셋 조정
        if self.selected_index >= self.scroll_offset + self.tree_height:
            self.scroll_offset = max(0, self.selected_index - self.tree_height + 1)


def main():
    parser = argparse.ArgumentParser(description="Data Structure Monitor - TUI")
    parser.add_argument("--path", "-p", default="/nas/Dataset/Dataset_2026/dataset_v2",
                       help="Initial monitoring path")
    parser.add_argument("--auto-start", "-a", action="store_true",
                       help="Start monitoring automatically")
    
    args = parser.parse_args()
    
    def run_tui(stdscr):
        monitor = DataMonitorTUI(stdscr, args.path)
        
        if args.auto_start:
            monitor.toggle_monitoring()
            
        monitor.run()
        
    try:
        curses.wrapper(run_tui)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()