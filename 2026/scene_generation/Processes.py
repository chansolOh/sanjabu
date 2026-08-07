
import subprocess
import threading
import time
import json
import os
import signal
import numpy as np


def _start_python_process(python_path, target_path, arguments, workdir=None):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [python_path, "-u", target_path, *arguments],
        cwd=workdir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name != "nt"),
    )


def _stop_process_group(process):
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        process.kill()
        process.wait(timeout=10)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def get_scenegen_pid(process_name=""):
   try:
       # Windows의 경우
       result = subprocess.run(['tasklist'], capture_output=True, text=True)
       for line in result.stdout.split('\n'):
           if process_name in line:
               pid = line.split()[1]
               return int(pid)
   except:
       try:
           # Linux/Mac의 경우
           result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
           for line in result.stdout.split('\n'):
               if process_name in line:
                   pid = line.split()[1]
                   return int(pid)
       except:
           pass
   
   return None


class MainProcess_SceneGen:
    def __init__(self, 
                 python_path, 
                 target_path, 
                 reset_time_th,
                 scene_gen_time_th,
                 sock,
                 pid_name,
                 workdir=None,):
        self.stage = "INIT"
        self.timer = time.time()
        self.python_path = python_path
        self.target_path = target_path
        self.reset_time_th = reset_time_th
        self.scene_gen_time_th = scene_gen_time_th
        self.sock = sock
        self.process = None
        self.pid_name = pid_name
        self.workdir = workdir
        self.restart_requested = False


    def start(self, 
              output_root_path,
              env_name,
              section_name,
              platform_name,
              scene_start,
              scene_end,
              object_num=5):
        self.timer = time.time()
        self.restart_requested = False
        self.process = _start_python_process(
            self.python_path,
            self.target_path,
            [
                "--output_root_path", output_root_path,
                "--env_name", env_name,
                "--section_name", section_name,
                "--platform_name", platform_name,
                "--scene_start", str(scene_start),
                "--scene_end", str(scene_end),
                "--object_num", str(object_num),
            ],
            self.workdir,
        )


        threading.Thread(target=self.stream_output).start()


    def stream_output(self):
        for line in self.process.stdout:
            output = line.strip()
            id = output.split(">")[0].strip()
            if id == "SceneGen":
                msg = output.split(">")[1].strip()
                print(f"\033[32m{msg}\033[0m")
                if msg == "START":
                    self.stage = "START"
                    self.timer = time.time()
                elif msg.split(":")[0] == "SCENE":
                    self.sock.sendall(json.dumps({"cmd": "scene_num_check",
                                                 "name": self.sock.getsockname()[0],
                                                 "data": msg.split(":")[1].strip()}).encode())
                elif msg == "END":
                    self.stage = "END"
                    self.sock.sendall(json.dumps({"cmd": "complete",
                            "name":self.sock.getsockname()[0],
                            "data":""}).encode())
            elif output:
                print(output)

    def stop(self):
        if self.process:
            _stop_process_group(self.process)
            print("Process terminated.")



    def step(self):
        # print(time.time() - self.timer)
        if self.stage == "INIT":
            if time.time() - self.timer > self.reset_time_th:
                print("\033[1;31mResetting the scene generator...\033[0m")
                self.restart_requested = True
                self.stop()


        elif self.stage == "START":
            if time.time() - self.timer > self.scene_gen_time_th:
                print("\033[1;31mScene generation timed out, restarting...\033[0m")
                self.restart_requested = True
                self.stop()

    def is_running(self):
        return self.process and self.process.poll() is None

    def current_scene_num_check(self, output_root_path, env_name, section_name, platform_name, scene_start, scene_end):
        if not os.path.exists(os.path.join(output_root_path, env_name, section_name, platform_name, "conf")):
            return scene_start
        num_list = [int(i.strip(".json")) for i in os.listdir(os.path.join(output_root_path, env_name, section_name, platform_name, "conf")) if i.endswith(".json")]
        for scene_num in range(scene_start, scene_end+1):
            if scene_num not in num_list:
                return scene_num
        return None
    






class MainProcess_PreGrasp:
    def __init__(self, 
                 python_path, 
                 target_path, 
                 gen_time_th,
                 sock,
                 pid_name,
                 workdir=None,):
        self.timer = time.time()
        self.python_path = python_path
        self.target_path = target_path
        self.gen_time_th = gen_time_th
        self.sock = sock
        self.process = None
        self.pid_name = pid_name
        self.stage = "INIT"
        self.workdir = workdir
        self.restart_requested = False


    def start(self, 
              output_root_path,
              env_name,
              section_name,
              platform_name,
              scene_start,):
        self.timer = time.time()
        self.restart_requested = False
        self.process = _start_python_process(
            self.python_path,
            self.target_path,
            [
                "--output_root_path", output_root_path,
                "--env_name", env_name,
                "--section_name", section_name,
                "--platform_name", platform_name,
                "--scene_start", str(scene_start),
            ],
            self.workdir,
        )
        
        # self.process2 = subprocess.Popen([self.python_path, self.target_path,
        #                     "--output_root_path", output_root_path,
        #                     "--env_name", env_name,
        #                     "--section_name", section_name,
        #                     "--platform_name", platform_name,
        #                     "--scene_start", str(scene_start),],
        #                     stdout=subprocess.PIPE, 
        #                     stderr=subprocess.PIPE,
        #                     text=True)  # close_fds=True is not supported on Windows


        threading.Thread(target=self.stream_output).start()


    def stream_output(self):
        for line in self.process.stdout:
            output = line.strip()
            id = output.split(">")[0].strip()
            if id == "PreGrasp":
                msg = output.split(">")[1].strip()
                print(f"\033[32m{msg}\033[0m")
                if msg == "START":
                    self.stage = "START"
                    self.timer = time.time()
                elif msg.split(":")[0] == "SCENE":
                    self.sock.sendall(json.dumps({"cmd": "scene_num_check",
                                                 "name": self.sock.getsockname()[0],
                                                 "data": msg.split(":")[1].strip()}).encode())
                elif msg == "END":
                    self.stage = "END"
            elif output:
                print(output)


    def stop(self):
        if self.process:
            _stop_process_group(self.process)
            print("Process terminated.")



    def step(self):
        # print(time.time() - self.timer)
        if self.stage == "INIT":
            if time.time() - self.timer > self.gen_time_th:
                print("\033[1;31mResetting the generator...\033[0m")
                self.restart_requested = True
                self.stop()


        elif self.stage == "START":
            if time.time() - self.timer > self.gen_time_th:
                print("\033[1;31mScene generation timed out, restarting...\033[0m")
                self.restart_requested = True
                self.stop()

    def is_running(self):
        return self.process and self.process.poll() is None

    def current_scene_num_check(self, output_root_path, env_name, section_name, platform_name, scene_start, scene_end):
        dir_path = os.path.join(output_root_path, env_name, section_name, platform_name)
        pre_grasp_dir = os.path.join(dir_path, "pre_grasp")
        num_list = (
            [
                int(os.path.splitext(name)[0])
                for name in os.listdir(pre_grasp_dir)
                if name.endswith(".json") and os.path.splitext(name)[0].isdigit()
            ]
            if os.path.isdir(pre_grasp_dir)
            else []
        )
        for scene_num in range(scene_start, scene_end+1):
            if scene_num not in num_list and os.path.exists(os.path.join(dir_path, "conf", f"{scene_num:04d}.json")):
                return scene_num
        return None














class MainProcess_Grasp:
    def __init__(self, 
                 python_path, 
                 target_path, 
                 gen_time_th,
                 reset_time_th,
                 sock,
                 pid_name,
                 output_root_path,
                env_name,
                section_name,
                platform_name,
                workdir=None,):
        self.timer = time.time()
        self.python_path = python_path
        self.target_path = target_path
        self.gen_time_th = gen_time_th
        self.reset_time_th = reset_time_th
        self.sock = sock
        self.process = None
        self.pid_name = pid_name
        self.stage = "INIT"
        self.output_root_path = output_root_path
        self.env_name = env_name
        self.section_name = section_name
        self.platform_name = platform_name
        self.scene_num = None
        self.pre_grasp_index = 0
        self.workdir = workdir
        self.restart_requested = False
        


    def start(self ):
        self.timer = time.time()
        self.restart_requested = False
        self.process = _start_python_process(
            self.python_path,
            self.target_path,
            [
                "--root_path", self.output_root_path,
                "--env_name", self.env_name,
                "--section_name", self.section_name,
                "--platform_name", self.platform_name,
                "--scene_start", str(self.scene_num),
                "--pre_grasp_index", str(self.pre_grasp_index),
            ],
            self.workdir,
        )

        threading.Thread(target=self.stream_output).start()


    def stream_output(self):
        for line in self.process.stdout:
            output = line.strip()
            id = output.split(">")[0].strip()
            if id == "Grasp":
                msg = output.split(">")[1].strip()
                print(f"\033[32m{msg}\033[0m")
                if msg == "START":
                    self.stage = "START"
                    self.timer = time.time()
                elif msg.split(":")[0] == "SCENE":
                    self.sock.sendall(json.dumps({"cmd": "scene_num_check",
                                                 "name": self.sock.getsockname()[0],
                                                 "data": msg.split(":")[1].strip()}).encode())
                elif msg == "END":
                    self.stage = "END"
            elif output:
                print(output)


    def stop(self):
        if self.process:
            _stop_process_group(self.process)
            print("Process terminated.")



    def step(self):
        # print(time.time() - self.timer)
        if self.stage == "INIT":
            if time.time() - self.timer > self.reset_time_th:
                print(f"\033[1;31mResetting the generator...{time.time() - self.timer}\033[0m")
                self.restart_requested = True
                self.stop()


        elif self.stage == "START":
            if time.time() - self.timer > self.gen_time_th:
                print(f"\033[1;31mScene generation timed out {time.time() - self.timer}, restarting...\033[0m")
                self.restart_requested = True
                self.stop()

    def is_running(self):
        return self.process and self.process.poll() is None

    def current_num_check(self, scene_start, scene_end):
        dir_path = os.path.join(self.output_root_path, self.env_name, self.section_name, self.platform_name)
        output_grasp_dir = os.path.join(dir_path, "output_grasp")
        num_list = (
            [
                int(os.path.splitext(name)[0])
                for name in os.listdir(output_grasp_dir)
                if name.endswith(".json") and os.path.splitext(name)[0].isdigit()
            ]
            if os.path.isdir(output_grasp_dir)
            else []
        )
        for scene_num in range(scene_start, scene_end+1):
            pre_grasp_path = os.path.join(
                dir_path, "pre_grasp", f"{scene_num:04d}.json"
            )
            if not os.path.exists(pre_grasp_path):
                continue
            if scene_num in num_list:
                with open(os.path.join(dir_path, "output_grasp", f"{scene_num:04d}.json"), 'r') as f:
                    output_grasp_data = json.load(f)
                with open(pre_grasp_path, 'r') as f:
                    pre_grasp_data = json.load(f)
                gripper_model_list = []
                for gd in output_grasp_data:
                    gripper_model_list.append(gd["gripper_model"])

                gripper_model_list = np.unique(gripper_model_list)

                for idx, data in enumerate(pre_grasp_data):
                    if data["gripper_model"] not in gripper_model_list:
                        self.scene_num = scene_num
                        self.pre_grasp_index = idx
                        return

            elif scene_num not in num_list:
                self.scene_num = scene_num
                self.pre_grasp_index = 0
                return

        self.scene_num = None
