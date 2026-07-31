
import subprocess
import socket
import threading
import time
import json
import os
import getpass

python_path = f"/home/{getpass.getuser()}/ochansol/isaac_sim_4.2/python.sh"
target_path = f"/home/{getpass.getuser()}/ochansol/isaac_code/python/sanjabu/2025/SceneGen_arg.py"
utils_path =  f"/home/{getpass.getuser()}/ochansol/isaac_code/python/utils"

reset_time_th = 400
scene_gen_time_th = 100


server_cmd = {} #start, stop, restart



def listen_to_server(sock):
    global server_cmd, server_flag
    try:
        while True:
            data = sock.recv(1024)
            if not data:
                break
            server_cmd = json.loads(data.decode())
            print(f"[SERVER COMMAND] {server_cmd}")
    except:
        pass
    finally:
        sock.close()
        server_flag = False  # 연결 끊어지면 재연결하도록
        pid = get_scenegen_pid("SceneGen_arg.py")
        if pid is not None: os.kill(pid, 9)



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


class MainProcess:
    def __init__(self, 
                 python_path, 
                 target_path, 
                 utils_path,
                 reset_time_th,
                 scene_gen_time_th,
                 sock):
        self.stage = "INIT"
        self.timer = time.time()
        self.python_path = python_path
        self.target_path = target_path
        self.utils_path = utils_path
        self.reset_time_th = reset_time_th
        self.scene_gen_time_th = scene_gen_time_th
        self.sock = sock
        self.process = None


    def start(self, 
              output_root_path,
              env_name,
              section_name,
              platform_name,
              scene_start,
              scene_end,
              object_num=5):
        self.timer = time.time()
        self.process = subprocess.Popen([self.python_path, self.target_path,
                            "--output_root_path", output_root_path,
                            "--env_name", env_name,
                            "--section_name", section_name,
                            "--platform_name", platform_name,
                            "--scene_start", str(scene_start),
                            "--scene_end",  str(scene_end),
                            "--object_num", str(object_num),
                            "--utils_path", self.utils_path],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            text=True)  # close_fds=True is not supported on Windows


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

    def stop(self):
        if self.process:
            self.process.kill()
            print("Process terminated.")
            time.sleep(3)
            pid = get_scenegen_pid("SceneGen_arg.py")
            if pid is not None: os.kill(pid, 9)
            time.sleep(3)



    def step(self):
        # print(time.time() - self.timer)
        if self.stage == "INIT":
            if time.time() - self.timer > self.reset_time_th:
                print("\033[1;31mResetting the scene generator...\033[0m")
                self.stop()


        elif self.stage == "START":
            if time.time() - self.timer > self.scene_gen_time_th:
                print("\033[1;31mScene generation timed out, restarting...\033[0m")
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

def kill_all():
    global start_flag, server_cmd, main_process
    if 'main_process' in globals():
        main_process.stop()
        del main_process
    start_flag = False
    server_cmd["cmd"] = {}
    pid = get_scenegen_pid("SceneGen_arg.py")
    if pid is not None: os.kill(pid, 9)
    




server_flag = False
start_flag = False
while True:
    time.sleep(0.1)
    if not server_flag:
        try:
            if 'main_process' in globals(): main_process.stop(); server_cmd["cmd"] = {} ; start_flag = False; main_process = None
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("192.168.0.137", 1823))
            sock.sendall(sock.getsockname()[0].encode())
            threading.Thread(target=listen_to_server, args=(sock,), daemon=True).start()
            server_flag = True
        except:
            print("sock err")
            server_flag = False


    if server_cmd.get("cmd") is None:
        continue
    if server_cmd["cmd"] == "start":
        if not start_flag:
            print("\033[1;32mStarting the scene generator...\033[0m")
            # kill_all()
            # if 'main_process' in globals():
            #     main_process.stop()
            #     start_flag = False
            #     server_cmd["cmd"] = {}

            # pid = get_scenegen_pid("SceneGen_arg.py")
            # if pid is not None: os.kill(pid, 9)

            main_process = MainProcess(python_path=python_path, 
                                       target_path=target_path, 
                                       utils_path=utils_path,
                                       reset_time_th=reset_time_th,
                                       scene_gen_time_th=scene_gen_time_th,
                                       sock=sock)
            scene_num = main_process.current_scene_num_check(
                                output_root_path = server_cmd["output_root_path"],
                                env_name        = server_cmd["env_name"],
                                section_name    = server_cmd["section_name"],
                                platform_name   = server_cmd["platform_name"],
                                scene_start       = server_cmd["scene_start"],
                                scene_end         = server_cmd["scene_end"])
            sock.sendall(json.dumps({"cmd": "scene_num_check",
                                    "name":sock.getsockname()[0],
                                    "data":scene_num}).encode())
            if scene_num is None:
                print("\033[1;31mAll scenes are already generated.\033[0m")
                sock.sendall(json.dumps({"cmd": "complete",
                                         "name":sock.getsockname()[0],
                                         "data":""}).encode())
                kill_all()
                continue
            main_process.start( 
                                output_root_path = server_cmd["output_root_path"],
                                env_name        = server_cmd["env_name"],
                                section_name    = server_cmd["section_name"],
                                platform_name   = server_cmd["platform_name"],
                                scene_start       = scene_num,
                                scene_end         = server_cmd["scene_end"],
                                object_num      = server_cmd["object_num"])
            start_flag = True
        else:
            try:
                main_process.step()
                if not main_process.is_running():
                    start_flag = False
            except:
                print("err")

    elif server_cmd["cmd"] == "stop":
        kill_all()
        sock.sendall(json.dumps({"cmd": "stop",
                            "name":sock.getsockname()[0],
                            "data":""}).encode())
        # if 'main_process' in globals():
        #     main_process.stop()
        #     start_flag = False
        #     server_cmd["cmd"] = {}

    # print(time.time() - timer)



