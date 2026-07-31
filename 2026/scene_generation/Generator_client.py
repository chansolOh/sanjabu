
import subprocess
import socket
import threading
import time
import json
import os
import getpass
import Processes

python_path_dict = {
    # "SceneGen": f"/home/{getpass.getuser()}/ochansol/isaac_sim_5.1/python.sh",
    "SceneGen": f"/home/{getpass.getuser()}/ochansol/isaac_code/isaac_chansol/.venv/bin/python",
    "PreGrasp": f"/home/{getpass.getuser()}/ochansol/isaac_sim_5.1/python.sh",
    "Grasp":    f"/home/{getpass.getuser()}/ochansol/isaac_sim_5.1/python.sh",
}
target_python_path_dict = {
    "SceneGen": f"/home/{getpass.getuser()}/ochansol/isaac_code/python/sanjabu/2026/scene_generation/SceneGen_arg.py",
    "PreGrasp": f"/home/{getpass.getuser()}/ochansol/IsaacLab/scripts/chansol/Pre_Grasp_arg.py",
    "Grasp":    f"/home/{getpass.getuser()}/ochansol/IsaacLab/scripts/chansol/Grasp_arg.py",
}
reset_time_th = 400
scene_gen_time_th = 130

pre_grasp_gen_time_th = 300

grasp_reset_time_th = 60
grasp_gen_time_th = 400
grasp_scene_num_loop_count_th = 4


server_cmd = {} #start, stop, restart
command =""
pid_name =""


def listen_to_server(sock):
    global server_cmd, server_flag, command, pid_name
    try:
        while True:
            data = sock.recv(1024)
            if not data:
                break
            server_cmd = json.loads(data.decode())
            command = server_cmd["command"]
            pid_name = target_python_path_dict[command]
            print(f"[SERVER COMMAND] {server_cmd}")
    except:
        pass
    finally:
        sock.close()
        server_flag = False  # 연결 끊어지면 재연결하도록
        pid = get_scenegen_pid(pid_name)
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




def kill_all():
    global start_flag, server_cmd, main_process
    if 'main_process' in globals():
        main_process.stop()
        del main_process
    start_flag = False
    server_cmd["cmd"] = {}
    pid = get_scenegen_pid(pid_name)
    if pid is not None: os.kill(pid, 9)
    




server_flag = False
start_flag = False
grasp_scene_num_loop_count = 0
scene_num_old = 0
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
        if server_cmd["command"] == "SceneGen":
            if not start_flag:
                print("\033[1;32mStarting the generator...\033[0m")
                main_process = Processes.MainProcess_SceneGen(
                    python_path=python_path_dict[server_cmd["command"]], 
                    target_path=target_python_path_dict[server_cmd["command"]],
                    reset_time_th=reset_time_th,
                    scene_gen_time_th=scene_gen_time_th,
                    sock=sock,
                    pid_name=pid_name,)
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










                    
        elif server_cmd["command"] == "PreGrasp":
            if not start_flag:
                print("\033[1;32mStarting the scene generator...\033[0m")
                main_process = Processes.MainProcess_PreGrasp(
                    python_path=python_path_dict[server_cmd["command"]], 
                    target_path=target_python_path_dict[server_cmd["command"]],
                    gen_time_th = pre_grasp_gen_time_th,
                    sock=sock,
                    pid_name=pid_name,)
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
                                    scene_start       = scene_num,)
                start_flag = True
            else:
                # try:
                main_process.step()
            
                if not main_process.is_running():
                    start_flag = False
                # except:
                #     print("err")






        elif server_cmd["command"] == "Grasp":
            if not start_flag:
                print("\033[1;32mStarting the scene generator...\033[0m")
                main_process = Processes.MainProcess_Grasp(
                    python_path=python_path_dict[server_cmd["command"]], 
                    target_path=target_python_path_dict[server_cmd["command"]],
                    gen_time_th = grasp_gen_time_th,
                    reset_time_th = grasp_reset_time_th,
                    sock=sock,
                    pid_name=pid_name,
                    output_root_path = server_cmd["output_root_path"],
                    env_name        = server_cmd["env_name"],
                    section_name    = server_cmd["section_name"],
                    platform_name   = server_cmd["platform_name"],)
                
                main_process.current_num_check(
                    scene_start       = server_cmd["scene_start"],
                    scene_end         = server_cmd["scene_end"])
                scene_num = main_process.scene_num
                if scene_num_old != scene_num:
                    grasp_scene_num_loop_count = 0
                else:
                    grasp_scene_num_loop_count += 1
                    if grasp_scene_num_loop_count > grasp_scene_num_loop_count_th:
                        server_cmd["scene_start"] = scene_num + 1
                        print("\033[1;31mScene number is passed....................\033[0m")
                        grasp_scene_num_loop_count = 0
                scene_num_old = scene_num
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
                main_process.start()
                start_flag = True

            else:
                # try:
                main_process.step()
                if not main_process.is_running():
                    start_flag = False

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
