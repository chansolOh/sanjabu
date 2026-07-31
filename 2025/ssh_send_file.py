import subprocess
import os

def scp_transfer(host, username, password, local_path, remote_path):
   try:
       cmd = f"sshpass -p '{password}' scp -r {local_path} {username}@{host}:{remote_path}"
       result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
       
       if result.returncode == 0:
           print(f"[{host}] 파일 전송 완료: {local_path} -> {remote_path}")
       else:
           print(f"[{host}] 전송 실패: {result.stderr}")
           
   except Exception as e:
       print(f"[{host}] 전송 실패: {e}")


def ssh_command(host, user, password, command):
    return subprocess.run([
        'sshpass', '-p', password,
        'ssh', '-o', 'StrictHostKeyChecking=no',
        f'{user}@{host}',
        command
    ], capture_output=True, text=True)

def cmd_sender(host, username, password):
    try:
        # cmd = [ f"mkdir /home/{username}/ochansol/isaac_sim_4.5",
        #         f"wget -P /home/{username}/ochansol/isaac_sim_4.5 https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip",
        #         f"unzip /home/{username}/ochansol/isaac_sim_4.5/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip -d /home/{username}/ochansol/isaac_sim_4.5",]
        if host.split(".")[-1] in ["150", "151", "152", "153"]:
            cmd = [
                f"/home/{username}/ochansol/isaac_sim_4.5/python.sh -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121",
                f"/home/{username}/ochansol/isaac_sim_4.5/python.sh -m pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com",
                f"/home/{username}/ochansol/isaac_sim_4.5/python.sh -m pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128",
            ]
        else:
            cmd = [
                f"/home/{username}/ochansol/isaac_sim_4.5/python.sh -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121",
                f"/home/{username}/ochansol/isaac_sim_4.5/python.sh -m pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com",
            ]
        # cmd=[
        #     f"unzip /home/{username}/ochansol/isaac_sim_4.5/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip -d /home/{username}/ochansol/isaac_sim_4.5",
        # ]
        for c in cmd:
            ssh_command(host, username, password, c)
        print(f"[{host}] 명령 전송 완료")
           
    except Exception as e:
        print(f"[{host}] 전송 실패: {e}")



def transfer_to_multiple_pcs(pcs, local_file, remote_file):
   # PC 정보 설정

   
   # 각 PC에 순차적으로 전송
   for pc in pcs:
       scp_transfer(pc["host"], pc["username"], pc["password"], local_file, remote_file.replace("cubox", pc["username"]))
   
   print("모든 PC에 파일 전송 완료")

# 사용 예시
if __name__ == "__main__":
    local_file = "/home/cubox/Downloads/code_1.104.1-1758154125_amd64.deb"
    remote_file = "/home/cubox/Downloads/code_1.104.1-1758154125_amd64.deb"
    pcs = [
       {"host": "192.168.0.150", "username": "uon", "password": "uon"},
       {"host": "192.168.0.151", "username": "uon", "password": "uon"},
       {"host": "192.168.0.152", "username": "uon", "password": "uon"},
       {"host": "192.168.0.153", "username": "uon", "password": "uon"},
    #    {"host": "192.168.0.154", "username": "uon", "password": "uon"},
    #    {"host": "192.168.0.155", "username": "jinkyo2", "password": "456456"},
    #    {"host": "192.168.0.138", "username": "uon", "password": "uon"},
    #    {"host": "192.168.0.135", "username": "cubox", "password": "cubox"},
    ]
    
    # for pc in pcs:
    #    cmd_sender(pc["host"], pc["username"], pc["password"] )

   
    transfer_to_multiple_pcs(pcs, local_file, remote_file)