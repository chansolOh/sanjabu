
import json
import socket
import threading


clients = {}  # id: conn 저장

def handle_client(conn, addr):
    try:
        client_id = conn.recv(1024).decode().strip()
        clients[client_id] = conn
        print(f"[CONNECTED] {client_id} from {addr}")

        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"[{client_id}] {data.decode()}")

    except Exception as e:
        print(f"[ERROR] {addr} - {e}")
    finally:
        print(f"[DISCONNECTED] {addr}")
        conn.close()
        clients.pop(client_id, None)

def command_loop(clients):
    # while True:
    print(clients)
    msg_dict = {
        "cmd": "start",
        "env_name":"Manufactory",
        "section_name":"FOODnamoo_poultry_plant",
        "platform_name":"catch_table",
        "scene_num":0,
        "gen_num":10,
        "object_num":5
    }
    send_msg = json.dumps(msg_dict).encode()
    import pdb; pdb.set_trace()
    clients["main_pc"].sendall(send_msg)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("192.168.0.137", 1823))
server.listen()

print("[SERVER STARTED] Waiting for clients...")

# 명령어 입력용 쓰레드 시작
threading.Thread(target=command_loop, args=(clients,),daemon=True).start()

# 클라이언트 대기 루프
while True:
    conn, addr = server.accept()
    threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
