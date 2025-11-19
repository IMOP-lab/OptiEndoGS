import torch
import traceback
import socket
import json
import struct
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def send_json_data(conn, data):

    serialized_data = json.dumps(data)

    bytes_data = serialized_data.encode('utf-8')

    conn.sendall(struct.pack('I', len(bytes_data)))

    conn.sendall(bytes_data)

def try_connect(render_items):
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
        send_json_data(conn, render_items)
    except Exception as inst:
        pass

            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify, metrics):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))
    send_json_data(conn, metrics)

def receive():
    message = read()
    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            render_mode = message["render_mode"]
        except Exception as e:
            print("")
            traceback.print_exc()

        return custom_cam, do_training, keep_alive, scaling_modifier, render_mode
    else:
        return None, None, None, None, None