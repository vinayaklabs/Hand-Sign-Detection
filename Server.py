import http.server
import socketserver
import os
import webbrowser

PORT = 8000

# Get the folder where server.py is located
folder = os.path.dirname(os.path.abspath(__file__))

print("--------------------------------")
print("SignSpeak Server")
print("--------------------------------")
print("Server folder:")
print(folder)
print()
print("Files in this folder:")

# Show every file in the folder
for file in os.listdir(folder):
    print(" -", file)

print("--------------------------------")

# Move into the server.py folder
os.chdir(folder)

# Start server
server = socketserver.TCPServer(
    ("localhost", PORT),
    http.server.SimpleHTTPRequestHandler
)

url = f"http://localhost:{PORT}/index.html"

print()
print("Server started!")
print("Open:")
print(url)
print()

webbrowser.open(url)

server.serve_forever()