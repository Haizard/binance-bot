import socket
import requests
import sys

def test_connectivity():
    print("Testing basic internet connectivity...")
    
    # Test general internet connectivity
    try:
        response = requests.get("https://www.google.com")
        print("✓ Internet connection is working")
    except Exception as e:
        print("✗ Cannot connect to internet:", str(e))
        return

    # Test MongoDB Atlas connectivity
    hosts = [
        "mongodb.com",
        "mongodb.net",
        "cloud.mongodb.com"
    ]
    
    print("\nTesting MongoDB Atlas domains...")
    for host in hosts:
        try:
            ip = socket.gethostbyname(host)
            print(f"✓ {host} resolves to {ip}")
        except socket.gaierror as e:
            print(f"✗ Cannot resolve {host}: {str(e)}")

    # Test MongoDB Atlas ports
    print("\nTesting MongoDB Atlas ports...")
    ports = [27017, 27018, 27019]
    test_host = "cloud.mongodb.com"
    
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((test_host, port))
            if result == 0:
                print(f"✓ Port {port} is open")
            else:
                print(f"✗ Port {port} is closed")
        except Exception as e:
            print(f"✗ Error testing port {port}: {str(e)}")
        finally:
            sock.close()

if __name__ == "__main__":
    test_connectivity() 