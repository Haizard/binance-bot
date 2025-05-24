import socket
import ssl
import time

def test_mongodb_connection(host, port=27017):
    try:
        # Create SSL context
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        print(f"\nTesting connection to {host}:{port}")
        print("1. Creating socket...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        print("2. Wrapping socket with SSL...")
        wrapped_sock = context.wrap_socket(sock)
        
        print("3. Attempting to connect...")
        start_time = time.time()
        wrapped_sock.connect((host, port))
        end_time = time.time()
        
        print(f"✓ Successfully connected! Response time: {(end_time - start_time):.2f} seconds")
        return True
        
    except socket.timeout:
        print(f"✗ Connection timed out after {(time.time() - start_time):.2f} seconds")
        return False
    except socket.error as e:
        print(f"✗ Connection failed: {str(e)}")
        return False
    finally:
        try:
            wrapped_sock.close()
        except:
            pass

def main():
    print("=== MongoDB Connectivity Test ===")
    
    hosts = [
        "ac-biem6go-shard-00-00.pzhq7bm.mongodb.net",
        "ac-biem6go-shard-00-01.pzhq7bm.mongodb.net",
        "ac-biem6go-shard-00-02.pzhq7bm.mongodb.net"
    ]
    
    success = 0
    for host in hosts:
        if test_mongodb_connection(host):
            success += 1
    
    print(f"\nSummary: Successfully connected to {success} out of {len(hosts)} servers")

if __name__ == "__main__":
    main() 