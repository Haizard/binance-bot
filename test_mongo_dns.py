from pymongo import MongoClient
import dns.resolver
import socket
import sys

# Configure DNS resolver to use Cloudflare's DNS
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['1.1.1.1']  # Cloudflare DNS

def resolve_mongodb_hosts():
    hosts = [
        "ac-yvwqxvl-shard-00-00.pzhq7bm.mongodb.net",
        "ac-yvwqxvl-shard-00-01.pzhq7bm.mongodb.net",
        "ac-yvwqxvl-shard-00-02.pzhq7bm.mongodb.net"
    ]
    
    resolved_ips = {}
    for host in hosts:
        try:
            print(f"\nResolving {host}...")
            answers = dns.resolver.resolve(host, 'A')
            ips = [rdata.address for rdata in answers]
            resolved_ips[host] = ips
            print(f"✓ Resolved to: {', '.join(ips)}")
            
            # Try to connect to the IP directly
            for ip in ips:
                try:
                    sock = socket.create_connection((ip, 27017), timeout=5)
                    print(f"✓ Successfully connected to {ip}:27017")
                    sock.close()
                except Exception as e:
                    print(f"✗ Failed to connect to {ip}:27017: {str(e)}")
                    
        except Exception as e:
            print(f"✗ Failed to resolve {host}: {str(e)}")
    
    return resolved_ips

if __name__ == "__main__":
    print("=== MongoDB DNS Resolution Test ===")
    print("Using Cloudflare DNS (1.1.1.1)")
    resolved_ips = resolve_mongodb_hosts() 