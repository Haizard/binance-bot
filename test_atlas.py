import socket
import requests
import dns.resolver

def test_atlas_connectivity():
    print("=== MongoDB Atlas Connectivity Test ===")
    
    # Test MongoDB Atlas main domain
    main_domains = [
        "mongodb.net",
        "mongodb.com",
        "algobot.pzhq7bm.mongodb.net"
    ]
    
    # Try both default DNS and Cloudflare DNS
    dns_servers = [
        ("Default DNS", None),
        ("Cloudflare DNS", "1.1.1.1"),
        ("Google DNS", "8.8.8.8")
    ]
    
    for dns_name, dns_server in dns_servers:
        print(f"\nTesting with {dns_name}...")
        
        if dns_server:
            resolver = dns.resolver.Resolver(configure=False)
            resolver.nameservers = [dns_server]
        else:
            resolver = dns.resolver.Resolver()
        
        for domain in main_domains:
            try:
                print(f"\nTrying to resolve {domain}...")
                answers = resolver.resolve(domain, 'A')
                print(f"✓ Successfully resolved {domain}")
                print(f"IP addresses: {', '.join(rdata.address for rdata in answers)}")
            except Exception as e:
                print(f"✗ Failed to resolve {domain}: {str(e)}")

if __name__ == "__main__":
    test_atlas_connectivity() 