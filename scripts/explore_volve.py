"""
Explore Volve Dataset via Databricks Workspace API
Browses the file structure in Volumes.
"""
import json
import os

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request

TOKEN = "sSfPlmCXmvUg_VC3Xdgi5pdPXZPnn1gq60PI0fsNnjyqkQAxxUZWN6BApof1_kHc"

# Databricks Workspace endpoint
WORKSPACE = "https://dbc-21947d43-6780.cloud.databricks.com"
DELTA_BASE = "https://northeurope-c2.azuredatabricks.net/api/2.0/delta-sharing/metastores/89504886-e5d1-4ca9-83ee-66855573787b"

SAVE_DIR = r"E:\oil-gas-data\volve_delta"
DOWNLOAD_DIR = r"E:\oil-gas-data\volve_raw"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def api_get(url, headers=None):
    """Simple GET with requests or urllib."""
    if headers is None:
        headers = {}
    headers["Authorization"] = f"Bearer {TOKEN}"

    if HAS_REQUESTS:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code >= 400:
            return {"_error": f"HTTP {resp.status_code}", "_body": resp.text[:500]}
    else:
        import ssl
        req = urllib.request.Request(url, headers=headers)
        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"_error": str(e)}


def list_volume(path=""):
    """List files in a Databricks Volume via Files API."""
    volume_path = f"/Volumes/equinor_asa_volve_data_village/public/volve{path}"
    url = f"{WORKSPACE}/api/2.0/fs/directories{volume_path}"
    return api_get(url)


def download_file(file_path, output_name):
    """Download a file from Databricks Volume."""
    volume_path = f"/Volumes/equinor_asa_volve_data_village/public/volve{file_path}"
    url = f"{WORKSPACE}/api/2.0/fs/files{volume_path}"

    headers = {"Authorization": f"Bearer {TOKEN}"}
    output_path = os.path.join(DOWNLOAD_DIR, output_name)

    print(f"  Downloading {file_path} -> {output_path}...")

    if HAS_REQUESTS:
        resp = requests.get(url, headers=headers, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    if pct % 20 == 0:
                        print(f"    {pct}% ({downloaded}/{total})")
        print(f"  Done: {output_path} ({downloaded} bytes)")
    else:
        import ssl
        req = urllib.request.Request(url, headers=headers)
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
            with open(output_path, 'wb') as f:
                f.write(resp.read())
        size = os.path.getsize(output_path)
        print(f"  Done: {output_path} ({size} bytes)")


def explore_volume():
    """Browse the Volume folder structure."""
    print("=" * 60)
    print("Browsing Volve Dataset in Databricks Volume")
    print("=" * 60)

    root = list_volume()
    if "_error" in root:
        print(f"Volume API error: {root['_error']}")
        print("\nTrying Delta Sharing API instead...")
        explore_delta()
        return

    items = root.get("items", [])
    print(f"\nRoot level: {len(items)} items\n")

    folders = []
    files = []

    for item in items:
        name = item["name"]
        itype = item["type"]
        size = item.get("size", 0)
        if itype == "directory":
            print(f"  📁 {name}/")
            folders.append(name)
        else:
            size_mb = size / (1024 * 1024)
            print(f"  📄 {name} ({size_mb:.1f} MB)")
            files.append(name)

    # Explore each subfolder one level deeper
    for folder in folders:
        print(f"\n  📁 {folder}/ contents:")
        sub = list_volume(f"/{folder}")
        if "_error" in sub:
            print(f"    Error: {sub['_error']}")
            continue
        sub_items = sub.get("items", [])
        sub_folders = []
        for item in sub_items:
            name = item["name"]
            itype = item["type"]
            size = item.get("size", 0)
            if itype == "directory":
                print(f"      📁 {name}/")
                sub_folders.append(name)
            else:
                size_mb = size / (1024 * 1024)
                print(f"      📄 {name} ({size_mb:.1f} MB)")
                # Download small files directly
                if size < 10 * 1024 * 1024:  # < 10MB
                    full_path = f"/{folder}/{name}"
                    download_file(full_path, f"{folder}/{name}")

        # Explore second level subfolders
        for sf in sub_folders[:5]:  # Limit to first 5
            s_path = f"/{folder}/{sf}"
            print(f"\n      📁 {sf}/ contents:")
            sub2 = list_volume(s_path)
            if "_error" in sub2:
                print(f"        Error: {sub2['_error']}")
                continue
            for item in sub2.get("items", []):
                name = item["name"]
                itype = item["type"]
                size = item.get("size", 0)
                if itype == "directory":
                    print(f"          📁 {name}/")
                else:
                    size_mb = size / (1024 * 1024)
                    label = f"{size_mb:.1f} MB" if size_mb < 1000 else f"{size_mb/1024:.1f} GB"
                    print(f"          📄 {name} ({label})")

    # Summary: show where well headers are
    print("\n" + "=" * 60)
    print("To find well coordinates (井口坐标), look for:")
    print("  - CSV files with 'well' or 'header' in name")
    print("  - Folders named 'Well' or 'Wells'")
    print("=" * 60)


def explore_delta():
    """Fallback: explore via Delta Sharing."""
    share = "equinoropendata_volve"
    tables_result = api_get(
        f"{DELTA_BASE}/shares/{share}/all-tables"
    )
    print("Delta Sharing all-tables:")
    print(json.dumps(tables_result, indent=2)[:3000])


if __name__ == "__main__":
    explore_volume()
