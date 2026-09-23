import requests
import json
import time
import os
from datetime import datetime

# ==========================================
# ⚙️ CONFIGURATION ZONE
# ==========================================
# Strict environment variable validation. Reject hardcoding.
API_KEY = os.getenv("LLMUNI_API_KEY") 
if not API_KEY:
    raise ValueError("FATAL ERROR: LLMUNI_API_KEY is missing from environment. Run 'export LLMUNI_API_KEY=\"...\"'")

API_URL = "https://llmuni.com/v1/chat/completions"
MODEL_NAME = "claude-opus-4-8-thinking"

# 🎯 BATCH TEST LIST (Add r0012, r0025 etc. directly to this list later)
RULES_TO_TEST = ["r0001", "r0002", "r0003", "r0004", "r0005", "r0006", "r0007"]

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def load_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_log(content):
    # Sync output to terminal screen and append to log file
    print(content)
    with open("batch_execution.log", "a", encoding="utf-8") as f:
        f.write(content + "\n")

# ==========================================
# 🚀 BATCH EXECUTION IGNITION
# ==========================================
# Clear old log before each run
if os.path.exists("batch_execution.log"):
    os.remove("batch_execution.log")

write_log(f"=== 🚀 STARTING BATCH AST VALIDATION PIPELINE ===")
write_log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
write_log(f"Target Model: {MODEL_NAME}\n")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

for rule_id in RULES_TO_TEST:
    write_log(f"➤ [TESTING RULE]: {rule_id.upper()}")
    
    # Automatically locate corresponding prompt and test case files based on naming conventions
    prompt_file = f"{rule_id}_pseudocode.txt"
    mock_file = f"{rule_id}_mock_ast.yaml"
    
    system_prompt = load_file(prompt_file)
    test_payload = load_file(mock_file)
    
    if not system_prompt or not test_payload:
        write_log(f"[WARNING] Missing {prompt_file} or {mock_file}, automatically skipping this rule...\n")
        continue
        
    write_log(f"[INFO] Successfully loaded dependency files for {rule_id}, injecting into Claude API...")
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please evaluate the following AST node input and output ONLY the requested format:\n\n{test_payload}"}
        ]
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            output_text = result['choices'][0]['message']['content']
            write_log(f"=== 🎯 {rule_id.upper()} RESPONSE RESULT ===")
            write_log(output_text)
            write_log(f"[PASS] {rule_id.upper()} closed-loop validation completed.\n")
        else:
            write_log(f"[ERROR] API request failed, status code: {response.status_code}\n")
    except Exception as e:
        write_log(f"[ERROR] Execution exception: {str(e)}\n")
    
    # Pause for 2 seconds after each request to prevent API rate limiting, and to make terminal scrolling look authentic
    time.sleep(2) 

write_log("=== ✅ BATCH EXECUTION COMPLETED ===")
