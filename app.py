import os
import time
import threading
import json
import urllib.request
import urllib.error
from typing import cast as _cast
from flask import Flask, request, jsonify, send_from_directory
import PyPDF2
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder='.', static_url_path='')

# Load OpenRouter API Key and default models from environment variables (.env)
DEFAULT_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_MODELS_STR = os.environ.get("OPENROUTER_MODELS", "")
DEFAULT_MODELS = [m.strip() for m in DEFAULT_MODELS_STR.split(",")] if DEFAULT_MODELS_STR else []

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
RENDER_DEPLOY_URL = os.environ.get("RENDER_DEPLOY_URL", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized.")
    except ImportError:
        print("Supabase package not installed. Run `pip install supabase`")


def get_system_prompt_for_language(language):
    return """You are ManiLoan, a concise and smart Loan Prediction Assistant referencing the 2026 Indian Bank Underwriting & Credit Criteria (SBI, HDFC, ICICI, Axis, BoB, PNB, Cosmos, Saraswat).

### STRICT CONCISENESS & PARAMETERS DIRECTIVE (MANDATORY):
- Keep all responses SHORT, CRISP, and TO THE POINT.
- Give ONLY what is needed. Avoid unnecessary fluff or long explanations.
- All numbers and currency must be in Indian Rupees (₹) (e.g. ₹50,000, ₹15 Lakhs, ₹12,500/month). Never use dollar ($).
- **TENURE REQUIREMENT**: In the Tenure line, ALWAYS state the **Minimum to Maximum tenure range** for that product (e.g. `Min 1 to Max 30 Years (12 to 360 Months) [Requested: 20 Years]`).

### 2026 INDIAN BANK UNDERWRITING BENCHMARKS & TENURE RANGES (REFERENCE):
- **Home Loan**: Tenure Min 1–5 Years to Max 30 Years (12–360 Months), Age 18–70, Min Salary ₹25k–₹30k/mo (ICICI/HDFC), FOIR/EMI burden <= 45–50%, CIBIL 750+ prime cutoff.
- **Vehicle Loan**: Tenure Min 1 Year to Max 7–8 Years (12–96 Months), Age 21–60/65, Min Annual Income ₹2.4L–₹3.0L (Axis/HDFC), Cosmos up to 90% on-road.
- **Gold Loan**: Tenure Min 6 Months to Max 36 Months (3 Years), RBI mandatory max 75% LTV, 18K–22K gold, SBI ₹20k–₹50L, Cosmos 70% LTV.
- **Education Loan**: Tenure Min 1 Year to Max 15 Years (12–180 Months) + Moratorium course+6/12 mo, Co-applicant mandatory.
- **Consumer Durable Loan**: Tenure Min 3 Months to Max 24–36 Months, low/0% down payment.
- **Cooperative Banks (Cosmos, Saraswat, UCBs)**: UCB board-approved credit policies, Cosmos Home Loan up to ₹3 Cr, Saraswat microfinance up to ₹3L household income.
- **Other Products**: Personal Loan (Tenure Min 1 Year to Max 5 Years / 12–60 Months, unsecured, FOIR <= 40–50%), LAP (Tenure Min 1 Year to Max 15 Years / 12–180 Months, 50–65% LTV), MSME/Business (Tenure Min 1 to Max 10 Years).

### FORMAT FOR LOAN APPLICATION:
Once the user provides their details, you MUST evaluate them and respond EXACTLY in this step-by-step format:
1. **Eligible Banks**: [List of 2-3 eligible banks based on the criteria, e.g., SBI, HDFC]
2. **Interest Rate**: [Estimated interest rates for the eligible banks, e.g., SBI (8.5%), HDFC (8.75%)]
3. **EMI**: [Calculated Estimated EMI in ₹]
4. **Tenure**: [Eligible Tenure range, e.g., 15 to 20 Years]
5. **Required documents to be submitted to the bank**: (ONLY include this section if the user is eligible for at least one bank. Do NOT include if Eligible Banks is None)
- [Document 1]
- [Document 2]
- [Document 3]

### GENERAL QUESTIONS & BANK INQUIRIES:
- Answer directly in 1–2 short sentences or concise bullet points with amounts in ₹ using 2026 Indian banking benchmarks. No filler.

### GREETING:
"Hi! I'm your ManiLoan assistant referencing 2026 Indian Bank Underwriting Criteria (SBI, HDFC, ICICI, Axis, Cooperative Banks). Please select the type of Loan:
1. Home Loan
2. Vehicle Loan
3. Gold Loan
4. Education Loan
5. Consumer Durable Loan
6. Personal Loan
7. LAP (Loan Against Property)
8. MSME / Business Loan"

### AFTER LOAN SELECTION:
When the user selects a loan type, you MUST respond EXACTLY with the matching template below (pick the one for the selected loan).
Append the hidden marker [[LOAN_TYPE:X]] at the end (X = Home | Car | Gold | Education | Durable | Personal | LAP | MSME).

For Home Loan:
"Please fill in the borrower's details below for quick evaluation:[[LOAN_TYPE:Home]]"

For Vehicle Loan:
"Please fill in the borrower's details below for quick evaluation:[[LOAN_TYPE:Car]]"

For Gold Loan:
"Please fill in the borrower's details below for quick evaluation:[[LOAN_TYPE:Gold]]"

For Education Loan:
"Please fill in the borrower's details below for quick evaluation:[[LOAN_TYPE:Education]]"

For Consumer Durable Loan:
"Please fill in the borrower's details below for quick evaluation:[[LOAN_TYPE:Durable]]"

For Personal Loan:
"Please fill in the borrower's details below for quick evaluation:[[LOAN_TYPE:Personal]]"

For LAP (Loan Against Property):
"Please fill in the borrower's details below for quick evaluation:[[LOAN_TYPE:LAP]]"

For MSME / Business Loan:
"Please fill in the borrower's details below for quick evaluation:[[LOAN_TYPE:MSME]]"
"""

# Default system prompt for backwards compatibility
SYSTEM_PROMPT = get_system_prompt_for_language("English")

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "backend": "Python / Flask",
        "openrouter_configured": bool(DEFAULT_OPENROUTER_API_KEY),
        "default_api_key": DEFAULT_OPENROUTER_API_KEY,
        "default_models": DEFAULT_MODELS
    })

@app.route('/api/test-openrouter', methods=['POST'])
def test_openrouter_connection():
    data = request.get_json() or {}
    api_key = data.get('apiKey', '').strip() or DEFAULT_OPENROUTER_API_KEY
    model = data.get('model', '').strip()

    if not api_key:
        return jsonify({"success": False, "message": "No OpenRouter API Key provided."}), 400

    models_to_test = [m.strip() for m in model.split(",")] if model else DEFAULT_MODELS
    if not models_to_test:
        return jsonify({"success": False, "message": "No OpenRouter models provided."}), 400

    last_error_msg = ""
    for test_model in models_to_test:
        start_time = time.time()
        payload = {
            "model": test_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10
        }

        req = urllib.request.Request(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'ManiLoan-App'
            },
            data=json.dumps(payload).encode('utf-8')
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as _:
                latency = int((time.time() - start_time) * 1000)
                return jsonify({
                    "success": True,
                    "latencyMs": latency,
                    "model_used": test_model,
                    "message": f"OpenRouter API connection verified via backend ({test_model}, {latency}ms)"
                })
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='ignore')
            try:
                err_json = json.loads(err_body)
                last_error_msg = err_json.get('error', {}).get('message', f"HTTP {e.code}")
            except Exception:
                last_error_msg = f"HTTP {e.code} {e.reason}"
            continue # Try next model
        except Exception as e:
            last_error_msg = str(e)
            continue # Try next model

    return jsonify({"success": False, "message": f"OpenRouter API Error: All fallback models failed. Last error: {last_error_msg}"}), 400

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json() or {}
    user_query = data.get('query', '').strip()
    api_key = data.get('apiKey', '').strip() or DEFAULT_OPENROUTER_API_KEY
    model = data.get('model', '').strip()
    chat_history = data.get('chatHistory', [])
    language = data.get('language', 'English').strip()

    if not user_query:
        return jsonify({"error": "Query parameter cannot be empty."}), 400

    if not api_key:
        return jsonify({"error": "No OpenRouter API key configured on backend or provided in request."}), 400

    # Build prompt messages payload with dedicated single-language system prompt
    system_content = get_system_prompt_for_language(language)

    messages = [{"role": "system", "content": system_content}]
    
    # Add conversation history
    for item in chat_history[-8:]:
        if isinstance(item, dict) and 'role' in item and 'content' in item:
            messages.append({"role": item['role'], "content": item['content']})

    messages.append({"role": "user", "content": user_query})

    models_to_test = [m.strip() for m in model.split(",")] if model else DEFAULT_MODELS
    if not models_to_test:
        return jsonify({"error": "No model provided or configured."}), 400

    last_error_msg = ""
    for test_model in models_to_test:
        payload = {
            "model": test_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 500
        }

        req = urllib.request.Request(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            },
            data=json.dumps(payload).encode('utf-8')
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                reply_text = res_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                usage = res_data.get('usage', {})

                if supabase:
                    session_id = data.get('sessionId')
                    if session_id:
                        try:
                            updated_history = chat_history + [
                                {"role": "user", "content": user_query},
                                {"role": "assistant", "content": reply_text}
                            ]
                            # Trim to last 50 messages to keep DB rows small
                            updated_history = updated_history[-50:]
                            messages_json = json.dumps(updated_history)

                            # Select-first approach: works without UNIQUE constraint on session_id
                            existing = supabase.table("chat_history") \
                                .select("id") \
                                .eq("session_id", session_id) \
                                .limit(1) \
                                .execute()

                            existing_rows = _cast(list[dict], existing.data or [])
                            if existing_rows:
                                # Row exists — update it
                                row_id = existing_rows[0]["id"]
                                supabase.table("chat_history") \
                                    .update({"messages": messages_json}) \
                                    .eq("id", row_id) \
                                    .execute()
                            else:
                                # No row yet — insert a fresh one
                                supabase.table("chat_history").insert({
                                    "session_id": session_id,
                                    "messages": messages_json
                                }).execute()
                        except Exception as e:
                            print("Supabase chat logging error:", repr(e))

                return jsonify({
                    "text": reply_text,
                    "source": "openrouter",
                    "backend": "Python / Flask",
                    "model_used": test_model,
                    "usage": usage
                })
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='ignore')
            try:
                err_json = json.loads(err_body)
                last_error_msg = err_json.get('error', {}).get('message', f"HTTP {e.code}")
            except Exception:
                last_error_msg = f"HTTP {e.code} {e.reason}"
            print(f"Model {test_model} failed with: {last_error_msg}")
            continue # Try next model
        except Exception as e:
            last_error_msg = str(e)
            print(f"Model {test_model} failed with: {last_error_msg}")
            continue # Try next model

    return jsonify({"error": f"OpenRouter API Error: All models failed. Last error: {last_error_msg}"}), 400

@app.route('/api/parse-document', methods=['POST'])
def parse_document():
    if 'document' not in request.files:
        return jsonify({"error": "No document uploaded"}), 400
    
    file = request.files['document']
    api_key = request.form.get('apiKey', '').strip() or DEFAULT_OPENROUTER_API_KEY
    model = request.form.get('model', '').strip()
    
    if not api_key:
        return jsonify({"error": "No OpenRouter API key configured"}), 400

    filename = secure_filename(file.filename or "")
    text_content = ""
    # Customer documents are intentionally NOT stored — read in-memory only

    try:
        if filename.lower().endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file.stream)
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
        else:
            text_content = file.read().decode('utf-8', errors='ignore')
            
        # truncate if too long
        text_content = text_content[:8000]
        
        prompt = f"""
Extract the following information from this document. Return ONLY a valid JSON object. Do not include markdown code blocks, do not explain. Use these exact keys:
"ApplicantAge": (number or null),
"ApplicantIncome": (number or null, monthly income in INR),
"CoapplicantIncome": (number or null, monthly income in INR),
"LoanAmountRequested": (number or null, loan amount in lakhs, convert to plain number like 2 for 2 Lakhs)

Document text:
{text_content}
        """
        
        models_to_test = [m.strip() for m in model.split(",")] if model else DEFAULT_MODELS
        if not models_to_test:
            return jsonify({"error": "No OpenRouter model configured"}), 400

        last_error_msg = ""
        for test_model in models_to_test:
            payload = {
                "model": test_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
                "response_format": {"type": "json_object"}
            }

            req = urllib.request.Request(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                data=json.dumps(payload).encode('utf-8')
            )
            
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                    if 'choices' in res_data and len(res_data['choices']) > 0:
                        content = res_data['choices'][0]['message']['content']
                        parsed_json = json.loads(content)
                        parsed_json['model_used'] = test_model
                        return jsonify({"success": True, "data": parsed_json})
                    else:
                        last_error_msg = "Invalid format from API"
                        continue
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8', errors='ignore')
                try:
                    err_json = json.loads(err_body)
                    last_error_msg = err_json.get('error', {}).get('message', f"HTTP {e.code}")
                except Exception:
                    last_error_msg = f"HTTP {e.code}"
                continue
            except Exception as e:
                last_error_msg = str(e)
                continue

        return jsonify({"error": f"Failed to extract info from all models. Last error: {last_error_msg}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# SUPABASE ENDPOINTS
# -----------------------------

@app.route('/api/register', methods=['POST'])
def register():
    if not supabase:
        return jsonify({"error": "Supabase not configured"}), 500
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        return jsonify({"success": True, "user": response.user.id if response.user else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
    if not supabase:
        return jsonify({"error": "Supabase not configured"}), 500
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return jsonify({"success": True, "session": response.session.access_token if response.session else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/submit-loan', methods=['POST'])
def submit_loan():
    if not supabase:
        return jsonify({"error": "Supabase not configured"}), 500
    
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "Missing Authorization header"}), 401
    
    token = auth_header.replace("Bearer ", "")
    data = request.get_json()
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            return jsonify({"error": "Invalid token"}), 401
            
        user_id = user_response.user.id
        insert_data = {
            "user_id": user_id,
            "applicant_age": data.get("age"),
            "income": data.get("income"),
            "loan_amount_requested": data.get("loan_amount"),
            "tenure_months": data.get("tenure")
        }
        
        res = supabase.table("loan_applications").insert(insert_data).execute()
        return jsonify({"success": True, "data": res.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------
# AUTO-PING TO PREVENT RENDER COLD START
# -----------------------------
def ping_self():
    while True:
        try:
            if RENDER_DEPLOY_URL:
                ping_url = f"{RENDER_DEPLOY_URL.rstrip('/')}/api/health"
                req = urllib.request.Request(ping_url)
                urllib.request.urlopen(req, timeout=10).close()
                print(f"Auto-ping sent to {ping_url}")
        except Exception as e:
            print(f"Auto-ping failed: {e}")
        time.sleep(600)  # 10 minutes (600 seconds)

ping_thread = threading.Thread(target=ping_self, daemon=True)
ping_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Python Risk Analytics Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
