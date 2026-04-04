"""CWE definitions, language applicability, and prompt generation."""

from dataclasses import dataclass


@dataclass
class CWEEntry:
    id: str
    name: str
    description: str
    languages: set[str]  # empty = all languages
    example_vulnerable: dict[str, str]  # language -> code example
    example_safe: dict[str, str]  # language -> code example


# CWE Top 25 (2024) with language applicability
# Languages left empty means applicable to all
CWE_TOP_25: list[CWEEntry] = [
    CWEEntry(
        id="CWE-787",
        name="Out-of-bounds Write",
        description="The product writes data past the end, or before the beginning, of the intended buffer.",
        languages={"c", "cpp"},
        example_vulnerable={"c": 'char buf[10];\nstrcpy(buf, user_input);  // no bounds check'},
        example_safe={"c": 'char buf[10];\nstrncpy(buf, user_input, sizeof(buf) - 1);\nbuf[sizeof(buf) - 1] = \'\\0\';'},
    ),
    CWEEntry(
        id="CWE-79",
        name="Cross-site Scripting (XSS)",
        description="The product does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output used as a web page served to other users.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={"python": 'return f"<p>Hello {user_name}</p>"  # unsanitized'},
        example_safe={"python": 'from markupsafe import escape\nreturn f"<p>Hello {escape(user_name)}</p>"'},
    ),
    CWEEntry(
        id="CWE-89",
        name="SQL Injection",
        description="The product constructs all or part of an SQL command using externally-influenced input without neutralizing special elements that could modify the intended SQL command.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={
            "python": 'cursor.execute(f"SELECT * FROM users WHERE name = \'{user_input}\'")' ,
            "perl": 'my $sth = $dbh->prepare("SELECT * FROM users WHERE name = \'$input\'");',
        },
        example_safe={
            "python": 'cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))',
            "perl": 'my $sth = $dbh->prepare("SELECT * FROM users WHERE name = ?");\n$sth->execute($input);',
        },
    ),
    CWEEntry(
        id="CWE-416",
        name="Use After Free",
        description="The product dereferences a pointer that has been freed, leading to undefined behavior.",
        languages={"c", "cpp"},
        example_vulnerable={"c": "free(ptr);\nprintf(\"%s\", ptr);  // use after free"},
        example_safe={"c": "free(ptr);\nptr = NULL;"},
    ),
    CWEEntry(
        id="CWE-78",
        name="OS Command Injection",
        description="The product constructs all or part of an OS command using externally-influenced input without neutralizing special elements that could modify the intended command.",
        languages=set(),  # all languages
        example_vulnerable={"python": 'os.system(f"ping {user_host}")'},
        example_safe={"python": 'subprocess.run(["ping", user_host], check=True)'},
    ),
    CWEEntry(
        id="CWE-20",
        name="Improper Input Validation",
        description="The product receives input but does not validate or incorrectly validates that the input has the properties required to process the data safely and correctly.",
        languages=set(),
        example_vulnerable={"python": 'def set_port(port_str):\n    port = int(port_str)  # no range check\n    listen(port)'},
        example_safe={"python": 'def set_port(port_str):\n    port = int(port_str)\n    if not (1 <= port <= 65535):\n        raise ValueError("invalid port")\n    listen(port)'},
    ),
    CWEEntry(
        id="CWE-125",
        name="Out-of-bounds Read",
        description="The product reads data past the end, or before the beginning, of the intended buffer.",
        languages={"c", "cpp"},
        example_vulnerable={"c": "int arr[10];\nreturn arr[index];  // index not validated"},
        example_safe={"c": "int arr[10];\nif (index >= 0 && index < 10) return arr[index];"},
    ),
    CWEEntry(
        id="CWE-22",
        name="Path Traversal",
        description="The product uses external input to construct a pathname intended to identify a file or directory below a restricted parent, but does not properly neutralize special elements like '..' that can resolve to a location outside of that directory.",
        languages=set(),
        example_vulnerable={"python": 'path = os.path.join(BASE_DIR, user_filename)\nreturn open(path).read()'},
        example_safe={"python": 'path = os.path.join(BASE_DIR, user_filename)\nif not os.path.realpath(path).startswith(os.path.realpath(BASE_DIR)):\n    raise ValueError("path traversal")\nreturn open(path).read()'},
    ),
    CWEEntry(
        id="CWE-352",
        name="Cross-Site Request Forgery (CSRF)",
        description="The web application does not sufficiently verify that a well-formed, valid, consistent request was intentionally provided by the user who submitted the request.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={"python": '@app.route("/transfer", methods=["POST"])\ndef transfer():\n    # no CSRF token check'},
        example_safe={"python": '@app.route("/transfer", methods=["POST"])\n@csrf_protect\ndef transfer():\n    ...'},
    ),
    CWEEntry(
        id="CWE-434",
        name="Unrestricted Upload of File with Dangerous Type",
        description="The product allows the upload of files without restricting the file type.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={"python": 'uploaded = request.files["file"]\nuploaded.save(os.path.join(UPLOAD_DIR, uploaded.filename))'},
        example_safe={"python": 'uploaded = request.files["file"]\nif not allowed_extension(uploaded.filename):\n    abort(400)\nfilename = secure_filename(uploaded.filename)\nuploaded.save(os.path.join(UPLOAD_DIR, filename))'},
    ),
    CWEEntry(
        id="CWE-862",
        name="Missing Authorization",
        description="The product does not perform an authorization check when an actor attempts to access a resource or perform an action.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={"python": '@app.route("/admin/users")\ndef list_users():\n    return jsonify(User.query.all())  # no auth check'},
        example_safe={"python": '@app.route("/admin/users")\n@login_required\n@admin_required\ndef list_users():\n    return jsonify(User.query.all())'},
    ),
    CWEEntry(
        id="CWE-476",
        name="NULL Pointer Dereference",
        description="The product dereferences a pointer that it expects to be valid but is NULL.",
        languages={"c", "cpp"},
        example_vulnerable={"c": "char *p = get_data();\nprintf(\"%s\", p);  // p might be NULL"},
        example_safe={"c": 'char *p = get_data();\nif (p == NULL) { handle_error(); return; }\nprintf("%s", p);'},
    ),
    CWEEntry(
        id="CWE-287",
        name="Improper Authentication",
        description="The product does not prove or insufficiently proves that an actor's claimed identity is correct.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={"python": 'if request.headers.get("X-User") == "admin":\n    grant_access()  # trusting client header'},
        example_safe={"python": 'token = request.headers.get("Authorization")\nuser = verify_jwt(token)\nif user.is_admin:\n    grant_access()'},
    ),
    CWEEntry(
        id="CWE-190",
        name="Integer Overflow or Wraparound",
        description="The product performs a calculation that can produce an integer overflow or wraparound.",
        languages={"c", "cpp", "java"},
        example_vulnerable={"c": "size_t len = a + b;  // could overflow\nmalloc(len);"},
        example_safe={"c": "if (a > SIZE_MAX - b) { abort(); }\nsize_t len = a + b;\nmalloc(len);"},
    ),
    CWEEntry(
        id="CWE-502",
        name="Deserialization of Untrusted Data",
        description="The product deserializes untrusted data without sufficiently verifying that the resulting data will be valid.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={
            "python": "data = pickle.loads(request.data)  # arbitrary code execution",
            "perl": 'my $data = Storable::thaw($user_input);  # arbitrary code execution',
        },
        example_safe={
            "python": "data = json.loads(request.data)  # safe format",
            "perl": 'my $data = decode_json($user_input);  # safe format',
        },
    ),
    CWEEntry(
        id="CWE-77",
        name="Command Injection",
        description="The product constructs all or part of a command using externally-influenced input but does not neutralize or incorrectly neutralizes special elements that could modify the intended command.",
        languages=set(),
        example_vulnerable={"python": 'subprocess.run(f"convert {user_file} output.png", shell=True)'},
        example_safe={"python": 'subprocess.run(["convert", user_file, "output.png"])'},
    ),
    CWEEntry(
        id="CWE-119",
        name="Improper Restriction of Operations within the Bounds of a Memory Buffer",
        description="The product performs operations on a memory buffer but can read from or write to a memory location outside the intended boundary.",
        languages={"c", "cpp"},
        example_vulnerable={"c": "memcpy(dst, src, len);  // len not validated against dst size"},
        example_safe={"c": "if (len > sizeof(dst)) len = sizeof(dst);\nmemcpy(dst, src, len);"},
    ),
    CWEEntry(
        id="CWE-798",
        name="Use of Hard-coded Credentials",
        description="The product contains hard-coded credentials such as passwords or cryptographic keys.",
        languages=set(),
        example_vulnerable={"python": 'DB_PASSWORD = "supersecret123"\nconnect(password=DB_PASSWORD)'},
        example_safe={"python": 'DB_PASSWORD = os.environ["DB_PASSWORD"]\nconnect(password=DB_PASSWORD)'},
    ),
    CWEEntry(
        id="CWE-918",
        name="Server-Side Request Forgery (SSRF)",
        description="The product receives a URL or similar request from an upstream component and retrieves the contents of the URL, but does not sufficiently ensure that the request is being sent to the expected destination.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={"python": 'url = request.args["url"]\nresponse = requests.get(url)  # fetches arbitrary URLs'},
        example_safe={"python": 'url = request.args["url"]\nif not is_allowed_host(url):\n    abort(400)\nresponse = requests.get(url)'},
    ),
    CWEEntry(
        id="CWE-306",
        name="Missing Authentication for Critical Function",
        description="The product does not perform any authentication for functionality that requires a provable user identity.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={"python": '@app.route("/api/delete_user/<uid>", methods=["POST"])\ndef delete_user(uid):\n    User.delete(uid)  # no auth'},
        example_safe={"python": '@app.route("/api/delete_user/<uid>", methods=["POST"])\n@require_auth\ndef delete_user(uid):\n    User.delete(uid)'},
    ),
    CWEEntry(
        id="CWE-362",
        name="Race Condition",
        description="The product contains a code sequence that can run concurrently with other code, and the code sequence requires temporary, exclusive access to a shared resource, but a timing window exists in which the shared resource can be modified by another code sequence.",
        languages=set(),
        example_vulnerable={"python": "if os.path.exists(path):  # TOCTOU\n    os.remove(path)"},
        example_safe={"python": "try:\n    os.remove(path)\nexcept FileNotFoundError:\n    pass"},
    ),
    CWEEntry(
        id="CWE-269",
        name="Improper Privilege Management",
        description="The product does not properly assign, modify, track, or check privileges for an actor, creating an unintended sphere of control.",
        languages=set(),
        example_vulnerable={"python": "os.setuid(0)  # running as root unnecessarily"},
        example_safe={"python": "os.setuid(service_uid)  # drop to least privilege"},
    ),
    CWEEntry(
        id="CWE-94",
        name="Code Injection",
        description="The product constructs all or part of a code segment using externally-influenced input but does not neutralize or incorrectly neutralizes special elements that could modify the intended code segment.",
        languages={"python", "typescript", "javascript", "ruby", "php", "perl"},
        example_vulnerable={
            "python": 'result = eval(user_expression)  # arbitrary code execution',
            "perl": 'my $result = eval($user_input);  # arbitrary code execution',
        },
        example_safe={
            "python": "import ast\nresult = ast.literal_eval(user_expression)",
            "perl": '# Use a safe parser instead of eval\nmy $result = JSON::decode_json($user_input);',
        },
    ),
    CWEEntry(
        id="CWE-863",
        name="Incorrect Authorization",
        description="The product performs an authorization check but does not correctly determine whether the actor is authorized to access the resource.",
        languages={"python", "typescript", "javascript", "ruby", "php", "go", "java", "perl"},
        example_vulnerable={"python": 'if user.role == "user" or user.role == "admin":\n    # gives access to all users, not just admins'},
        example_safe={"python": 'if user.role == "admin":\n    grant_admin_access()'},
    ),
    CWEEntry(
        id="CWE-276",
        name="Incorrect Default Permissions",
        description="The product sets incorrect permissions during installation or resource creation, allowing unintended actors to access or modify the resource.",
        languages=set(),
        example_vulnerable={"python": "os.chmod(config_file, 0o777)  # world-writable"},
        example_safe={"python": "os.chmod(config_file, 0o600)  # owner-only"},
    ),
]


def applicable_cwes(language: str) -> list[CWEEntry]:
    """Return CWEs that apply to the given language."""
    return [
        cwe for cwe in CWE_TOP_25
        if not cwe.languages or language in cwe.languages
    ]


def build_prompt(file_path: str, file_content: str, language: str, cwe: CWEEntry) -> str:
    """Build the analysis prompt for a specific file and CWE."""
    # Pick language-specific example or fall back to python or first available
    vuln_example = (
        cwe.example_vulnerable.get(language)
        or cwe.example_vulnerable.get("python")
        or next(iter(cwe.example_vulnerable.values()), "N/A")
    )
    safe_example = (
        cwe.example_safe.get(language)
        or cwe.example_safe.get("python")
        or next(iter(cwe.example_safe.values()), "N/A")
    )

    return f"""You are a security auditor. Analyze the following {language} file for exactly one type of vulnerability:

{cwe.id} ({cwe.name}): {cwe.description}

Example of VULNERABLE code in {language}:
```
{vuln_example}
```

Example of SAFE code in {language}:
```
{safe_example}
```

IMPORTANT INSTRUCTIONS:
- Only look for {cwe.id} ({cwe.name}). Do not report other vulnerability types.
- If you find NO instances of this vulnerability, you MUST return exactly: []
- If you find instances, return a JSON array of objects with these fields:
  - "line": the line number (integer)
  - "code": the vulnerable code snippet (string, just the relevant lines)
  - "explanation": why this is vulnerable to {cwe.id} (string)
  - "confidence": "high", "medium", or "low" (string)
- Return ONLY the JSON array, no other text, no markdown fences, no explanation outside the JSON.
- Be precise. Only report actual vulnerabilities, not theoretical ones.
- Do not report vulnerabilities in comments or dead code.

File: {file_path}
```{language}
{file_content}
```"""


# Sentinel CWE entry for open-ended scanning
CWE_OPEN = CWEEntry(
    id="OPEN",
    name="Open Scan",
    description="Scan for any vulnerability without a specific CWE focus.",
    languages=set(),
    example_vulnerable={},
    example_safe={},
)


def build_open_prompt(file_path: str, file_content: str, language: str) -> str:
    """Build an open-ended vulnerability analysis prompt."""
    return f"""You are a security researcher performing a vulnerability audit. Analyze the following {language} file and find any security vulnerabilities.

Look for all classes of vulnerability including but not limited to:
- Memory safety issues (buffer overflows, use-after-free, etc.)
- Injection attacks (SQL, command, code, XSS, etc.)
- Authentication and authorization flaws
- Cryptographic weaknesses
- Race conditions
- Path traversal
- Deserialization issues
- Hard-coded credentials
- Improper input validation
- Any other security-relevant bugs

IMPORTANT INSTRUCTIONS:
- If you find NO vulnerabilities, you MUST return exactly: []
- If you find vulnerabilities, return a JSON array of objects with these fields:
  - "line": the line number (integer)
  - "code": the vulnerable code snippet (string, just the relevant lines)
  - "cwe": the CWE ID if you can identify one, otherwise "unknown" (string)
  - "explanation": what the vulnerability is and why it matters (string)
  - "confidence": "high", "medium", or "low" (string)
- Return ONLY the JSON array, no other text, no markdown fences, no explanation outside the JSON.
- Be precise. Only report actual vulnerabilities, not theoretical ones or style issues.
- Do not report vulnerabilities in comments or dead code.
- Rank by severity — put the most serious vulnerability first.

File: {file_path}
```{language}
{file_content}
```"""
