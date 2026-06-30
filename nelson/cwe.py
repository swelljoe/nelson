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


# CWE Top 25 (2025) with language applicability.
# Source: https://cwe.mitre.org/top25/archive/2025/2025_cwe_top25.html
# Languages left empty means applicable to all. Not stored in rank order.
CWE_TOP_25: list[CWEEntry] = [
    CWEEntry(
        id="CWE-787",
        name="Out-of-bounds Write",
        description="The product writes data past the end, or before the beginning, of the intended buffer.",
        languages={"c", "cpp"},
        example_vulnerable={
            "c": "char buf[10];\nstrcpy(buf, user_input);  // no bounds check"
        },
        example_safe={
            "c": "char buf[10];\nstrncpy(buf, user_input, sizeof(buf) - 1);\nbuf[sizeof(buf) - 1] = '\\0';"
        },
    ),
    CWEEntry(
        id="CWE-79",
        name="Cross-site Scripting (XSS)",
        description="The product does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output used as a web page served to other users.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": 'return f"<p>Hello {user_name}</p>"  # unsanitized'
        },
        example_safe={
            "python": 'from markupsafe import escape\nreturn f"<p>Hello {escape(user_name)}</p>"'
        },
    ),
    CWEEntry(
        id="CWE-89",
        name="SQL Injection",
        description="The product constructs all or part of an SQL command using externally-influenced input without neutralizing special elements that could modify the intended SQL command.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": "cursor.execute(f\"SELECT * FROM users WHERE name = '{user_input}'\")",
            "perl": "my $sth = $dbh->prepare(\"SELECT * FROM users WHERE name = '$input'\");",
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
        example_vulnerable={"c": 'free(ptr);\nprintf("%s", ptr);  // use after free'},
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
        example_vulnerable={
            "python": "def set_port(port_str):\n    port = int(port_str)  # no range check\n    listen(port)"
        },
        example_safe={
            "python": 'def set_port(port_str):\n    port = int(port_str)\n    if not (1 <= port <= 65535):\n        raise ValueError("invalid port")\n    listen(port)'
        },
    ),
    CWEEntry(
        id="CWE-125",
        name="Out-of-bounds Read",
        description="The product reads data past the end, or before the beginning, of the intended buffer.",
        languages={"c", "cpp"},
        example_vulnerable={
            "c": "int arr[10];\nreturn arr[index];  // index not validated"
        },
        example_safe={
            "c": "int arr[10];\nif (index >= 0 && index < 10) return arr[index];"
        },
    ),
    CWEEntry(
        id="CWE-22",
        name="Path Traversal",
        description="The product uses external input to construct a pathname intended to identify a file or directory below a restricted parent, but does not properly neutralize special elements like '..' that can resolve to a location outside of that directory.",
        languages=set(),
        example_vulnerable={
            "python": "path = os.path.join(BASE_DIR, user_filename)\nreturn open(path).read()"
        },
        example_safe={
            "python": 'path = os.path.join(BASE_DIR, user_filename)\nif not os.path.realpath(path).startswith(os.path.realpath(BASE_DIR)):\n    raise ValueError("path traversal")\nreturn open(path).read()'
        },
    ),
    CWEEntry(
        id="CWE-352",
        name="Cross-Site Request Forgery (CSRF)",
        description="The web application does not sufficiently verify that a well-formed, valid, consistent request was intentionally provided by the user who submitted the request.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": '@app.route("/transfer", methods=["POST"])\ndef transfer():\n    # no CSRF token check'
        },
        example_safe={
            "python": '@app.route("/transfer", methods=["POST"])\n@csrf_protect\ndef transfer():\n    ...'
        },
    ),
    CWEEntry(
        id="CWE-434",
        name="Unrestricted Upload of File with Dangerous Type",
        description="The product allows the upload of files without restricting the file type.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": 'uploaded = request.files["file"]\nuploaded.save(os.path.join(UPLOAD_DIR, uploaded.filename))'
        },
        example_safe={
            "python": 'uploaded = request.files["file"]\nif not allowed_extension(uploaded.filename):\n    abort(400)\nfilename = secure_filename(uploaded.filename)\nuploaded.save(os.path.join(UPLOAD_DIR, filename))'
        },
    ),
    CWEEntry(
        id="CWE-862",
        name="Missing Authorization",
        description="The product does not perform an authorization check when an actor attempts to access a resource or perform an action.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": '@app.route("/admin/users")\ndef list_users():\n    return jsonify(User.query.all())  # no auth check'
        },
        example_safe={
            "python": '@app.route("/admin/users")\n@login_required\n@admin_required\ndef list_users():\n    return jsonify(User.query.all())'
        },
    ),
    CWEEntry(
        id="CWE-476",
        name="NULL Pointer Dereference",
        description="The product dereferences a pointer that it expects to be valid but is NULL.",
        languages={"c", "cpp"},
        example_vulnerable={
            "c": 'char *p = get_data();\nprintf("%s", p);  // p might be NULL'
        },
        example_safe={
            "c": 'char *p = get_data();\nif (p == NULL) { handle_error(); return; }\nprintf("%s", p);'
        },
    ),
    CWEEntry(
        id="CWE-502",
        name="Deserialization of Untrusted Data",
        description="The product deserializes untrusted data without sufficiently verifying that the resulting data will be valid.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": "data = pickle.loads(request.data)  # arbitrary code execution",
            "perl": "my $data = Storable::thaw($user_input);  # arbitrary code execution",
        },
        example_safe={
            "python": "data = json.loads(request.data)  # safe format",
            "perl": "my $data = decode_json($user_input);  # safe format",
        },
    ),
    CWEEntry(
        id="CWE-77",
        name="Command Injection",
        description="The product constructs all or part of a command using externally-influenced input but does not neutralize or incorrectly neutralizes special elements that could modify the intended command.",
        languages=set(),
        example_vulnerable={
            "python": 'subprocess.run(f"convert {user_file} output.png", shell=True)'
        },
        example_safe={"python": 'subprocess.run(["convert", user_file, "output.png"])'},
    ),
    CWEEntry(
        id="CWE-918",
        name="Server-Side Request Forgery (SSRF)",
        description="The product receives a URL or similar request from an upstream component and retrieves the contents of the URL, but does not sufficiently ensure that the request is being sent to the expected destination.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": 'url = request.args["url"]\nresponse = requests.get(url)  # fetches arbitrary URLs'
        },
        example_safe={
            "python": 'url = request.args["url"]\nif not is_allowed_host(url):\n    abort(400)\nresponse = requests.get(url)'
        },
    ),
    CWEEntry(
        id="CWE-306",
        name="Missing Authentication for Critical Function",
        description="The product does not perform any authentication for functionality that requires a provable user identity.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": '@app.route("/api/delete_user/<uid>", methods=["POST"])\ndef delete_user(uid):\n    User.delete(uid)  # no auth'
        },
        example_safe={
            "python": '@app.route("/api/delete_user/<uid>", methods=["POST"])\n@require_auth\ndef delete_user(uid):\n    User.delete(uid)'
        },
    ),
    CWEEntry(
        id="CWE-94",
        name="Code Injection",
        description="The product constructs all or part of a code segment using externally-influenced input but does not neutralize or incorrectly neutralizes special elements that could modify the intended code segment.",
        languages={"python", "typescript", "javascript", "ruby", "php", "perl"},
        example_vulnerable={
            "python": "result = eval(user_expression)  # arbitrary code execution",
            "perl": "my $result = eval($user_input);  # arbitrary code execution",
        },
        example_safe={
            "python": "import ast\nresult = ast.literal_eval(user_expression)",
            "perl": "# Use a safe parser instead of eval\nmy $result = JSON::decode_json($user_input);",
        },
    ),
    CWEEntry(
        id="CWE-863",
        name="Incorrect Authorization",
        description="The product performs an authorization check but does not correctly determine whether the actor is authorized to access the resource.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
        },
        example_vulnerable={
            "python": 'if user.role == "user" or user.role == "admin":\n    # gives access to all users, not just admins'
        },
        example_safe={"python": 'if user.role == "admin":\n    grant_admin_access()'},
    ),
    # ── New in the 2025 edition (replacing 2023's 190/119/798/287/362/269/276) ──
    CWEEntry(
        id="CWE-120",
        name="Buffer Copy without Checking Size of Input (Classic Buffer Overflow)",
        description="The product copies an input buffer to an output buffer without verifying that the size of the input buffer is less than the size of the output buffer.",
        languages={"c", "cpp"},
        example_vulnerable={
            "c": "char dst[64];\nstrcpy(dst, src);  // src may exceed 64 bytes"
        },
        example_safe={"c": 'char dst[64];\nsnprintf(dst, sizeof(dst), "%s", src);'},
    ),
    CWEEntry(
        id="CWE-121",
        name="Stack-based Buffer Overflow",
        description="A buffer overflow where the buffer that can be overwritten is allocated on the stack (a local variable or a function parameter).",
        languages={"c", "cpp"},
        example_vulnerable={
            "c": "char buf[16];\ngets(buf);  // unbounded write to a stack buffer"
        },
        example_safe={
            "c": "char buf[16];\nif (fgets(buf, sizeof(buf), stdin) == NULL) return;"
        },
    ),
    CWEEntry(
        id="CWE-122",
        name="Heap-based Buffer Overflow",
        description="A buffer overflow where the buffer that can be overwritten is allocated in the heap (e.g., memory from malloc()).",
        languages={"c", "cpp"},
        example_vulnerable={
            "c": "char *b = malloc(8);\nmemcpy(b, src, len);  // len may exceed 8"
        },
        example_safe={"c": "char *b = malloc(len);\nif (b) memcpy(b, src, len);"},
    ),
    CWEEntry(
        id="CWE-284",
        name="Improper Access Control",
        description="The product does not restrict or incorrectly restricts access to a resource from an unauthorized actor.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
            "rust",
        },
        example_vulnerable={
            "python": '@app.route("/admin")\ndef admin():\n    return render_admin()  # no access check'
        },
        example_safe={
            "python": '@app.route("/admin")\ndef admin():\n    if not current_user.is_admin:\n        abort(403)\n    return render_admin()'
        },
    ),
    CWEEntry(
        id="CWE-200",
        name="Exposure of Sensitive Information to an Unauthorized Actor",
        description="The product exposes sensitive information to an actor that is not explicitly authorized to have access to that information.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
            "rust",
        },
        example_vulnerable={
            "python": "return jsonify(user.__dict__)  # leaks password_hash, tokens"
        },
        example_safe={"python": 'return jsonify({"id": user.id, "name": user.name})'},
    ),
    CWEEntry(
        id="CWE-639",
        name="Authorization Bypass Through User-Controlled Key",
        description="The authorization functionality does not prevent one user from accessing another user's data by modifying the key value identifying that data.",
        languages={
            "python",
            "typescript",
            "javascript",
            "ruby",
            "php",
            "go",
            "java",
            "perl",
            "rust",
        },
        example_vulnerable={
            "python": 'doc = Document.get(request.args["id"])\nreturn doc.body  # no owner check on id'
        },
        example_safe={
            "python": 'doc = Document.get(request.args["id"])\nif doc.owner_id != current_user.id:\n    abort(403)\nreturn doc.body'
        },
    ),
    CWEEntry(
        id="CWE-770",
        name="Allocation of Resources Without Limits or Throttling",
        description="The product allocates a reusable resource or group of resources on behalf of an actor without imposing any restrictions on the size or number that can be allocated.",
        languages=set(),
        example_vulnerable={
            "python": "data = request.stream.read()  # unbounded read into memory"
        },
        example_safe={
            "python": "data = request.stream.read(MAX_BYTES)  # bounded read"
        },
    ),
]


# Rust is a general-purpose backend/CLI language subject to the same web/logic
# weakness classes as Go/Java/Python (auth, authz, injection, SSRF, deserialization,
# SSRF, missing-authn, ...), even though *safe* Rust rules out most memory-safety
# classes (those stay c/cpp). The corpus's Rust authorization case (GHSA-f26g, a
# source-confusion authz bug) made the omission concrete. Augment the relevant
# entries here rather than hand-editing each literal `languages` set.
_RUST_WEB_LOGIC_CWES = {
    "CWE-79",  # XSS — server-side templating (askama, maud)
    "CWE-89",  # SQL injection — raw queries via sqlx/diesel
    "CWE-352",  # CSRF
    "CWE-434",  # unrestricted file upload
    "CWE-862",  # missing authorization
    "CWE-863",  # incorrect authorization (the GHSA-f26g class)
    "CWE-502",  # unsafe deserialization (serde + untrusted formats)
    "CWE-918",  # SSRF
    "CWE-306",  # missing authentication for a critical function
}
# (CWE-284/200/639, new in the 2025 edition, already list "rust" inline.)
for _entry in CWE_TOP_25:
    if _entry.id in _RUST_WEB_LOGIC_CWES:
        _entry.languages.add("rust")
del _entry

# Names for weakness classes present in the benchmark corpus but outside the Top
# 25, so the oracle-CWE hint can render a human-readable name for them too.
# (CWE-639 and CWE-122 graduated into the 2025 Top 25, so they resolve from
# CWE_TOP_25 directly and no longer need a supplemental entry.)
_SUPPLEMENTAL_CWE_NAMES = {
    "CWE-349": "Acceptance of Extraneous Untrusted Data With Trusted Data",
    "CWE-183": "Permissive List of Allowed Inputs",
    "CWE-501": "Trust Boundary Violation",
}


def cwe_name(cwe_id: str) -> str | None:
    """Human-readable name for a CWE id, or None if unknown.

    Resolves against the Top-25 set first, then a small supplemental map of
    weakness classes that appear in the benchmark corpus but not the Top 25."""
    for entry in CWE_TOP_25:
        if entry.id == cwe_id:
            return entry.name
    return _SUPPLEMENTAL_CWE_NAMES.get(cwe_id)


def applicable_cwes(language: str) -> list[CWEEntry]:
    """Return CWEs that apply to the given language."""
    return [cwe for cwe in CWE_TOP_25 if not cwe.languages or language in cwe.languages]


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
