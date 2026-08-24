import openai

client = openai.OpenAI(base_url="http://localhost:9003/v1", api_key="TEST_API")

SYSTEM_PROMPT = """
Look at the following two expressions (answers to a puzzle problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Answer: Yes

    Expression 1: 3/2
    Expression 2: 1.5

Answer: Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

Answer: No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Answer: Yes

    Expression 1: 3245/5
    Expression 2: 649

Answer: No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: **suivliS**.
    Expression 2: lufesu

Answer: No
(These 2 strings are not equal)

    Expression 1: **lufesu**.
    Expression 2: lufesu

Answer: Yes

    Expression 1: 2/(-3)
    Expression 2: -2/3

Answer: Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Answer: Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Answer: Yes
(give benefit of the doubt to units)

    Expression 1: 0xb5fc
    Expression 2: 0xb5ff

Answer: No
(bytes matching these are not the same)

    Expression 1: 0x4537
    Expression 2: -0x2392

Answer: No
(bytes matching these are not the same)

    Expression 1: llebkc
    Expression 2: llebkcoc

Answer: No
(string matching, expression 2 has 2 more characters)

    Expression 1: 0x201a
    Expression 2: 0x201a

Answer: Yes

    Expression 1: {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8} 
    Expression 2: {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}

Answer: No

    Expression 1: {'R': 8, 'D': 7, 'M': 1, 'O': 0, 'N': 6, 'S': 9, 'E': 5} 
    Expression 2: {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8}

Answer: Yes

    Expression 1: 64
    Expression 2: 64 square feet

Answer: Yes


---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale. If any of the Expression 1 or 2 contain code, always return No
"""

EQUALITY_TEMPLATE = r"""
Expression 1: %(expression1)s
Expression 2: %(expression2)s
""".strip()



_DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

def llm_as_judge_equal(expr1, expr2, model=None, base_url=None, api_key=None, timeout=30):
    """Use an LLM to judge whether two math expressions are equivalent.

    Returns True if the model considers them equivalent, False otherwise.
    """
    import openai as _openai
    _model = model or _DEFAULT_MODEL
    _client = _openai.OpenAI(base_url=base_url, api_key=api_key) if (base_url or api_key) else client
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    try:
        response = _client.chat.completions.create(
            model=_model,
            messages=[{"role": 'system', 'content': SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
            timeout=timeout,
        )
        answer = response.choices[0].message.content.strip().lower()
        # Strip any thinking tags if present
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        return answer.lower().startswith("yes")
    except Exception as e:
        print(f"LLM judge error: {e}")
        return False


if __name__ == "__main__":
    # Quick sanity tests
    tests = [
        ("2x+3", "3+2x", True),
        ("3/2", "1.5", True),
        ("x^2+2x+1", "y^2+2y+1", False),
        ("x^2+2x+1", "(x+1)^2", True),
        ("72 degrees", "72", True),
        ("(x**3 - 7*x**2)*(6*x**3 + 10*x**2)", "6*x**6 - 32*x**5 - 70*x**4", True),
        ("53 - (57 - 42) - 10", "42 + 53 - 10 - 57", True),
        ("(57 - 53) * (10 + 42) / 10", "42 + 53 - 10 - 57", False),
        ("57 - 53 + 42 - 10", "42 + 53 - 10 - 57", False),
        ("0xc738", "0x33fa", False),
        ("nikkot", "nikmoT", False),
    ]

    for e1, e2, expected in tests:
        result = llm_as_judge_equal(e1, e2)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] '{e1}' == '{e2}' => {result} (expected {expected})")
