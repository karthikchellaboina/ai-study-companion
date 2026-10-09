from rag.generator import Generator

generator = Generator()

# Test 1: Valid answer
answer = generator._validate_output(
    "Python is a programming language. [Source 1]",
    1,
)
assert "[Source 1]" in answer
print("Valid citation: PASS")

# Test 2: Remove a citation to a nonexistent source
answer = generator._validate_output(
    "Python is useful. [Source 5]",
    1,
)
assert "[Source 5]" not in answer
print("Invalid citation removed: PASS")

# Test 3: Remove multiple invalid citations
answer = generator._validate_output(
    "Topic A [Source 1]. Topic B [Source 3].",
    1,
)
assert "[Source 1]" in answer
assert "[Source 3]" not in answer
print("Mixed citations: PASS")

# Test 4: Reject an empty response
try:
    generator._validate_output("   ", 1)
    print("Empty response: FAIL")
except ValueError:
    print("Empty response: PASS")

print("All output guardrail tests completed!")
