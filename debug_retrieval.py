# debug_retrieval.py
from backend.retrieval.retrieval import Retriever
from backend.generation.context_builder import ContextBuilder
from backend.generation.prompt_builder import PromptBuilder
from backend.llm.ollama_streaming import OllamaStreamingLLM
from backend.generation.citation_validator import CitationValidator
from backend.generation.grounding_validator import GroundingValidator

query = "How many days of leave do full-time employees get?"

# Step 1: Retrieve
r = Retriever(storage_path="test_storage")
results = r.search(query)


print("\nRETRIEVAL SCORES:")
for r_item in results:
    print(r_item["score"])
print()

if not r.confidence_gate(results, threshold=0.60):
    print("Insufficient information in provided documents.")
    exit()

# Step 2: Build context
context = ContextBuilder().build(results)

# Step 3: Build prompt
prompt = PromptBuilder().build(query, context)

# Step 4: Initialize LLM
llm = OllamaStreamingLLM(model_name="llama3:8b")

print("\n--- LLM RESPONSE ---\n")

full_response = ""

# Step 5: Stream response
for token in llm.stream_chat(prompt["system"], prompt["user"]):
    print(token, end="", flush=True)
    full_response += token

print("\n")

# Step 6: Validate citations
allowed_ids = {r["chunk"].chunk_id for r in results}

validator = CitationValidator()
is_valid = validator.validate(full_response, allowed_ids)

print("CITATION VALID:", is_valid)

# Step 6: Grounded Validation
allowed_ids = {r["chunk"].chunk_id for r in results}

validator = GroundingValidator()
is_valid = validator.validate(full_response, allowed_ids)

print("GROUNDED VALID:", is_valid)

if not is_valid:
    print("Insufficient information in provided documents.")