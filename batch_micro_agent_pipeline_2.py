import os
import time
from pathlib import Path
from openai import OpenAI
from typing import Optional, Tuple

# =====================================================================
# KCG-CONFIG: Sequential Micro-Agent AST Extraction Pipeline
# Architecture: 4-Node Pipeline (Direct -> Gather -> Ground -> Justify)
# Status: OpenRouter-Ready (Currently routed via local aggregator for testing)
# Architect: Jiamiao Wang (John)
# =====================================================================

# Strict environment variable validation. Reject hardcoding.
API_KEY = os.getenv("LLMUNI_API_KEY") # Configured for local aggregator test run
if not API_KEY:
    raise ValueError("FATAL ERROR: LLMUNI_API_KEY is missing from environment. Run 'export LLMUNI_API_KEY=\"...\"'")

BASE_URL = "https://api.llmuni.com/v1" # Swap to "https://openrouter.ai/api/v1" for production
TARGET_MODEL = "qwen3.8-27b" # Swap to specific OpenRouter Qwen model tag

# Strict I/O boundary isolation
INPUT_DIR = Path("./input_papers")
OUTPUT_DIR = Path("./output_asts")

# =====================================================================
# SYSTEM PROMPTS (Mapped from "Sequential Agent Pipeline Prompt.docx")
# =====================================================================

PROMPT_AGENT_1 = """[SYSTEM INSTRUCTION BEGIN]
(DEFCONFIG :MODEL "Qwen-27B"  :AGENT_ROLE 'STAGE_1_EXTRACTOR  :INPUT_FORMAT '(RAW_TEXT)  :OUTPUT_FORMAT 'S-EXPRESSION  :HALLUCINATION_POLICY 'ZERO_TOLERANCE)
(DEFSCHEMA Stage1Output  (stage1-result    (metadata      (title (type string))      (authors (list string))      (date (type string)))    (methodology_keywords (list string))))
(DEFUN fn_direct_extract (RAW_TEXT)  (LET ((result '()))    (FOREACH field IN '(title authors date methodology_keywords)      (IF (EXACT_MATCH_OR_DIRECT_STATEMENT_EXISTS RAW_TEXT field)        (THEN (APPEND result (LIST field (EXTRACT_EXACT RAW_TEXT field))))        (ELSE (APPEND result (LIST field '(nil))))))    (RETURN (LIST 'stage1-result (GET-NODE result 'metadata) (GET-NODE result 'methodology_keywords)))))
(MAIN  (LET ((input_text (READ_INPUT)))    (PRINT (fn_direct_extract input_text))))
[SYSTEM INSTRUCTION END]"""

PROMPT_AGENT_2 = """[SYSTEM INSTRUCTION BEGIN]
(DEFCONFIG :MODEL "Qwen-27B"  :AGENT_ROLE 'STAGE_2_GATHERER  :INPUT_FORMAT '(RAW_TEXT STAGE_1_SEXP)  :OUTPUT_FORMAT 'S-EXPRESSION  :HALLUCINATION_POLICY 'ZERO_TOLERANCE)
(DEFSCHEMA Stage2Output  (stage2-result    (INHERIT Stage1Output)    (methodology_details      (algorithm (list string))      (state (input (type string)) (output (type string)))      (steps (list string))      (tests (list string)))))
(DEFUN fn_gather_and_extract (RAW_TEXT STAGE_1_SEXP)  (LET ((context_blocks (AGGREGATE_SPANS RAW_TEXT :targets '(algorithm state steps tests)))        (meth_result '()))    (FOREACH target IN '(algorithm input output steps tests)      (IF (SPAN_FOUND context_blocks target)        (THEN          (LET ((extracted (EXTRACT_STRICTLY_FROM_SPAN context_blocks target)))            (IF (SYNTHESIZED_NEW_STEPS extracted)              (THEN (APPEND meth_result (LIST target '(error "not_found"))))              (ELSE (APPEND meth_result (LIST target extracted))))))        (ELSE (APPEND meth_result (LIST target '(nil))))))    (RETURN (MERGE_SEXP STAGE_1_SEXP (LIST 'methodology_details meth_result)))))
(MAIN  (LET ((raw (READ_INPUT :idx 0))        (s1_data (READ_INPUT :idx 1)))    (PRINT (fn_gather_and_extract raw s1_data))))
[SYSTEM INSTRUCTION END]"""

PROMPT_AGENT_3A = """[SYSTEM INSTRUCTION BEGIN]
(DEFCONFIG :MODEL "Qwen-27B"
  :AGENT_ROLE 'STAGE_3A_INTERNAL_GROUNDER
  :INPUT_FORMAT '(RAW_TEXT STAGE_2_SEXP)
  :OUTPUT_FORMAT 'S-EXPRESSION
  :HALLUCINATION_POLICY 'ZERO_TOLERANCE)
(DEFSCHEMA Stage3AOutput
  (stage3a-result
    (INHERIT Stage2Output)
    (internal_grounding
      (entities (list (entity (name) (resolved-from))))
      (relations (list (relation (type) (source) (target)))))))
(DEFUN fn_ground_entities_and_relations (RAW_TEXT STAGE_2_SEXP)
  (LET ((sentences (DEPENDENCY_PARSE RAW_TEXT))
        (entities '())
        (relations '()))
    (FOREACH s IN sentences
      ;; Rule 1: Referent Grounding (Strict AST Traversal)
      (FOREACH ref IN (FIND_UNDEFINED_REFERENTS s)
        (IF (IS_AMBIGUOUS ref)
          (THEN
            (LET ((resolved (TRAVERSE_AST_BACKWARD s :target 'NEAREST_EXPLICIT_NOUN)))
              (APPEND entities (LIST 'entity ref resolved))))
          (ELSE (APPEND entities (LIST 'entity ref '(nil))))))
      ;; Rule 2: Relational Ontology (Strict Mapping)
      (FOREACH rel IN (EXTRACT_RELATIONS s :types '(part-of type-of causal)))
        (IF (RELATION_EXISTS_IN_TEXT rel s)
          (THEN (APPEND relations (LIST 'relation (GET_TYPE rel) (GET_SOURCE rel) (GET_TARGET rel))))
          (ELSE 'CONTINUE)))
          
    (LET ((internal_block (LIST 'internal_grounding entities relations)))
      (RETURN (MERGE_SEXP STAGE_2_SEXP internal_block)))))
(MAIN
  (LET ((raw (READ_INPUT :idx 0))
        (s2_data (READ_INPUT :idx 1)))
    (PRINT (fn_ground_entities_and_relations raw s2_data))))
[SYSTEM INSTRUCTION END]"""

PROMPT_AGENT_3B = """[SYSTEM INSTRUCTION BEGIN]
(DEFCONFIG :MODEL "Qwen-27B"
  :AGENT_ROLE 'STAGE_3B_CITATION_JUSTIFIER
  :INPUT_FORMAT '(RAW_TEXT STAGE_3A_SEXP)
  :OUTPUT_FORMAT 'S-EXPRESSION
  :HALLUCINATION_POLICY 'ZERO_TOLERANCE)
(DEFSCHEMA MasterSchema
  (master-result
    (INHERIT Stage3AOutput)
    (external_grounding
      (citations (list (citation (ref-id) (usage-context) (sufficiency-purpose)))))))
(DEFUN fn_justify_citations (RAW_TEXT STAGE_3A_SEXP)
  (LET ((sentences (DEPENDENCY_PARSE RAW_TEXT))
        (citations '()))
    (FOREACH s IN sentences
      ;; Rule 3: Citations Justification (High Risk of Hallucination - Enforce Strict Bounds)
      (FOREACH cite IN (FIND_CITATIONS s)
        (LET ((ref_id (EXTRACT_REF_ID cite))
              (usage (DEFINE_USAGE_CONTEXT s cite))
              (purpose (EXTRACT_SUFFICIENCY_PURPOSE s cite :strict_justification #t)))
          (IF (OR (IS_EMPTY purpose) (IS_SYNTHESIZED_BY_MODEL purpose))
            (THEN (APPEND citations (LIST 'citation ref_id usage '(nil))))
            (ELSE (APPEND citations (LIST 'citation ref_id usage purpose)))))))
            
    (LET ((external_block (LIST 'external_grounding citations)))
      (RETURN (MERGE_SEXP STAGE_3A_SEXP external_block)))))
(MAIN
  (LET ((raw (READ_INPUT :idx 0))
        (s3a_data (READ_INPUT :idx 1)))
    (PRINT (fn_justify_citations raw s3a_data))))
[SYSTEM INSTRUCTION END]"""

def init_pipeline_client() -> OpenAI:
    """Pure OpenAI client instantiation, stripped of all local proxy dependencies."""
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

def call_micro_agent(client: OpenAI, agent_name: str, system_prompt: str, user_payload: str) -> Tuple[bool, str]:
    """
    Universal Agent Dispatcher.
    Enforces temperature=0.0 to strictly prevent hallucinations.
    """
    print(f"  -> [Awaiting {agent_name} Node Response...]")
    try:
        response = client.chat.completions.create(
            model=TARGET_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ]
        )
        result = response.choices[0].message.content.strip()
        return True, result
    except Exception as e:
        return False, f"(error \"{agent_name}_inference_failed\" \"{str(e)}\")"

def process_single_paper(client: OpenAI, filename: str, raw_text: str) -> str:
    """
    Core sequential logic: The previous agent's output MUST be fed as a strongly-typed 
    parameter to the next agent. Triggers a circuit breaker upon any node failure.
    """
    # ---------------------------------------------------------
    # Node 1: Direct Extraction
    # ---------------------------------------------------------
    payload_1 = f"(execute-extraction (raw-text \"{raw_text}\"))"
    success, s1_sexp = call_micro_agent(client, "Agent-1 (Lexer)", PROMPT_AGENT_1, payload_1)
    if not success: return s1_sexp

    # ---------------------------------------------------------
    # Node 2: Gather & Extract
    # ---------------------------------------------------------
    payload_2 = f"(execute-extraction (raw-text \"{raw_text}\")\n (stage1-sexp {s1_sexp}))"
    success, s2_sexp = call_micro_agent(client, "Agent-2 (Gatherer)", PROMPT_AGENT_2, payload_2)
    if not success: return s2_sexp

    # ---------------------------------------------------------
    # Node 3A: Entity & Relation Grounder
    # ---------------------------------------------------------
    payload_3a = f"(execute-extraction (raw-text \"{raw_text}\")\n (stage2-sexp {s2_sexp}))"
    success, s3a_sexp = call_micro_agent(client, "Agent-3A (Internal Grounder)", PROMPT_AGENT_3A, payload_3a)
    if not success: return s3a_sexp

    # ---------------------------------------------------------
    # Node 3B: Citation Justifier
    # ---------------------------------------------------------
    payload_3b = f"(execute-extraction (raw-text \"{raw_text}\")\n (stage3a-sexp {s3a_sexp}))"
    success, final_master_sexp = call_micro_agent(client, "Agent-3B (External Justifier)", PROMPT_AGENT_3B, payload_3b)
    
    return final_master_sexp

def execute_batch_pipeline():
    """Batch processing throughput engine ensuring I/O isolation and rate limiting."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = list(INPUT_DIR.glob("*.txt"))
    if not input_files:
        print(f"(status \"SYSTEM HALT: No raw paper files (.txt) found in {INPUT_DIR}.\")")
        return

    print(f"(status \"Initializing Micro-Agent Pipeline for {len(input_files)} unparsed papers...\")")
    client = init_pipeline_client()

    for file_path in input_files:
        print(f"\n(process-start \"{file_path.name}\")")
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Execute 4-node sequential inference
        master_ast = process_single_paper(client, file_path.name, raw_text)
        
        # Construct the static typing schema header requested by Synth
        schema_header = """(DEFSCHEMA MasterSchema
  (master-result
    (metadata (title string) (authors (list string)) (date string))
    (methodology_details (algorithm (list string)) (state (input string) (output string)) (steps (list string)) (tests (list string)))
    (internal_grounding (entities (list entity)) (relations (list relation)))
    (external_grounding (citations (list citation)))))
---
"""
        # Clean markdown code block tags to ensure pure LISP structure
        clean_ast = master_ast.replace("```scheme", "").replace("```", "").strip()
        
        output_filename = file_path.stem + "_MasterSchema.lisp"
        output_path = OUTPUT_DIR / output_filename
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(schema_header + clean_ast)
        
        print(f"(process-success \"{output_filename}\" \"Data mapped and exported to {OUTPUT_DIR}\")")
        
        # Sleep to relieve concurrent pressure on the API provider
        time.sleep(2.0)

if __name__ == "__main__":
    execute_batch_pipeline()
    print("\n(status \"All micro-agent pipeline tasks finalized. Exiting safely.\")")
