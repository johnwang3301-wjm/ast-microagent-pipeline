import os
import time
from pathlib import Path
from openai import OpenAI
from typing import Optional

# =====================================================================
# KCG-CONFIG: Batch AST Extraction Pipeline (Llmuni -> Qwen-27B)
# Strategy: Deterministic Batch Processing, Zero Hallucination
# Architect: Jiamiao Wang (John)
# =====================================================================

# 強制環境變量校驗
API_KEY = os.getenv("LLMUNI_API_KEY")
if not API_KEY:
    raise ValueError("FATAL ERROR: LLMUNI_API_KEY is missing from environment. Run 'export LLMUNI_API_KEY=\"your_key\"'")

BASE_URL = "https://llmuni.com/v1" 
TARGET_MODEL = "claude-opus-4-6-thinking" 

# I/O 邊界隔離
INPUT_DIR = Path("./input_papers")
OUTPUT_DIR = Path("./output_asts")

# =====================================================================
# 終極編譯器指令 (LISP / S-Expression 偽代碼)
# =====================================================================
MASTER_SYSTEM_PROMPT = """[SYSTEM INSTRUCTION BEGIN]

(DEFCONFIG :MODEL "Qwen-27B"
  :MODE 'STRICT_COMPILER
  :OUTPUT_FORMAT 'S-EXPRESSION
  :HALLUCINATION_POLICY 'ZERO_TOLERANCE)

(DEFSCHEMA MasterSchema
  (paper
    (metadata
      (title (type string))
      (authors (list string))
      (date (type string)))
    (methodology
      (keywords (list string))
      (algorithm (list string))
      (state (input (type string)) (output (type string)))
      (steps (list string))
      (tests (list string)))
    (grounding
      (entities (list (entity (name) (resolved-from))))
      (relations (list (relation (type) (source) (target))))
      (citations (list (citation (ref-id) (usage-context) (sufficiency-purpose)))))))

(DEFUN fn_direct_extract (RAW_TEXT)
  (LET ((result '()))
    (FOREACH field IN '(title authors date methodology_keywords)
      (IF (EXACT_MATCH_OR_DIRECT_STATEMENT_EXISTS RAW_TEXT field)
        (THEN (APPEND result (LIST field (EXTRACT_EXACT RAW_TEXT field))))
        (ELSE (APPEND result (LIST field '(nil))))))
    (RETURN result)))

(DEFUN fn_gather_and_extract (RAW_TEXT)
  (LET ((context_blocks (AGGREGATE_SPANS RAW_TEXT :targets '(algorithm state steps tests)))
        (result '()))
    (FOREACH target IN '(algorithm input output steps tests)
      (IF (SPAN_FOUND context_blocks target)
        (THEN
          (LET ((extracted (EXTRACT_STRICTLY_FROM_SPAN context_blocks target)))
            (IF (SYNTHESIZED_NEW_STEPS extracted)
              (THEN (APPEND result (LIST target '(error "not_found"))))
              (ELSE (APPEND result (LIST target extracted))))))
        (ELSE (APPEND result (LIST target '(nil))))))
    (RETURN result)))

(DEFUN fn_ground_and_justify (RAW_TEXT)
  (LET ((sentences (DEPENDENCY_PARSE RAW_TEXT))
        (entities '())
        (relations '())
        (citations '()))
    (FOREACH s IN sentences
      ;; Rule 1: Referent Grounding
      (FOREACH ref IN (FIND_UNDEFINED_REFERENTS s)
        (IF (IS_AMBIGUOUS ref)
          (THEN
            (LET ((resolved (TRAVERSE_AST_BACKWARD s :target 'NEAREST_EXPLICIT_NOUN)))
              (APPEND entities (LIST 'entity ref resolved))))
          (ELSE (APPEND entities (LIST 'entity ref '(nil))))))

      ;; Rule 2: Relational Ontology
      (FOREACH rel IN (EXTRACT_RELATIONS s :types '(part-of type-of causal))
        (APPEND relations (LIST 'relation (GET_TYPE rel) (GET_SOURCE rel) (GET_TARGET rel))))

      ;; Rule 3: Citations
      (FOREACH cite IN (FIND_CITATIONS s)
        (APPEND citations
          (LIST 'citation
                (EXTRACT_REF_ID cite)
                (DEFINE_USAGE_CONTEXT s cite)
                (EXTRACT_SUFFICIENCY_PURPOSE s cite :strict_justification #t)))))
    (RETURN (LIST 'grounding entities relations citations))))

(DEFUN execute_3_stage_pipeline (RAW_TEXT)
  (LET ((s1 (fn_direct_extract RAW_TEXT))
            (s2 (fn_gather_and_extract RAW_TEXT))
            (s3 (fn_ground_and_justify RAW_TEXT)))
    (RETURN (FORMAT_AS_S_EXPRESSION MasterSchema s1 s2 s3))))

(MAIN
  (LET ((input_text (READ_INPUT)))
    (PRINT (execute_3_stage_pipeline input_text))))

[SYSTEM INSTRUCTION END]"""

def init_ast_pipeline() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

def execute_extraction(client: OpenAI, raw_text: str) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=TARGET_MODEL,
            temperature=0.0, 
            messages=[
                {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"(execute_3_stage_pipeline (raw-text \"{raw_text}\"))"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] (pipeline_execution_failed \"{str(e)}\")")
        return f"(error \"model_inference_failed\" \"{str(e)}\")"

def process_batch():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = list(INPUT_DIR.glob("*.txt"))
    if not input_files:
        print(f"(status \"SYSTEM HALT: Drop raw paper files (.txt) into {INPUT_DIR} and restart.\")")
        return

    print(f"(status \"Initializing batch AST extraction for {len(input_files)} targets...\")")
    client = init_ast_pipeline()

    for file_path in input_files:
        print(f"\n(process-start \"{file_path.name}\")")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        result_ast = execute_extraction(client, raw_text)
        
        if result_ast:
            output_filename = file_path.stem + "_extracted.lisp"
            output_path = OUTPUT_DIR / output_filename
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(result_ast)
            print(f"(process-success \"{output_filename}\" \"AST written to {OUTPUT_DIR}\")")
        
        time.sleep(2.5) # Rate limit 控制

if __name__ == "__main__":
    process_batch()
    print("\n(status \"All pipeline tasks finalized.\")")
