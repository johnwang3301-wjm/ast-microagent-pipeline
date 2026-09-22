MANDATORY_DATA_SIGNATURE:
Required_Fields: Only the fields that the computation reads, bundled into 1 struct.
Access_Modifiers: Treated as read-only during the compute phase, instantiated as a fresh Ctx struct for each invocation, never cached, and never reused.
Prohibited_Elements: Unused fields, methods containing logic inside the struct, and references to the full state object.
GCA_LIFECYCLE_BOUNDARIES:
Gather_Phase:
Input_Source: The required fields from the source state object.
Constraint: Bundle inputs into a new struct named <Computation>Ctx, copying the field names of the source state or applying a clearer purpose name.
Compute_Phase:
Execution: Logic is restricted to reading the inputs provided in the Ctx struct, ensuring no unrelated state is read.
Side_Effect_Prohibitions: Modifying or writing to the Ctx struct fields is strictly prohibited.
Apply_Phase:
Write_Target: Outputs of the computation must be written to a Result struct.
Concurrency_Rule: Not specified in pattern.gca.context-struct.txt.
EXPLICIT_BANS:  
 Holding an unused field in a Ctx struct.  
 Caching a Ctx struct.  
 Reusing a Ctx struct.  
 Writing a method with logic inside a Ctx struct.  
 Holding a reference to the full state object in a Ctx struct.  
 Writing to Ctx struct fields during the compute function.

[VUKNERABILITY_ANALYSIS]

Stack Copying Overhead: The mandate to "ALWAYS create a fresh Ctx struct for each invocation" guarantees massive stack copying. A struct like ⁠ScriptExecutionCtx⁠ holding 5 reference fields consumes 40 bytes on a 64-bit architecture. Passing this struct by value into the compute function destroys memory bandwidth in hot paths.  

Honor-System Immutability: The documentation insists the struct is "Treated as read-only during the compute phase" and lists "Writing to Ctx struct fields" under explicitly banned actions.However, this relies entirely on human discipline. Without compiler-level enforcement, RyuJIT cannot safely assume immutability, missing critical register optimization opportunities.

Pointer-Chasing Architecture:Holding heap_allocated reference type like StsState and ⁠MechList⁠ inside the Context struct violates strict DOD principles. The Compute phase will suffer persistent L1/L2 cache misses as it chases pointers across scattered heap memory instead of iterating over contiguous data blocks.  ss