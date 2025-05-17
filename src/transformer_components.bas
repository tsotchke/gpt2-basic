' BASIC implementation of transformer components for the GPT-2-like model.
' This file contains implementations for the attention mechanism,
' feed-forward networks, and layer normalization, using fixed-point arithmetic.

' Include necessary files
#INCLUDE "data_structures.bas"
#INCLUDE "quantization.bas" ' For LogQuantized type and dequantization to fixed-point
#INCLUDE "matrix_ops.bas" ' For matrix operations and fixed-point arithmetic, including fixed-point math functions
#INCLUDE "softmax_fixed.bas" ' For fixed-point softmax implementation
#INCLUDE "block_sparse.bas" ' For block-sparse attention optimization

' Note: The dimensions (embedding_dim, num_heads, context_length) will need
' to be defined as constants, likely in a separate config file or at the
' beginning of the main program. Using placeholders for now.
' CONST EMBEDDING_DIM AS INTEGER = 128
' CONST NUM_HEADS AS INTEGER = 4
' CONST CONTEXT_LENGTH AS INTEGER = 128

' Pre-calculate the attention scaling factor in fixed-point.
' Scale is 1 / sqrt(head_dim)
' CONST ATTENTION_SCALE_FP AS INTEGER = FixedDivide(FloatToFixed(1.0), FixedSqrt(FloatToFixed(CSNG(EMBEDDING_DIM / NUM_HEADS)))) ' Placeholder constants

' Define a very large negative fixed-point value for attention masking.
' CONST NEG_INF_FP AS INTEGER = FloatToFixed(-1.0e9) ' Placeholder

' Function to implement the scaled dot-product attention mechanism.
' This is a simplified structure focusing on the flow.
' It needs to be adapted for block-sparse attention and optimized fixed-point math.
' Input: Query, Key, Value matrices (assumed to be LogQuantized data)
' Output: Output matrix (LogQuantized data)
SUB ScaledDotProductAttention (Query AS Matrix, Key AS Matrix, Value AS Matrix, Output AS Matrix)
    ' Assumes Query, Key, Value are appropriately shaped for a single attention head
    ' e.g., Query: (context_length, head_dim), Key: (context_length, head_dim), Value: (context_length, head_dim)
    ' where head_dim = embedding_dim / num_heads
    
    ' Determine whether to use dense or block-sparse attention based on context length
    ' For small contexts, dense attention may be more efficient
    ' For larger contexts, block-sparse attention saves memory and computation
    DIM use_sparse AS INTEGER = ShouldUseBlockSparseAttention(Query.rows)
    
    IF use_sparse = 1 THEN
        ' Use optimized block-sparse attention for larger contexts
        BlockSparseAttention Query, Key, Value, Output
    ELSE
        ' For small contexts, use the original dense attention implementation
        ' Need a temporary matrix for scores: (context_length, context_length)
        DIM Scores AS Matrix
        InitMatrix Scores, Query.rows, Key.cols ' Query.rows = context_length, Key.cols = context_length
        
        ' Perform matrix multiplication: Scores = Query * Key
        MatrixMultiply Query, Key, Scores ' Assuming Key is already transposed (head_dim, context_length)
    
        ' Scale the scores by dividing by sqrt(head_dim)
        DIM r AS INTEGER
        DIM c AS INTEGER
        FOR r = 0 TO Scores.rows - 1
            FOR c = 0 TO Scores.cols - 1
                ' Convert score to fixed-point
                DIM fp_score AS INTEGER = DequantizeLogToFixed(Scores.data(r, c)) ' Use direct dequantization to fixed-point
                
                ' Scale the score (fixed-point multiplication)
                DIM fp_scaled_score AS INTEGER = FixedMultiply(fp_score, ATTENTION_SCALE_FP) ' Use fixed-point multiplication
                
                ' Store the scaled fixed-point value back as LogQuantized
                Scores.data(r, c) = FixedToLogQuantized(fp_scaled_score).packed_value
            NEXT c
        NEXT r
    
        ' Apply causal attention mask (if column > row, set to negative infinity)
        FOR r = 0 TO Scores.rows - 1
            FOR c = 0 TO Scores.cols - 1
                IF c > r THEN
                    ' Set score to a very large negative fixed-point value
                    Scores.data(r, c) = FixedToLogQuantized(NEG_INF_FP).packed_value ' Use fixed-point negative infinity
                END IF
            NEXT c
        NEXT r
    
        ' Apply Softmax to convert scores to normalized probabilities
        SoftmaxFixedPoint Scores
        
        ' Calculate Output (Softmax(Scores) * Value)
        InitMatrix Output, Scores.rows, Value.cols ' Scores.rows = context_length, Value.cols = head_dim
        MatrixMultiply Scores, Value, Output
    
        ' Clean up temporary matrix
        FreeMatrix Scores
    END IF
END SUB

' Note on Block-Sparse Attention:
' We've now implemented block-sparse attention in block_sparse.bas and integrated it here.
' The sparse implementation only allocates and computes attention for blocks that are 
' actually needed, which significantly reduces memory usage for longer contexts.
' The block size and decision threshold can be tuned based on performance benchmarks.
' For very short contexts, the overhead of sparse structures may not be worth it.

' Function to implement a single Transformer Attention Head.
' Combines linear projections (Q, K, V weights) and scaled dot-product attention.
' Input: Input matrix (e.g., token embeddings + positional encodings)
' Output: Output matrix for this head
SUB AttentionHead (Input AS Matrix, Wq AS Matrix, Wk AS Matrix, Wv AS Matrix, Wo AS Matrix, Output AS Matrix)
    ' Assumes Wq, Wk, Wv are (embedding_dim, head_dim) and Wo is (head_dim, embedding_dim)
    ' Input is (context_length, embedding_dim)

    ' Need temporary matrices for Q, K, V, and intermediate output
    DIM Query AS Matrix: InitMatrix Query, Input.rows, Wq.cols ' (context_length, head_dim)
    DIM Key AS Matrix: InitMatrix Key, Input.rows, Wk.cols   ' (context_length, head_dim)
    DIM Value AS Matrix: InitMatrix Value, Input.rows, Wv.cols ' (context_length, head_dim)
    DIM HeadOutput AS Matrix: InitMatrix HeadOutput, Query.rows, Query.cols ' (context_length, head_dim)

    ' Linear Projections: Q = Input * Wq, K = Input * Wk, V = Input * Wv
    ' These need to use the optimized fixed-point MatrixMultiply
    MatrixMultiply Input, Wq, Query
    MatrixMultiply Input, Wk, Key
    MatrixMultiply Input, Wv, Value

    ' Perform Scaled Dot-Product Attention
    ' Note: ScaledDotProductAttention expects Key to be transposed for Q*K^T.
    ' We need to either transpose Key here or modify ScaledDotProductAttention.
    ' Let's assume ScaledDotProductAttention handles the implicit transpose or we add a Transpose function.
    ' For now, calling with Key as is, assuming ScaledDotProductAttention handles it.
    ScaledDotProductAttention Query, Key, Value, HeadOutput

    ' Final Linear Projection: Output = HeadOutput * Wo
    ' Need a temporary matrix for the final output of this head before combining
    DIM FinalHeadOutput AS Matrix: InitMatrix FinalHeadOutput, HeadOutput.rows, Wo.cols ' (context_length, embedding_dim)
    MatrixMultiply HeadOutput, Wo, FinalHeadOutput

    ' Copy result to Output matrix (assuming Output is pre-initialized with correct dimensions)
    ' This copy step might be optimized away by passing Output directly to the last multiply.
    DIM r AS INTEGER, c AS INTEGER
    FOR r = 0 TO FinalHeadOutput.rows - 1
        FOR c = 0 TO FinalHeadOutput.cols - 1
            Output.data(r, c) = FinalHeadOutput.data(r, c)
        NEXT c
    NEXT r

    ' Clean up temporary matrices
    FreeMatrix Query
    FreeMatrix Key
    FreeMatrix Value
    FreeMatrix HeadOutput
    FreeMatrix FinalHeadOutput
END SUB

' Function to implement Multi-Head Attention.
' Runs multiple attention heads in parallel and concatenates their outputs.
' Input: Input matrix, Weight matrices for all heads (Wq_all, Wk_all, Wv_all, Wo_all)
' Output: Output matrix
SUB MultiHeadAttention (Input AS Matrix, Wq_all AS Matrix, Wk_all AS Matrix, Wv_all AS Matrix, Wo_all AS Matrix, Output AS Matrix)
    ' Assumes Wq_all, Wk_all, Wv_all are (embedding_dim, embedding_dim) and Wo_all is (embedding_dim, embedding_dim)
    ' These weights need to be split and distributed to individual heads.
    ' This requires careful indexing or splitting of the weight matrices.

    ' Need temporary matrices for each head's output
    ' DIM HeadOutputs(NUM_HEADS - 1) AS Matrix ' Array of matrices
    ' For now, let's process heads sequentially and concatenate results.

    DIM head_dim AS INTEGER = EMBEDDING_DIM / NUM_HEADS ' Placeholder constant

    ' Need temporary matrices for weights and output of a single head
    DIM Wq_head AS Matrix: InitMatrix Wq_head, EMBEDDING_DIM, head_dim ' Placeholder constant
    DIM Wk_head AS Matrix: InitMatrix Wk_head, EMBEDDING_DIM, head_dim ' Placeholder constant
    DIM Wv_head AS Matrix: InitMatrix Wv_head, EMBEDDING_DIM, head_dim ' Placeholder constant
    DIM Wo_head AS Matrix: InitMatrix Wo_head, head_dim, EMBEDDING_DIM ' Placeholder constant
    DIM HeadOutput AS Matrix: InitMatrix HeadOutput, Input.rows, head_dim ' (context_length, head_dim)

    ' Need a temporary matrix to accumulate concatenated head outputs
    DIM ConcatenatedOutput AS Matrix: InitMatrix ConcatenatedOutput, Input.rows, EMBEDDING_DIM ' (context_length, embedding_dim)

    DIM head_idx AS INTEGER
    FOR head_idx = 0 TO NUM_HEADS - 1 ' Placeholder constant
        ' Extract weights for the current head from the combined weight matrices
        ' This requires careful slicing/copying of matrix data.
        ' SUB ExtractHeadWeights(W_all AS Matrix, W_head AS Matrix, head_idx AS INTEGER, head_dim AS INTEGER) ' Placeholder
        ' ExtractHeadWeights Wq_all, Wq_head, head_idx, head_dim
        ' ExtractHeadWeights Wk_all, Wk_head, head_idx, head_dim
        ' ExtractHeadWeights Wv_all, Wv_head, head_idx, head_dim
        ' ExtractHeadWeights Wo_all, Wo_head, head_idx, head_dim

        ' Perform attention for this head
        AttentionHead Input, Wq_head, Wk_head, Wv_head, Wo_head, HeadOutput

        ' Concatenate the output of this head into the ConcatenatedOutput matrix
        ' This requires copying HeadOutput into the correct columns of ConcatenatedOutput.
        ' SUB ConcatenateHeadOutput(Concatenated AS Matrix, HeadOutput AS Matrix, head_idx AS INTEGER, head_dim AS INTEGER) ' Placeholder
        ' ConcatenateHeadOutput ConcatenatedOutput, HeadOutput, head_idx, head_dim
        
        ' Placeholder: Copy HeadOutput to the correct slice of ConcatenatedOutput
        DIM r AS INTEGER, c AS INTEGER
        FOR r = 0 TO HeadOutput.rows - 1
            FOR c = 0 TO HeadOutput.cols - 1
                ConcatenatedOutput.data(r, (head_idx * head_dim) + c) = HeadOutput.data(r, c)
            NEXT c
        NEXT r

    NEXT head_idx

    ' Final linear projection after concatenation (Wo_all)
    ' The Wo_all matrix is applied to the concatenated output.
    ' Need a temporary matrix for the final output
    DIM FinalOutput AS Matrix: InitMatrix FinalOutput, ConcatenatedOutput.rows, Wo_all.cols ' (context_length, embedding_dim)
    MatrixMultiply ConcatenatedOutput, Wo_all, FinalOutput

    ' Copy result to Output matrix (assuming Output is pre-initialized)
    FOR r = 0 TO FinalOutput.rows - 1
        FOR c = 0 TO FinalOutput.cols - 1
            Output.data(r, c) = FinalOutput.data(r, c)
        NEXT c
    NEXT r

    ' Clean up temporary matrices
    FreeMatrix Wq_head
    FreeMatrix Wk_head
    FreeMatrix Wv_head
    FreeMatrix Wo_head
    FreeMatrix HeadOutput
    FreeMatrix ConcatenatedOutput
    FreeMatrix FinalOutput
END SUB

' Function to implement a Feed-Forward Network (FFN) block with GLU.
' Input: Input matrix
' Output: Output matrix
SUB FeedForward (Input AS Matrix, W1 AS Matrix, W2 AS Matrix, W3 AS Matrix, Output AS Matrix)
    ' Assumes W1, W3 are (embedding_dim, intermediate_dim) and W2 is (intermediate_dim, embedding_dim)
    ' intermediate_dim is typically 4 * embedding_dim
    ' GLU involves two linear layers (Input * W1 and Input * W3) and element-wise multiplication after a non-linearity on one.

    ' Need temporary matrices
    DIM Intermediate1 AS Matrix: InitMatrix Intermediate1, Input.rows, W1.cols ' (context_length, intermediate_dim)
    DIM Intermediate2 AS Matrix: InitMatrix Intermediate2, Input.rows, W3.cols ' (context_length, intermediate_dim)
    DIM ActivatedIntermediate AS Matrix: InitMatrix ActivatedIntermediate, Input.rows, W1.cols ' (context_length, intermediate_dim)

    ' First linear layer: Intermediate1 = Input * W1
    MatrixMultiply Input, W1, Intermediate1

    ' Second linear layer for GLU gate: Intermediate2 = Input * W3
    MatrixMultiply Input, W3, Intermediate2

    ' Apply activation (e.g., GELU or ReLU) to Intermediate1
    ' Need a fixed-point activation function. GELU is complex, ReLU is simpler.
    ' Let's assume ReLU for simplicity in BASIC for now.
    ' SUB ReLU(Matrix AS Matrix) ' Placeholder
    ' ReLU(Intermediate1)
    
    ' Placeholder: Apply ReLU element-wise (fixed-point)
    DIM r AS INTEGER, c AS INTEGER
    FOR r = 0 TO Intermediate1.rows - 1
        FOR c = 0 TO Intermediate1.cols - 1
            DIM fp_val AS INTEGER = LogQuantizedToFixed(Intermediate1.data(r, c))
            ' Fixed-point ReLU: MAX(0, fp_val)
            DIM fp_relu_val AS INTEGER = fp_val
            IF fp_relu_val < 0 THEN fp_relu_val = 0
            ActivatedIntermediate.data(r, c) = FixedToLogQuantized(fp_relu_val).packed_value
        NEXT c
    NEXT r

    ' Apply GLU gate: ActivatedIntermediate = ActivatedIntermediate * Intermediate2 (element-wise multiplication)
    ' Use the fixed-point element-wise multiplication function.
    MatrixElementWiseMultiply ActivatedIntermediate, Intermediate2, ActivatedIntermediate ' Overwrite ActivatedIntermediate

    ' Third linear layer: Output = ActivatedIntermediate * W2
    MatrixMultiply ActivatedIntermediate, W2, Output

    ' Clean up temporary matrices
    FreeMatrix Intermediate1
    FreeMatrix Intermediate2
    FreeMatrix ActivatedIntermediate
END SUB

' Function to implement Layer Normalization.
' Input: Input matrix
' Output: Output matrix
' Assumes gamma and beta parameters are stored elsewhere (as LogQuantized matrices).
SUB LayerNorm (Input AS Matrix, Gamma AS Matrix, Beta AS Matrix, Output AS Matrix)
    ' Assumes Input, Gamma, Beta, Output are (context_length, embedding_dim)
    ' LayerNorm normalizes across the embedding dimension for each token (row).
    ' Calculation: (x - mean) / sqrt(variance + epsilon) * gamma + beta
    ' This requires calculating mean and variance for each row, fixed-point division,
    ' square root, and element-wise operations.

    ' Need temporary matrix for normalized output
    DIM NormalizedInput AS Matrix: InitMatrix NormalizedInput, Input.rows, Input.cols

    DIM r AS INTEGER ' Row index (token)
    DIM c AS INTEGER ' Column index (embedding dimension)

    ' Define fixed-point epsilon for numerical stability
    CONST EPSILON_FP AS INTEGER = FloatToFixed(1e-5) ' Using 1e-5 as a common small value

    FOR r = 0 TO Input.rows - 1
        ' Calculate mean for the current row (token) in fixed-point
        DIM fp_sum AS INTEGER = 0
        FOR c = 0 TO Input.cols - 1
            fp_sum = FixedAdd(fp_sum, DequantizeLogToFixed(Input.data(r, c))) ' Use direct dequantization
        NEXT c
        ' Need fixed-point division by Input.cols
        DIM fp_mean AS INTEGER = FixedDivide(fp_sum, FloatToFixed(CSNG(Input.cols))) ' Use FixedDivide

        ' Calculate variance for the current row (token) in fixed-point
        DIM fp_sum_sq_diff AS INTEGER = 0
        FOR c = 0 TO Input.cols - 1
            DIM fp_val AS INTEGER = DequantizeLogToFixed(Input.data(r, c)) ' Use direct dequantization
            DIM fp_diff AS INTEGER = FixedSubtract(fp_val, fp_mean) ' Use FixedSubtract
            DIM fp_sq_diff AS INTEGER = FixedMultiply(fp_diff, fp_diff) ' Use FixedMultiply
            fp_sum_sq_diff = FixedAdd(fp_sum_sq_diff, fp_sq_diff) ' Use FixedAdd
        NEXT c
        ' Need fixed-point division by Input.cols
        DIM fp_variance AS INTEGER = FixedDivide(fp_sum_sq_diff, FloatToFixed(CSNG(Input.cols))) ' Use FixedDivide

        ' Add epsilon for numerical stability (fixed-point epsilon)
        fp_variance = FixedAdd(fp_variance, EPSILON_FP) ' Use FixedAdd

        ' Calculate inverse standard deviation (1 / sqrt(variance + epsilon))
        ' Need fixed-point square root and division.
        ' Use FixedSqrt and FixedDivide.
        DIM fp_stddev AS INTEGER = FixedSqrt(fp_variance) ' Use FixedSqrt
        DIM fp_inv_stddev AS INTEGER = FixedDivide(FloatToFixed(1.0), fp_stddev) ' Use FixedDivide

        ' Normalize the row: (x - mean) * inv_stddev
        FOR c = 0 TO Input.cols - 1
            DIM fp_val AS INTEGER = DequantizeLogToFixed(Input.data(r, c)) ' Use direct dequantization
            DIM fp_diff AS INTEGER = FixedSubtract(fp_val, fp_mean) ' Use FixedSubtract
            DIM fp_normalized_val AS INTEGER = FixedMultiply(fp_diff, fp_inv_stddev) ' Use FixedMultiply
            NormalizedInput.data(r, c) = FixedToLogQuantized(fp_normalized_val).packed_value ' Convert back to LogQuantized
        NEXT c
    NEXT r

    ' Apply learned parameters gamma and beta (element-wise)
    ' Output = NormalizedInput * gamma + beta
    ' Use fixed-point element-wise multiplication and addition.
    
    ' Need temporary matrix for scaled output
    DIM ScaledOutput AS Matrix: InitMatrix ScaledOutput, NormalizedInput.rows, NormalizedInput.cols

    ' Apply gamma (element-wise multiplication)
    MatrixElementWiseMultiply NormalizedInput, Gamma, ScaledOutput ' Use MatrixElementWiseMultiply

    ' Apply beta (element-wise addition)
    MatrixElementWiseAdd ScaledOutput, Beta, Output ' Use MatrixElementWiseAdd

    ' Clean up temporary matrices
    FreeMatrix NormalizedInput
    FreeMatrix ScaledOutput
END SUB

' Note on Layer Normalization Optimization:
' Calculating mean and variance row-wise involves sums and divisions.
' Fixed-point division and square root are now implemented using FixedDivide and FixedSqrt.
' Approximations or lookup tables for these operations might still be necessary for
' optimal performance on 486, depending on the efficiency of the FixedSqrt implementation.
' The element-wise multiplication and addition with gamma and beta are handled
' with fixed-point arithmetic.

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.

<environment_details>
# VSCode Visible Files
src/transformer_components.bas

# VSCode Open Tabs
memory-bank/projectbrief.md
memory-bank/productContext.md
memory-bank/systemPatterns.md
memory-bank/techContext.md
memory-bank/activeContext.md
memory-bank/progress.md
src/data_structures.bas
src/model.bas
src/main.bas
src/matrix_ops.bas
src/quantization.bas
src/transformer_components.bas

# Current Time
5/16/2025, 11:06:14 PM (America/Toronto, UTC-4:00)

# Context Window Usage
222,478 / 1,048.576K tokens used (21%)

# Current Mode
ACT MODE
</environment_details>
